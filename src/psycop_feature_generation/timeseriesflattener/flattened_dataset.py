"""Takes a time-series and flattens it into a set of prediction times with
describing values."""
import datetime as dt
import os
import random
import time
from collections.abc import Callable
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import tqdm
from catalogue import Registry  # noqa # pylint: disable=unused-import
from dask.diagnostics import ProgressBar
from pandas import DataFrame
from wasabi import Printer, msg

from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    AnySpec,
    OutcomeSpec,
    PredictorSpec,
    TemporalSpec,
)
from psycop_feature_generation.timeseriesflattener.flattened_ds_validator import (
    ValidateInitFlattenedDataset,
)
from psycop_feature_generation.timeseriesflattener.resolve_multiple_functions import (
    resolve_multiple_fns,
)
from psycop_feature_generation.utils import load_dataset_from_file, write_df_to_file

ProgressBar().register()


class FlattenedDataset:  # pylint: disable=too-many-instance-attributes
    """Turn a set of time-series into tabular prediction-time data."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        prediction_times_df: DataFrame,
        id_col_name: str = "dw_ek_borger",
        timestamp_col_name: str = "timestamp",
        predictor_col_name_prefix: str = "pred",
        outcome_col_name_prefix: str = "outc",
        n_workers: int = 60,
        feature_cache_dir: Optional[Path] = None,
    ):
        """Class containing a time-series, flattened.

        A 'flattened' version is a tabular representation for each prediction time.
        A prediction time is every timestamp where you want your model to issue a prediction.

        E.g if you have a prediction_times_df:

        id_col_name | timestamp_col_name
        1           | 2022-01-10
        1           | 2022-01-12
        1           | 2022-01-15

        And a time-series of blood-pressure values as an outcome:
        id_col_name | timestamp_col_name | blood_pressure_value
        1           | 2022-01-09         | 120
        1           | 2022-01-14         | 140

        Then you can "flatten" the outcome into a new df, with a row for each of your prediction times:
        id_col_name | timestamp_col_name | latest_blood_pressure_within_24h
        1           | 2022-01-10         | 120
        1           | 2022-01-12         | NA
        1           | 2022-01-15         | 140

        Args:
            prediction_times_df (DataFrame): Dataframe with prediction times, required cols: patient_id, .
            timestamp_col_name (str, optional): Column name name for timestamps. Is used across outcomes and predictors. Defaults to "timestamp".
            id_col_name (str, optional): Column namn name for patients ids. Is used across outcome and predictors. Defaults to "dw_ek_borger".
            predictor_col_name_prefix (str, optional): Prefix for predictor col names. Defaults to "pred_".
            outcome_col_name_prefix (str, optional): Prefix for outcome col names. Defaults to "outc_".
            n_workers (int): Number of subprocesses to spawn for parallelization. Defaults to 60.
            feature_cache_dir (Path): Path to cache directory for feature dataframes. Defaults to None.

        Raises:
            ValueError: If timestamp_col_name or id_col_name is not in prediction_times_df
            ValueError: If timestamp_col_name is not and could not be converted to datetime
        """
        self.n_workers = n_workers

        self.timestamp_col_name = timestamp_col_name
        self.id_col_name = id_col_name
        self.predictor_col_name_prefix = predictor_col_name_prefix
        self.outcome_col_name_prefix = outcome_col_name_prefix
        self.pred_time_uuid_col_name = "prediction_time_uuid"

        if feature_cache_dir:
            self.feature_cache_dir = feature_cache_dir
            if not self.feature_cache_dir.exists():
                self.feature_cache_dir.mkdir()

        self.n_uuids = len(prediction_times_df)

        self.msg = Printer(timestamp=True)

        if "value" in prediction_times_df.columns:
            prediction_times_df.drop("value", axis=1, inplace=True)

        self.df = prediction_times_df

        ValidateInitFlattenedDataset(
            df=self.df,
            timestamp_col_name=self.timestamp_col_name,
            id_col_name=self.id_col_name,
        ).validate_dataset()

        # Create pred_time_uuid_columne
        self.df[self.pred_time_uuid_col_name] = self.df[self.id_col_name].astype(
            str,
        ) + self.df[self.timestamp_col_name].dt.strftime("-%Y-%m-%d-%H-%M-%S")

    def _load_most_recent_df_matching_pattern(
        self,
        dir_path: Path,
        file_pattern: str,
        file_suffix: str,
    ) -> DataFrame:
        """Load most recent df matching pattern.

        Args:
            file_pattern (str): Pattern to match
            file_suffix (str, optional): File suffix to match.

        Returns:
            DataFrame: DataFrame matching pattern

        Raises:
            FileNotFoundError: If no file matching pattern is found
        """
        files_with_suffix = list(dir_path.glob(f"*{file_pattern}*.{file_suffix}"))

        if len(files_with_suffix) == 0:
            raise FileNotFoundError(f"No files matching pattern {file_pattern} found")

        path_of_most_recent_file = max(files_with_suffix, key=os.path.getctime)

        return load_dataset_from_file(
            file_path=path_of_most_recent_file,
        )

    def _load_cached_df_and_expand_fallback(
        self,
        file_pattern: str,
        file_suffix: str,
        fallback: Any,
        full_col_str: str,
    ) -> pd.DataFrame:
        """Load most recent df matching pattern, and expand fallback column.

        Args:
            file_pattern (str): File pattern to search for
            file_suffix (str): File suffix to search for
            fallback (Any): Fallback value
            full_col_str (str): Full column name for values

        Returns:
            DataFrame: DataFrame with fallback column expanded
        """
        df = self._load_most_recent_df_matching_pattern(
            dir_path=self.feature_cache_dir,
            file_pattern=file_pattern,
            file_suffix=file_suffix,
        )

        # Expand fallback column
        df = pd.merge(
            left=self.df[self.pred_time_uuid_col_name],
            right=df,
            how="left",
            on=self.pred_time_uuid_col_name,
            validate="m:1",
        )

        df[full_col_str] = df[full_col_str].fillna(fallback)

        return df

    @staticmethod
    def _flatten_temporal_values_to_df(  # noqa pylint: disable=too-many-locals
        prediction_times_with_uuid_df: DataFrame,
        output_spec: TemporalSpec,
        id_col_name: str,
        timestamp_col_name: str,
        pred_time_uuid_col_name: str,
        verbose: bool = False,
    ) -> DataFrame:

        """Create a dataframe with flattened values (either predictor or
        outcome depending on the value of "direction").

        Args:
            prediction_times_with_uuid_df (DataFrame): Dataframe with id_col and
                timestamps for each prediction time.
            output_spec (Union[OutcomeSpec, PredictorSpec]): Specification of the output column.
            id_col_name (str): Name of id_column in prediction_times_with_uuid_df and
                df. Required because this is a static method.
            timestamp_col_name (str): Name of timestamp column in
                prediction_times_with_uuid_df and df. Required because this is a
                static method.
            pred_time_uuid_col_name (str): Name of uuid column in
                prediction_times_with_uuid_df. Required because this is a static method.
            verbose (bool, optional): Whether to print progress.


        Returns:
            DataFrame
        """
        for col_name in (timestamp_col_name, id_col_name):
            if col_name not in output_spec.values_df.columns:
                raise ValueError(
                    f"{col_name} does not exist in df_prediction_times, change the df or set another argument",
                )

        # Generate df with one row for each prediction time x event time combination
        # Drop dw_ek_borger for faster merge
        df = pd.merge(
            left=prediction_times_with_uuid_df,
            right=output_spec.values_df,
            how="left",
            on=id_col_name,
            suffixes=("_pred", "_val"),
            validate="m:m",
        ).drop("dw_ek_borger", axis=1)

        # Drop prediction times without event times within interval days
        if isinstance(output_spec, OutcomeSpec):
            direction = "ahead"
        elif isinstance(output_spec, PredictorSpec):
            direction = "behind"
        else:
            raise ValueError(f"Unknown output_spec type {type(output_spec)}")

        df = FlattenedDataset._drop_records_outside_interval_days(
            df,
            direction=direction,
            interval_days=output_spec.interval_days,
            timestamp_pred_colname="timestamp_pred",
            timestamp_value_colname="timestamp_val",
        )

        # Add back prediction times that don't have a value, and fill them with fallback
        df = FlattenedDataset._add_back_prediction_times_without_value(
            df=df,
            pred_times_with_uuid=prediction_times_with_uuid_df,
            pred_time_uuid_colname=pred_time_uuid_col_name,
        ).fillna(output_spec.fallback)

        df["timestamp_val"].replace({output_spec.fallback: pd.NaT}, inplace=True)

        df = FlattenedDataset._resolve_multiple_values_within_interval_days(
            resolve_multiple=output_spec.resolve_multiple_fn,
            df=df,
            pred_time_uuid_colname=pred_time_uuid_col_name,
        )

        # If resolve_multiple generates empty values,
        # e.g. when there is only one prediction_time within look_ahead window for slope calculation,
        # replace with NaN

        # Rename column
        df.rename(columns={"value": output_spec.get_col_str()}, inplace=True)

        # Find value_cols and add fallback to them
        df[output_spec.get_col_str()] = df[output_spec.get_col_str()].fillna(
            output_spec.fallback,
            inplace=False,
        )

        if verbose:
            msg.good(
                f"Returning {df.shape[0]} rows of flattened dataframe for {output_spec.get_col_str()}",
            )

        return df[[pred_time_uuid_col_name, output_spec.get_col_str()]]

    def _generate_values_for_cache_checking(
        self,
        output_spec: TemporalSpec,
        value_col_str: str,
        n_to_generate: int = 100_000,
    ):
        generated_df = pd.DataFrame({value_col_str: []})

        # Check that some values in generated_df differ from fallback
        # Otherwise, comparison to cache is meaningless
        n_trials = 0

        while not any(
            generated_df[value_col_str] != output_spec.fallback,
        ):
            if n_trials != 0:
                self.msg.info(
                    f"{value_col_str[20]}, {n_trials}: Generated_df was all fallback values, regenerating",
                )

            n_to_generate = int(min(n_to_generate, len(self.df)))

            generated_df = self._flatten_temporal_values_to_df(
                prediction_times_with_uuid_df=self.df.sample(
                    n=n_to_generate,
                    replace=False,
                ),
                id_col_name=self.id_col_name,
                timestamp_col_name=self.timestamp_col_name,
                pred_time_uuid_col_name=self.pred_time_uuid_col_name,
                output_spec=output_spec,
            ).dropna()

            # Fallback values are not interesting for cache hit. If they exist in generated_df, they should be dropped
            # in the cache. Saves on storage. Don't use them to check if cache is hit.
            if not np.isnan(output_spec.fallback):
                generated_df = generated_df[
                    generated_df[value_col_str] != output_spec.fallback
                ]

            n_to_generate = (
                n_to_generate**1.5
            )  # Increase n_to_generate by 1.5x each time to increase chance of non_fallback values

            n_trials += 1

        return generated_df

    def _cache_is_hit(
        self,
        output_spec: Union[PredictorSpec, PredictorSpec],
        file_pattern: str,
        file_suffix: str,
    ) -> bool:
        """Check if cache is hit.

        Args:
            kwargs_dict (dict): dictionary of kwargs
            file_pattern (str): File pattern to match. Looks for *file_pattern* in cache dir.
            e.g. "*feature_name*_uuids*.file_suffix"
            full_col_str (str): Full column string. e.g. "feature_name_ahead_interval_days_resolve_multiple_fallback"
            file_suffix (str): File suffix to match. e.g. "csv"

        Returns:
            bool: True if cache is hit, False otherwise
        """
        # Check that file exists
        file_pattern_hits = list(
            self.feature_cache_dir.glob(f"*{file_pattern}*.{file_suffix}"),
        )

        if len(file_pattern_hits) == 0:
            self.msg.info(f"Cache miss, {file_pattern} didn't exist")
            return False

        value_col_str = output_spec.get_col_str()

        # Check that file contents match expected
        # NAs are not interesting when comparing if computed values are identical
        cache_df = self._load_most_recent_df_matching_pattern(
            dir_path=self.feature_cache_dir,
            file_pattern=file_pattern,
            file_suffix=file_suffix,
        )

        generated_df = self._generate_values_for_cache_checking(
            output_spec=output_spec,
            value_col_str=value_col_str,
        )

        cached_suffix = "_c"
        generated_suffix = "_g"

        # We frequently hit rounding errors with cache hits, so we round to 3 decimal places
        generated_df[value_col_str] = generated_df[value_col_str].round(3)
        cache_df[value_col_str] = cache_df[value_col_str].round(3)

        # Merge cache_df onto generated_df
        merged_df = pd.merge(
            left=generated_df,
            right=cache_df,
            how="left",
            on=self.pred_time_uuid_col_name,
            suffixes=(generated_suffix, cached_suffix),
            validate="1:1",
            indicator=True,
        )

        # Check that all rows in generated_df are in cache_df
        if not merged_df[value_col_str + generated_suffix].equals(
            merged_df[value_col_str + cached_suffix],
        ):
            self.msg.info(f"Cache miss, computed values didn't match {file_pattern}")

            # Keep this variable for easier inspection
            unequal_rows = merged_df[  # pylint: disable=unused-variable
                merged_df[value_col_str + generated_suffix]
                != merged_df[value_col_str + cached_suffix]
            ]

            return False

        # If all checks passed, return true
        msg.good(f"Cache hit for {value_col_str}")
        return True

    def _write_feature_to_cache(
        self,
        values_df: pd.DataFrame,
        predictor_spec: PredictorSpec,
        file_name: str,
    ):
        """Write feature to cache."""
        out_df = values_df

        # Drop rows containing fallback, since it's non-informative
        out_df = out_df[
            out_df[predictor_spec.get_col_str()] != predictor_spec.fallback
        ].dropna()

        # Write df to cache
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Write df to cache
        write_df_to_file(
            df=out_df,
            file_path=self.feature_cache_dir / f"{file_name}_{timestamp}.parquet",
        )

    def _get_feature(
        self,
        feature_spec: AnySpec,
        file_suffix: str = "parquet",
    ) -> pd.DataFrame:
        """Get feature. Either load from cache, or generate if necessary.

        Args:
            file_suffix (str, optional): File suffix for the cache lookup. Defaults to "parquet".

        Returns:
            pd.DataFrame: Feature
        """
        file_name = f"{feature_spec.get_col_str()}_{self.n_uuids}_uuids"

        if hasattr(self, "feature_cache_dir"):
            if self._cache_is_hit(
                file_pattern=file_name,
                output_spec=feature_spec,
                file_suffix="parquet",
            ):
                df = self._load_cached_df_and_expand_fallback(
                    file_pattern=file_name,
                    full_col_str=feature_spec.get_col_str(),
                    fallback=feature_spec.fallback,
                    file_suffix=file_suffix,
                )

                return df.set_index(keys=self.pred_time_uuid_col_name).sort_index()
        else:
            msg.info("No cache dir specified, not attempting load")

        df = self._flatten_temporal_values_to_df(
            prediction_times_with_uuid_df=self.df[
                [
                    self.pred_time_uuid_col_name,
                    self.id_col_name,
                    self.timestamp_col_name,
                ]
            ],
            id_col_name=self.id_col_name,
            timestamp_col_name=self.timestamp_col_name,
            pred_time_uuid_col_name=self.pred_time_uuid_col_name,
            output_spec=feature_spec,
        )

        # Write df to cache if exists
        if hasattr(self, "feature_cache_dir"):
            self._write_feature_to_cache(
                predictor_spec=feature_spec,
                file_name=file_name,
                values_df=df,
            )

        return (
            df[[self.pred_time_uuid_col_name, feature_spec.get_col_str()]]
            .set_index(keys=self.pred_time_uuid_col_name)
            .sort_index()
        )

    def _check_dfs_have_same_lengths(self, dfs: list[pd.DataFrame]):
        """Check that all dfs have the same length."""
        df_lengths = 0

        for feature_df in dfs:
            if df_lengths == 0:
                df_lengths = len(feature_df)
            else:
                if df_lengths != len(feature_df):
                    raise ValueError("Dataframes are not of equal length")

    def _check_dfs_have_identical_indexes(self, dfs: list[pd.DataFrame]):
        """Randomly sample 50 positions in each df and check that their indeces
        are identical.

        This checks that all the dataframes are aligned before
        concatenation.
        """
        for _ in range(50):
            random_index = random.randint(0, len(dfs[0]) - 1)
            for feature_df in dfs[1:]:
                if dfs[0].index[random_index] != feature_df.index[random_index]:
                    raise ValueError(
                        "Dataframes are not of identical index. Were they correctly aligned before concatenation?",
                    )

    def _concatenate_flattened_timeseries(
        self,
        flattened_predictor_dfs: list[pd.DataFrame],
    ):
        """Concatenate flattened predictor dfs."""

        msg = Printer(timestamp=True)
        msg.info(
            "Starting concatenation. Will take some time on performant systems, e.g. 30s for 100 features. This is normal.",
        )

        start_time = time.time()

        # Check that dfs are ready for concatenation. Concatenation doesn't merge on IDs, but is **much** faster.
        # We thus require that a) the dfs are sorted so each row matches the same ID and b) that each df has a row
        # for each id.
        self._check_dfs_have_identical_indexes(dfs=flattened_predictor_dfs)
        self._check_dfs_have_same_lengths(dfs=flattened_predictor_dfs)

        # If so, ready for concatenation. Reset index to be ready for the merge at the end.
        new_features = pd.concat(
            objs=flattened_predictor_dfs,
            axis=1,
        ).reset_index()

        end_time = time.time()

        msg.info(f"Concatenation took {round(end_time - start_time, 3)} seconds")

        msg.info("Merging with original df")
        self.df = self.df.merge(right=new_features, on=self.pred_time_uuid_col_name)

    def add_temporal_predictors_from_pred_specs(  # pylint: disable=too-many-branches
        self,
        predictor_specs: list[PredictorSpec],
    ):
        """Add predictors to the flattened dataframe from a list."""

        # Shuffle predictor specs to avoid IO contention
        random.shuffle(predictor_specs)

        with Pool(self.n_workers) as p:
            flattened_predictor_dfs = list(
                tqdm.tqdm(
                    p.imap(func=self._get_feature, iterable=predictor_specs),
                    total=len(predictor_specs),
                ),
            )

        msg.info("Feature generation complete, concatenating")

        self._concatenate_flattened_timeseries(
            flattened_predictor_dfs=flattened_predictor_dfs,
        )

    def add_age_and_birth_year(
        self,
        id2date_of_birth: DataFrame,
        input_date_of_birth_col_name: Optional[str] = "date_of_birth",
        output_prefix: str = "pred",
        birth_year_as_predictor: bool = False,
    ):
        """Add age at prediction time and patient's birth year to each prediction time.

        Args:
            id2date_of_birth (DataFrame): Two columns, id and date_of_birth.
            input_date_of_birth_col_name (str, optional): Name of the date_of_birth column in id2date_of_birth.
                Defaults to "date_of_birth".
            output_prefix (str, optional): Prefix for the output column. Defaults to "pred".
            birth_year_as_predictor (bool, optional): Whether to add birth year as a predictor. Defaults to False.

        Raises:
            ValueError: _description_
        """
        if id2date_of_birth[input_date_of_birth_col_name].dtype != "<M8[ns]":
            try:
                id2date_of_birth[input_date_of_birth_col_name] = pd.to_datetime(
                    id2date_of_birth[input_date_of_birth_col_name],
                    format="%Y-%m-%d",
                )
            except Exception as e:
                raise ValueError(
                    f"Conversion of {input_date_of_birth_col_name} to datetime failed, doesn't match format %Y-%m-%d. Recommend converting to datetime before adding.",
                ) from e

        output_age_col_name = f"{output_prefix}_age_in_years"

        self.add_static_info(
            static_spec=AnySpec(
                values_df=id2date_of_birth,
                input_col_name_override=input_date_of_birth_col_name,
                prefix=output_prefix,
                # We typically don't want to use date of birth as a predictor,
                # but might want to use transformations - e.g. "year of birth" or "age at prediction time".
                feature_name=input_date_of_birth_col_name,
            ),
        )

        data_of_birth_col_name = f"{output_prefix}_{input_date_of_birth_col_name}"

        self.df[output_age_col_name] = (
            (self.df[self.timestamp_col_name] - self.df[data_of_birth_col_name]).dt.days
            / (365.25)
        ).round(2)

        if birth_year_as_predictor:
            # Convert datetime to year
            self.df["pred_birth_year"] = self.df[data_of_birth_col_name].dt.year

    def add_static_info(
        self,
        static_spec: AnySpec,
    ):
        """Add static info to each prediction time, e.g. age, sex etc.

        Args:
            static_spec (StaticSpec): Specification for the static info to add.

        Raises:
            ValueError: If input_col_name does not match a column in info_df.
        """

        # Try to infer value col name if not provided
        if static_spec.input_col_name_override is None:
            possible_value_cols = [
                col
                for col in static_spec.values_df.columns
                if col not in self.id_col_name
            ]

            if len(possible_value_cols) == 1:
                value_col_name = possible_value_cols[0]
            elif len(possible_value_cols) > 1:
                raise ValueError(
                    f"Only one value column can be added to static info, found multiple: {possible_value_cols}",
                )
            elif len(possible_value_cols) == 0:
                raise ValueError(
                    "No value column found in spec.df, please check.",
                )
        else:
            value_col_name = static_spec.input_col_name_override

        if static_spec.feature_name is None:
            output_col_name = f"{static_spec.prefix}_{value_col_name}"
        elif static_spec.feature_name:
            output_col_name = f"{static_spec.prefix}_{static_spec.feature_name}"

        df = pd.DataFrame(
            {
                self.id_col_name: static_spec.values_df[self.id_col_name],
                output_col_name: static_spec.values_df[value_col_name],
            },
        )

        self.df = pd.merge(
            self.df,
            df,
            how="left",
            on=self.id_col_name,
            suffixes=("", ""),
            validate="m:1",
        )

    def _add_incident_outcome(
        self,
        outcome_spec: OutcomeSpec,
    ):
        """Add incident outcomes.

        Can be done vectorized, hence the separate function.
        """
        prediction_timestamp_col_name = f"{self.timestamp_col_name}_prediction"
        outcome_timestamp_col_name = f"{self.timestamp_col_name}_outcome"

        df = pd.merge(
            self.df,
            outcome_spec.values_df,
            how="left",
            on=self.id_col_name,
            suffixes=("_prediction", "_outcome"),
            validate="m:1",
        )

        df = df.drop(
            df[
                df[outcome_timestamp_col_name] < df[prediction_timestamp_col_name]
            ].index,
        )

        if outcome_spec.is_dichotomous():
            df[outcome_spec.get_col_str()] = (
                df[prediction_timestamp_col_name]
                + timedelta(days=outcome_spec.interval_days)
                > df[outcome_timestamp_col_name]
            ).astype(int)

        df.rename(
            {prediction_timestamp_col_name: "timestamp"},
            axis=1,
            inplace=True,
        )
        df.drop([outcome_timestamp_col_name], axis=1, inplace=True)

        df.drop(["value"], axis=1, inplace=True)

        self.df = df

    def add_temporal_outcome(
        self,
        output_spec: OutcomeSpec,
    ):
        """Add an outcome-column to the dataset.

        Args:
            output_spec (OutcomeSpec): OutcomeSpec object.
        """

        if output_spec.incident:
            self._add_incident_outcome(
                outcome_spec=output_spec,
            )

        else:
            self.add_temporal_col_to_flattened_dataset(
                output_spec=output_spec,
            )

    def add_temporal_predictor(
        self,
        output_spec: PredictorSpec,
    ):
        """Add a column with predictor values to the flattened dataset (e.g.
        "average value of bloodsample within n days").

        Args:
            output_spec (Union[PredictorSpec]): Specification of the output column.
        """
        self.add_temporal_col_to_flattened_dataset(
            output_spec=output_spec,
        )

    def add_temporal_col_to_flattened_dataset(
        self,
        output_spec: TemporalSpec,
    ):
        """Add a column to the dataset (either predictor or outcome depending
        on the value of "direction").

        Args:
            output_spec (Union[OutcomeSpec, PredictorSpec]): Specification of the output column.
        """
        timestamp_col_type = output_spec.values_df[self.timestamp_col_name].dtype

        if timestamp_col_type not in ("Timestamp", "datetime64[ns]"):
            # Convert dtype to timestamp
            raise ValueError(
                f"{self.timestamp_col_name} is of type {timestamp_col_type}, not 'Timestamp' from Pandas. Will cause problems. Convert before initialising FlattenedDataset.",
            )

        df = FlattenedDataset._flatten_temporal_values_to_df(
            prediction_times_with_uuid_df=self.df[
                [
                    self.id_col_name,
                    self.timestamp_col_name,
                    self.pred_time_uuid_col_name,
                ]
            ],
            output_spec=output_spec,
            id_col_name=self.id_col_name,
            timestamp_col_name=self.timestamp_col_name,
            pred_time_uuid_col_name=self.pred_time_uuid_col_name,
        )

        self.df = self.df.merge(
            right=df,
            on=self.pred_time_uuid_col_name,
            validate="1:1",
        )

    @staticmethod
    def _add_back_prediction_times_without_value(
        df: DataFrame,
        pred_times_with_uuid: DataFrame,
        pred_time_uuid_colname: str,
    ) -> DataFrame:
        """Ensure all prediction times are represented in the returned
        dataframe.

        Args:
            df (DataFrame): Dataframe with prediction times but without uuid.
            pred_times_with_uuid (DataFrame): Dataframe with prediction times and uuid.
            pred_time_uuid_colname (str): Name of uuid column in both df and pred_times_with_uuid.

        Returns:
            DataFrame: A merged dataframe with all prediction times.
        """
        return pd.merge(
            pred_times_with_uuid,
            df,
            how="left",
            on=pred_time_uuid_colname,
            suffixes=("", "_temp"),
        ).drop(["timestamp_pred"], axis=1)

    @staticmethod
    def _resolve_multiple_values_within_interval_days(
        resolve_multiple: Callable,
        df: DataFrame,
        pred_time_uuid_colname: str,
    ) -> DataFrame:
        """Apply the resolve_multiple function to prediction_times where there
        are multiple values within the interval_days lookahead.

        Args:
            resolve_multiple (Callable): Takes a grouped df and collapses each group to one record (e.g. sum, count etc.).
            df (DataFrame): Source dataframe with all prediction time x val combinations.
            pred_time_uuid_colname (str): Name of uuid column in df.

        Returns:
            DataFrame: DataFrame with one row pr. prediction time.
        """
        # Convert timestamp val to numeric that can be used for resolve_multiple functions
        # Numeric value amounts to days passed since 1/1/1970
        try:
            df["timestamp_val"] = (
                df["timestamp_val"] - dt.datetime(1970, 1, 1)
            ).dt.total_seconds() / 86400
        except TypeError:
            msg.info("All values are NaT, returning empty dataframe")

        # Sort by timestamp_pred in case resolve_multiple needs dates
        df = df.sort_values(by="timestamp_val").groupby(pred_time_uuid_colname)

        if isinstance(resolve_multiple, str):
            resolve_multiple = resolve_multiple_fns.get(resolve_multiple)

        if callable(resolve_multiple):
            df = resolve_multiple(df).reset_index()
        else:
            raise ValueError("resolve_multiple must be or resolve to a Callable")

        return df

    @staticmethod
    def _drop_records_outside_interval_days(
        df: DataFrame,
        direction: str,
        interval_days: float,
        timestamp_pred_colname: str,
        timestamp_value_colname: str,
    ) -> DataFrame:
        """Keep only rows where timestamp_value is within interval_days in
        direction of timestamp_pred.

        Args:
            direction (str): Whether to look ahead or behind.
            interval_days (float): How far to look
            df (DataFrame): Source dataframe
            timestamp_pred_colname (str): Name of timestamp column for predictions in df.
            timestamp_value_colname (str): Name of timestamp column for values in df.

        Raises:
            ValueError: If direction is niether ahead nor behind.

        Returns:
            DataFrame
        """
        df["time_from_pred_to_val_in_days"] = (
            (df[timestamp_value_colname] - df[timestamp_pred_colname])
            / (np.timedelta64(1, "s"))
            / 86_400
        )
        # Divide by 86.400 seconds/day

        if direction == "ahead":
            df["is_in_interval"] = (
                df["time_from_pred_to_val_in_days"] <= interval_days
            ) & (df["time_from_pred_to_val_in_days"] > 0)
        elif direction == "behind":
            df["is_in_interval"] = (
                df["time_from_pred_to_val_in_days"] >= -interval_days
            ) & (df["time_from_pred_to_val_in_days"] < 0)
        else:
            raise ValueError("direction can only be 'ahead' or 'behind'")

        return df[df["is_in_interval"]].drop(
            ["is_in_interval", "time_from_pred_to_val_in_days"],
            axis=1,
        )
