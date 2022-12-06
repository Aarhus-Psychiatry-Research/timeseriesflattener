"""Takes a time-series and flattens it into a set of prediction times with.

describing values.
"""
import datetime as dt
import logging
import random
import time
from collections.abc import Callable
from datetime import timedelta
from multiprocessing import Pool
from typing import Optional

import coloredlogs
import numpy as np
import pandas as pd
import tqdm
from catalogue import Registry  # noqa # pylint: disable=unused-import
from dask.diagnostics import ProgressBar
from pandas import DataFrame

from timeseriesflattener.feature_cache.abstract_feature_cache import FeatureCache
from timeseriesflattener.feature_spec_objects import (
    AnySpec,
    OutcomeSpec,
    PredictorSpec,
    TemporalSpec,
)
from timeseriesflattener.flattened_ds_validator import ValidateInitFlattenedDataset
from timeseriesflattener.resolve_multiple_functions import resolve_multiple_fns

ProgressBar().register()

log = logging.getLogger(__name__)


class TimeseriesFlattener:  # pylint: disable=too-many-instance-attributes
    """Turn a set of time-series into tabular prediction-time data."""

    def _override_cache_attributes_with_self_attributes(
        self,
        prediction_times_df: DataFrame,
    ):
        """Make cache inherit attributes from flattened dataset.

        Avoids duplicate specification.
        """
        if (
            not hasattr(self.cache, "prediction_times_df")
            or self.cache.prediction_times_df is None
        ):
            self.cache.prediction_times_df = prediction_times_df
        elif not self.cache.prediction_times_df.equals(prediction_times_df):
            log.warning(
                "Overriding prediction_times_df in cache with prediction_times_df passed to init",
            )
            self.cache.prediction_times_df = prediction_times_df

        for attr in ["pred_time_uuid_col_name", "timestamp_col_name", "id_col_name"]:
            if hasattr(self.cache, attr) and getattr(self.cache, attr) is not None:
                if getattr(self.cache, attr) != getattr(self, attr):
                    log.warning(
                        f"Overriding {attr} in cache with {attr} passed to init of flattened dataset",
                    )
                    setattr(self.cache, attr, getattr(self, attr))

    def __init__(  # pylint: disable=too-many-arguments
        self,
        prediction_times_df: DataFrame,
        cache: Optional[FeatureCache] = None,
        id_col_name: str = "dw_ek_borger",
        timestamp_col_name: str = "timestamp",
        predictor_col_name_prefix: str = "pred",
        outcome_col_name_prefix: str = "outc",
        n_workers: int = 60,
        log_to_stdout: bool = True,
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
            cache (Optional[FeatureCache], optional): Object for feature caching. Should be initialised when passed to init. Defaults to None.
            id_col_name (str, optional): Column namn name for patients ids. Is used across outcome and predictors. Defaults to "dw_ek_borger".
            timestamp_col_name (str, optional): Column name name for timestamps. Is used across outcomes and predictors. Defaults to "timestamp".
            predictor_col_name_prefix (str, optional): Prefix for predictor col names. Defaults to "pred_".
            outcome_col_name_prefix (str, optional): Prefix for outcome col names. Defaults to "outc_".
            n_workers (int): Number of subprocesses to spawn for parallelization. Defaults to 60.
            log_to_stdout (bool): Whether to log to stdout. Either way, also logs to the __name__ namespace, which you can capture with a root logger. Defaults to True.

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
        self.cache = cache

        if self.cache:
            self._override_cache_attributes_with_self_attributes(prediction_times_df)

        self.n_uuids = prediction_times_df.shape[0]

        if "value" in prediction_times_df.columns:
            raise ValueError(
                "Column 'value' should not occur in prediction_times_df, only timestamps and ids.",
            )

        self._df = prediction_times_df

        ValidateInitFlattenedDataset(
            df=self._df,
            timestamp_col_name=self.timestamp_col_name,
            id_col_name=self.id_col_name,
        ).validate_dataset()

        # Create pred_time_uuid_columne
        self._df[self.pred_time_uuid_col_name] = self._df[self.id_col_name].astype(
            str,
        ) + self._df[self.timestamp_col_name].dt.strftime("-%Y-%m-%d-%H-%M-%S")

        if log_to_stdout:
            # Setup logging to stdout by default
            coloredlogs.install(
                level=logging.INFO,
                fmt="%(asctime)s [%(levelname)s] %(message)s",
            )

    @staticmethod
    def _flatten_temporal_values_to_df(  # noqa pylint: disable=too-many-locals
        prediction_times_with_uuid_df: DataFrame,
        output_spec: AnySpec,
        id_col_name: str,
        timestamp_col_name: str,
        pred_time_uuid_col_name: str,
        verbose: bool = False,
    ) -> DataFrame:
        """Create a dataframe with flattened values (either predictor or.

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
            if col_name not in output_spec.values_df.columns:  # type: ignore
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

        df = TimeseriesFlattener._drop_records_outside_interval_days(
            df,
            direction=direction,
            interval_days=output_spec.interval_days,
            timestamp_pred_colname="timestamp_pred",
            timestamp_value_colname="timestamp_val",
        )

        # Add back prediction times that don't have a value, and fill them with fallback
        df = TimeseriesFlattener._add_back_prediction_times_without_value(
            df=df,
            pred_times_with_uuid=prediction_times_with_uuid_df,
            pred_time_uuid_colname=pred_time_uuid_col_name,
        ).fillna(output_spec.fallback)

        df["timestamp_val"].replace({output_spec.fallback: pd.NaT}, inplace=True)

        df = TimeseriesFlattener._resolve_multiple_values_within_interval_days(
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
            log.info(
                f"Returning {df.shape[0]} rows of flattened dataframe for {output_spec.get_col_str()}",
            )

        return df[[pred_time_uuid_col_name, output_spec.get_col_str()]]

    def _get_temporal_feature(
        self,
        feature_spec: TemporalSpec,
    ) -> pd.DataFrame:
        """Get feature. Either load from cache, or generate if necessary.

        Args:
            file_suffix (str, optional): File suffix for the cache lookup. Defaults to "parquet".

        Returns:
            pd.DataFrame: Feature
        """

        if self.cache and self.cache.feature_exists(feature_spec=feature_spec):
            df = self.cache.read_feature(feature_spec=feature_spec)
            return df.set_index(keys=self.pred_time_uuid_col_name).sort_index()
        elif not self.cache:
            log.info("No cache specified, not attempting load")

        df = self._flatten_temporal_values_to_df(
            prediction_times_with_uuid_df=self._df[
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
        if self.cache:
            self.cache.write_feature(
                feature_spec=feature_spec,
                df=df,
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
        """Randomly sample 50 positions in each df and check that their.

        indeces.

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
        log.info(
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

        log.info(f"Concatenation took {round(end_time - start_time, 3)} seconds")

        log.info("Merging with original df")
        self._df = self._df.merge(right=new_features, on=self.pred_time_uuid_col_name)

    def add_temporal_predictor_batch(  # pylint: disable=too-many-branches
        self,
        predictor_batch: list[PredictorSpec],
    ):
        """Add predictors to the flattened dataframe from a list."""

        # Shuffle predictor specs to avoid IO contention
        random.shuffle(predictor_batch)

        with Pool(self.n_workers) as p:
            flattened_predictor_dfs = list(
                tqdm.tqdm(
                    p.imap(func=self._get_temporal_feature, iterable=predictor_batch),
                    total=len(predictor_batch),
                ),
            )

        log.info("Feature generation complete, concatenating")

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
        """Add age at prediction time as predictor.

        Also add patient's birth date. Has its own function because of its very frequent use.

        Args:
            id2date_of_birth (DataFrame): Two columns, id and date_of_birth.
            input_date_of_birth_col_name (str, optional): Name of the date_of_birth column in id2date_of_birth.
                Defaults to "date_of_birth".
            output_prefix (str, optional): Prefix for the output column. Defaults to "pred".
            birth_year_as_predictor (bool, optional): Whether to add birth year as a predictor. Defaults to False.
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

        self._df[output_age_col_name] = (
            (
                self._df[self.timestamp_col_name] - self._df[data_of_birth_col_name]
            ).dt.days
            / (365.25)
        ).round(2)

        if birth_year_as_predictor:
            # Convert datetime to year
            self._df["pred_birth_year"] = self._df[data_of_birth_col_name].dt.year

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
                for col in static_spec.values_df.columns  # type: ignore
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
                self.id_col_name: static_spec.values_df[self.id_col_name],  # type: ignore
                output_col_name: static_spec.values_df[value_col_name],  # type: ignore
            },
        )

        self._df = pd.merge(
            self._df,
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
            self._df,
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

        self._df = df

    def _add_temporal_col_to_flattened_dataset(
        self,
        output_spec: AnySpec,
    ):
        """Add a column to the dataset.

        Either predictor or outcome depending on the type of specification.

        Args:
            output_spec (Union[OutcomeSpec, PredictorSpec]): Specification of the output column.
        """
        timestamp_col_type = output_spec.values_df[self.timestamp_col_name].dtype  # type: ignore

        if timestamp_col_type not in ("Timestamp", "datetime64[ns]"):
            # Convert dtype to timestamp
            raise ValueError(
                f"{self.timestamp_col_name} is of type {timestamp_col_type}, not 'Timestamp' from Pandas. Will cause problems. Convert before initialising FlattenedDataset.",
            )

        df = TimeseriesFlattener._flatten_temporal_values_to_df(
            prediction_times_with_uuid_df=self._df[
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

        self._df = self._df.merge(
            right=df,
            on=self.pred_time_uuid_col_name,
            validate="1:1",
        )

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
            self._add_temporal_col_to_flattened_dataset(
                output_spec=output_spec,
            )

    def add_temporal_predictor(
        self,
        output_spec: PredictorSpec,
    ):
        """Add a column with predictor values to the flattened dataset.

        Args:
            output_spec (Union[PredictorSpec]): Specification of the output column.
        """
        self._add_temporal_col_to_flattened_dataset(
            output_spec=output_spec,
        )

    @staticmethod
    def _add_back_prediction_times_without_value(
        df: DataFrame,
        pred_times_with_uuid: DataFrame,
        pred_time_uuid_colname: str,
    ) -> DataFrame:
        """Ensure all prediction times are represented in the returned.

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
        """Apply the resolve_multiple function to prediction_times where there.

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
            log.info("All values are NaT, returning empty dataframe")

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
        """Keep only rows where timestamp_value is within interval_days in.

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

    def get_df(self) -> DataFrame:
        """Get the flattened dataframe.

        Returns:
            DataFrame: Flattened dataframe.
        """
        return self._df
