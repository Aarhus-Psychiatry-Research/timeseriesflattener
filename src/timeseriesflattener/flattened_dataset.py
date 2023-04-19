"""Flattens timeseries.

Takes a time-series and flattens it into a set of prediction times describing values.
"""
import datetime as dt
import logging
import random
import time
from datetime import timedelta
from multiprocessing import Pool
from typing import Callable, List, Optional, Union

import coloredlogs
import numpy as np
import pandas as pd
import tqdm
from pandas import DataFrame
from pydantic import BaseModel as PydanticBaseModel

from timeseriesflattener.column_handler import ColumnHandler
from timeseriesflattener.feature_cache.abstract_feature_cache import FeatureCache
from timeseriesflattener.feature_spec_objects import (
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
    TemporalSpec,
    TextPredictorSpec,
    _AnySpec,
)
from timeseriesflattener.flattened_ds_validator import ValidateInitFlattenedDataset
from timeseriesflattener.utils import print_df_dimensions_diff

log = logging.getLogger(__name__)


class SpecCollection(PydanticBaseModel):
    """A collection of specs."""

    outcome_specs: List[OutcomeSpec] = []
    predictor_specs: List[PredictorSpec] = []
    static_specs: List[_AnySpec] = []

    def __len__(self) -> int:
        """Return number of specs in collection."""
        return (
            len(self.outcome_specs) + len(self.predictor_specs) + len(self.static_specs)
        )


class TimeseriesFlattener:
    """Turn a set of time-series into tabular prediction-time data."""

    def _override_cache_attributes_with_self_attributes(
        self,
        prediction_times_df: DataFrame,
    ):
        """Make cache inherit attributes from flattened dataset.

        Avoids duplicate specification.
        """
        if self.cache is None:
            raise ValueError("Cache is None, cannot override attributes")

        if (
            not hasattr(self.cache, "prediction_times_df")
            or self.cache.prediction_times_df is None
        ):
            self.cache.prediction_times_df = prediction_times_df
        elif not self.cache.prediction_times_df.equals(prediction_times_df):
            log.info(
                "Overriding prediction_times_df cache with the one passed to init",
            )
            self.cache.prediction_times_df = prediction_times_df

        for attr in (
            "pred_time_uuid_col_name",
            "timestamp_col_name",
            "entity_id_col_name",
        ):
            if (
                hasattr(self.cache, attr)
                and getattr(self.cache, attr) is not None
                and getattr(self.cache, attr) != getattr(self, attr)
            ):
                log.info(
                    f"Overriding {attr} in cache with {attr} passed to init of flattened dataset",
                )
                setattr(self.cache, attr, getattr(self, attr))

    def __init__(
        self,
        prediction_times_df: DataFrame,
        drop_pred_times_with_insufficient_look_distance: bool,
        cache: Optional[FeatureCache] = None,
        entity_id_col_name: str = "entity_id",
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

        entity_id_col_name | timestamp_col_name
        1           | 2022-01-10
        1           | 2022-01-12
        1           | 2022-01-15

        And a time-series of blood-pressure values as an outcome:
        entity_id_col_name | timestamp_col_name | blood_pressure_value
        1           | 2022-01-09         | 120
        1           | 2022-01-14         | 140

        Then you can "flatten" the outcome into a new df, with a row for each of your prediction times:
        entity_id_col_name | timestamp_col_name | latest_blood_pressure_within_24h
        1           | 2022-01-10         | 120
        1           | 2022-01-12         | NA
        1           | 2022-01-15         | 140

        Args:
            prediction_times_df (DataFrame): Dataframe with prediction times, required cols: patient_id, .
            cache (Optional[FeatureCache], optional): Object for feature caching. Should be initialised when passed to init. Defaults to None.
            entity_id_col_name (str, optional): Column namn name for patients ids. Is used across outcome and predictors. Defaults to "entity_id".
            timestamp_col_name (str, optional): Column name name for timestamps. Is used across outcomes and predictors. Defaults to "timestamp".
            predictor_col_name_prefix (str, optional): Prefix for predictor col names. Defaults to "pred_".
            outcome_col_name_prefix (str, optional): Prefix for outcome col names. Defaults to "outc_".
            n_workers (int): Number of subprocesses to spawn for parallelization. Defaults to 60.
            log_to_stdout (bool): Whether to log to stdout. Either way, also logs to the __name__ namespace, which you can capture with a root logger. Defaults to True.
            drop_pred_times_with_insufficient_look_distance (bool): Whether to drop prediction times with insufficient look distance.
                For example, say your feature has a lookbehind of 2 years, and your first datapoint is 2013-01-01.
                The first prediction time that has sufficient look distance will be on 2015-01-1.
                Otherwise, your feature will imply that you've looked two years into the past, even though you have less than two years of data to look at.

        Raises:
            ValueError: If timestamp_col_name or entity_id_col_name is not in prediction_times_df
            ValueError: If timestamp_col_name is not and could not be converted to datetime
        """
        self.n_workers = n_workers

        self.timestamp_col_name = timestamp_col_name
        self.entity_id_col_name = entity_id_col_name
        self.predictor_col_name_prefix = predictor_col_name_prefix
        self.outcome_col_name_prefix = outcome_col_name_prefix
        self.pred_time_uuid_col_name = "prediction_time_uuid"
        self.cache = cache
        self.unprocessed_specs: SpecCollection = SpecCollection()
        self.drop_pred_times_with_insufficient_look_distance = (
            drop_pred_times_with_insufficient_look_distance
        )

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
            entity_id_col_name=self.entity_id_col_name,
        ).validate_dataset()

        # Create pred_time_uuid_columne
        self._df[self.pred_time_uuid_col_name] = self._df[
            self.entity_id_col_name
        ].astype(str) + self._df[self.timestamp_col_name].dt.strftime(
            "-%Y-%m-%d-%H-%M-%S",
        )

        if log_to_stdout:
            # Setup logging to stdout by default
            coloredlogs.install(
                level=logging.INFO,
                fmt="%(asctime)s [%(levelname)s] %(message)s",
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
        )

    @staticmethod
    def _resolve_multiple_values_within_interval_days(
        resolve_multiple: Callable,
        df: DataFrame,
        pred_time_uuid_colname: str,
        val_timestamp_col_name: str,
    ) -> DataFrame:
        """Apply the resolve_multiple function to prediction_times where there

        are multiple values within the interval_days lookahead.

        Args:
            resolve_multiple (Callable): Takes a grouped df and collapses each group to one record (e.g. sum, count etc.).
            df (DataFrame): Source dataframe with all prediction time x val combinations.
            pred_time_uuid_colname (str): Name of uuid column in df.
            val_timestamp_col_name (str): Name of timestamp column in df.

        Returns:
            DataFrame: DataFrame with one row pr. prediction time.
        """
        # Convert timestamp val to numeric that can be used for resolve_multiple functions
        # Numeric value amounts to days passed since 1/1/1970
        try:
            df[val_timestamp_col_name] = (
                df[val_timestamp_col_name] - dt.datetime(1970, 1, 1)
            ).dt.total_seconds() / 86400
        except TypeError:
            log.info("All values are NaT, returning empty dataframe")

        # Sort by timestamp_pred in case resolve_multiple needs dates
        grouped_df = df.sort_values(by=val_timestamp_col_name).groupby(
            pred_time_uuid_colname,
        )

        if callable(resolve_multiple):
            df = resolve_multiple(grouped_df).reset_index()
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
        """Filter by time from from predictions to values.

        Drop if distance from timestamp_pred to timestamp_value is outside interval_days. Looks in `direction`.

        Args:
            direction (str): Whether to look ahead or behind.
            interval_days (float): How far to look
            df (DataFrame): Source dataframe
            timestamp_pred_colname (str): Name of timestamp column for predictions in df.
            timestamp_value_colname (str): Name of timestamp column for values in df.

        Raises:
            ValueError: If direction is neither ahead nor behind.

        Returns:
            DataFrame
        """
        df["time_from_pred_to_val_in_days"] = (
            (df[timestamp_value_colname] - df[timestamp_pred_colname])  # type: ignore
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

    @staticmethod
    def _flatten_temporal_values_to_df(
        prediction_times_with_uuid_df: DataFrame,
        output_spec: TemporalSpec,
        entity_id_col_name: str,
        pred_time_uuid_col_name: str,
        timestamp_col_name: str,
        verbose: bool = False,
    ) -> DataFrame:
        """Create a dataframe with flattened values (either predictor or

        outcome depending on the value of "direction").

        Args:
            prediction_times_with_uuid_df (DataFrame): Dataframe with id_col and
                timestamps for each prediction time.
            output_spec (TemporalSpec): Specification of the output column.
            entity_id_col_name (str): Name of id_column in prediction_times_with_uuid_df and
                df. Required because this is a static method.
            timestamp_col_name (str): Name of timestamp column in
                prediction_times_with_uuid_df and df. Required because this is a
                static method.
            pred_time_uuid_col_name (str): Name of uuid column in
                prediction_times_with_uuid_df. Required because this is a static method.
            timestamp_col_name (str): Name of timestamp column in. Required because this is a static method.
            verbose (bool, optional): Whether to print progress.


        Returns:
            DataFrame
        """
        # Generate df with one row for each prediction time x event time combination
        # Drop id for faster merge
        df = pd.merge(
            left=prediction_times_with_uuid_df,
            right=output_spec.values_df,
            how="left",
            on=entity_id_col_name,
            suffixes=("_pred", "_val"),
            validate="m:m",
        ).drop(entity_id_col_name, axis=1)

        timestamp_val_col_name = f"{timestamp_col_name}_val"
        timestamp_pred_col_name = f"{timestamp_col_name}_pred"
        df = TimeseriesFlattener.rename_input_col_to_value(
            df=df,
            output_spec=output_spec,
        )

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
            interval_days=output_spec.interval_days,  # type: ignore
            timestamp_pred_colname=timestamp_pred_col_name,
            timestamp_value_colname=timestamp_val_col_name,
        )

        df[timestamp_val_col_name].replace(
            {output_spec.fallback: pd.NaT},
            inplace=True,  # noqa
        )

        df = TimeseriesFlattener._resolve_multiple_values_within_interval_days(
            resolve_multiple=output_spec.resolve_multiple_fn,  # type: ignore
            df=df,
            pred_time_uuid_colname=pred_time_uuid_col_name,
            val_timestamp_col_name=timestamp_val_col_name,
        )

        # handle embedding and dimensionality reduction if text predictor
        if isinstance(output_spec, TextPredictorSpec):
            df = ColumnHandler.embed_text_column(
                df=df,
                text_col_name="value",
                embedding_fn=output_spec.embedding_fn,
                embedding_fn_kwargs=output_spec.embedding_fn_kwargs,
            )

        # If resolve_multiple generates empty values,
        # e.g. when there is only one prediction_time within look_ahead window for slope calculation,
        # replace with NaN

        # Rename column
        df = ColumnHandler.rename_value_column(df=df, output_spec=output_spec)

        # Find value_cols and add fallback to them
        value_col_str_name = ColumnHandler.get_value_col_str_name(
            df=df,
            output_spec=output_spec,
        )
        df = ColumnHandler.replace_na_in_spec_col_with_fallback(
            df=df,
            output_spec=output_spec,
        )

        # check if multiindex and flatten
        df = ColumnHandler.flatten_multiindex(df)
        if verbose:
            log.info(
                f"Returning {df.shape[0]} rows of flattened dataframe for {output_spec.get_col_str()}",
            )

        # Add back prediction times that don't have a value, and fill them with fallback
        df = TimeseriesFlattener._add_back_prediction_times_without_value(
            df=df,
            pred_times_with_uuid=prediction_times_with_uuid_df,
            pred_time_uuid_colname=pred_time_uuid_col_name,
        ).fillna(
            output_spec.fallback,  # type: ignore
        )

        return df[[*value_col_str_name, pred_time_uuid_col_name]]

    @staticmethod
    def rename_input_col_to_value(
        df: pd.DataFrame,
        output_spec: TemporalSpec,
    ) -> pd.DataFrame:
        """Checks whether 'value' is a column in df, and if not, renames the
        input column"""
        if "value" not in df.columns:
            df = df.rename(columns={output_spec.input_col_name_override: "value"})
        return df

    def _get_temporal_feature(
        self,
        feature_spec: TemporalSpec,
    ) -> pd.DataFrame:
        """Get feature. Either load from cache, or generate if necessary.

        Args:
            feature_spec (TemporalSpec): Specification of the feature.

        Returns:
            pd.DataFrame: Feature
        """
        if self.cache:
            if self.cache.feature_exists(feature_spec=feature_spec):
                log.debug(
                    f"Cache hit for {feature_spec.get_col_str()}, loading from cache",
                )
                df = self.cache.read_feature(feature_spec=feature_spec)
                return df.set_index(keys=self.pred_time_uuid_col_name).sort_index()
            log.debug(f"Cache miss for {feature_spec.get_col_str()}, generating")
        elif not self.cache:
            log.debug("No cache specified, not attempting load")

        df = self._flatten_temporal_values_to_df(
            prediction_times_with_uuid_df=self._df[
                [
                    self.pred_time_uuid_col_name,
                    self.entity_id_col_name,
                    self.timestamp_col_name,
                ]
            ],
            entity_id_col_name=self.entity_id_col_name,
            pred_time_uuid_col_name=self.pred_time_uuid_col_name,
            output_spec=feature_spec,
            timestamp_col_name=self.timestamp_col_name,
        )

        # Write df to cache if exists
        if self.cache:
            self.cache.write_feature(
                feature_spec=feature_spec,
                df=df,
            )

        return df.set_index(keys=self.pred_time_uuid_col_name).sort_index()

    @staticmethod
    def _check_dfs_are_ready_for_concat(dfs: List[pd.DataFrame]):
        """Sample each df and check for identical indices.

        This checks that all the dataframes are aligned before
        concatenation.
        """
        base_df = dfs[0]
        base_length = len(dfs[0])
        n_dfs = len(dfs)

        log.info(
            "Checking alignment of dataframes - this might take a little while (~2 minutes for 1.000 dataframes with 2.000.000 rows).",
        )

        log.debug(
            "Checking that dataframes are ready for concatenation - namely that their indices are aligned. This is a sanity check, and should not be necessary if the dataframes were correctly aligned before concatenation. However, any errors here will completely break predictions, so rather safe than sorry. Can take a while for a large number of dataframes, e.g. 2 minutes for 1_000 dataframes with 2_000_000 rows.",
        )

        for i, feature_df in enumerate(dfs[1:]):
            log.debug(f"Checking df {i+2} of {n_dfs}")

            errors = []

            # Check that dataframes are of equal length
            log.debug("Checking that dataframes are of equal length")
            if len(feature_df) != base_length:
                errors.append(
                    "Dataframes are not of equal length. ",
                )

            log.debug("Checking that indices are aligned")
            if not all(
                feature_df.index == base_df.index,
            ):
                errors.append(
                    "Dataframes are not aligned. ",
                )

            if errors:
                debug_info = f"Columns in dataframes: 0_df: {dfs[0].columns}, feature_df: {feature_df.columns}. Were they correctly aligned before concatenation?"
                raise ValueError(
                    f"Dataframes are not ready for concatenation. {errors}, {debug_info}",
                )

    def _concatenate_flattened_timeseries(
        self,
        flattened_predictor_dfs: List[pd.DataFrame],
    ) -> None:
        """Concatenate flattened predictor dfs."""

        start_time = time.time()

        # Check that dfs are ready for concatenation. Concatenation doesn't merge on IDs, but is **much** faster.
        # We thus require that a) the dfs are sorted so each row matches the same ID and b) that each df has a row
        # for each id.
        self._check_dfs_are_ready_for_concat(dfs=flattened_predictor_dfs)

        # If so, ready for concatenation. Reset index to be ready for the merge at the end.
        log.info(
            "Starting concatenation. Will take some time on performant systems, e.g. 30s for 100 features and 2_000_000 prediction times. This is normal.",
        )
        new_features = pd.concat(
            objs=flattened_predictor_dfs,
            axis=1,
        ).reset_index()

        end_time = time.time()

        log.info(f"Concatenation took {round(end_time - start_time, 3)} seconds")

        log.info("Merging with original df")
        self._df = self._df.merge(right=new_features, on=self.pred_time_uuid_col_name)

    def _add_temporal_batch(
        self,
        temporal_batch: List[TemporalSpec],
    ):
        """Add predictors to the flattened dataframe from a list."""
        # Shuffle predictor specs to avoid IO contention
        random.shuffle(temporal_batch)

        n_workers = min(self.n_workers, len(temporal_batch))

        # Chunksize is the number of predictors to process in each worker.
        # If we don't set chunksize, imap uses the default of 1, which means a bunch of IO overhead.
        chunksize = max(1, round(len(temporal_batch) / (n_workers)))

        log.info(
            f"Processing {len(temporal_batch)} temporal features in parallel with {n_workers} workers. Chunksize is {chunksize}. If this is above 1, it may take some time for the progress bar to move, as processing is batched. However, this makes for much faster total performance.",
        )

        with Pool(n_workers) as p:
            flattened_predictor_dfs = list(
                tqdm.tqdm(
                    p.imap(
                        func=self._get_temporal_feature,
                        iterable=temporal_batch,
                        chunksize=chunksize,
                    ),
                    total=len(temporal_batch),
                ),
            )

        self._concatenate_flattened_timeseries(
            flattened_predictor_dfs=flattened_predictor_dfs,
        )

    def _add_static_info(
        self,
        static_spec: _AnySpec,
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
                if col not in self.entity_id_col_name
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

        output_col_name = static_spec.get_col_str()

        df = pd.DataFrame(
            {
                self.entity_id_col_name: static_spec.values_df[self.entity_id_col_name],  # type: ignore
                output_col_name: static_spec.values_df[value_col_name],  # type: ignore
            },
        )

        self._df = pd.merge(
            self._df,
            df,
            how="left",
            on=self.entity_id_col_name,
            suffixes=("", ""),
            validate="m:1",
        )

    def _process_static_specs(self):
        """Process static specs."""
        for spec in self.unprocessed_specs.static_specs:
            self._add_static_info(
                static_spec=spec,
            )

        self.unprocessed_specs.static_specs = []

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
            on=self.entity_id_col_name,
            suffixes=("_prediction", "_outcome"),
            validate="m:1",
        )

        df = df.drop(
            df[
                df[outcome_timestamp_col_name] < df[prediction_timestamp_col_name]
            ].index,
        )

        if outcome_spec.is_dichotomous():
            outcome_is_within_lookahead = (
                df[prediction_timestamp_col_name]  # type: ignore
                + timedelta(days=outcome_spec.interval_days)  # type: ignore
                > df[outcome_timestamp_col_name]
            )

            df[outcome_spec.get_col_str()] = outcome_is_within_lookahead.astype(int)

        df = df.rename(
            {prediction_timestamp_col_name: "timestamp"},
            axis=1,
        )
        df = df.drop([outcome_timestamp_col_name], axis=1)

        df = df.drop(["value"], axis=1)

        self._df = df

    def _get_cutoff_date_from_spec(self, spec: TemporalSpec) -> pd.Timestamp:
        """Get the cutoff date from a spec.

        A cutoff date is the earliest date that a prediction time can get data from the values_df.
        We do not want to include those prediction times, as we might make incorrect inferences.
        For example, if a spec says to look 5 years into the future, but we only have one year of data,
        there will necessarily be fewer outcomes - without that reflecting reality. This means our model won't generalise.

        Returns:
            pd.Timestamp: A cutoff date.
        """

        if isinstance(spec, PredictorSpec):
            min_val_date = spec.values_df[self.timestamp_col_name].min()  # type: ignore
            return min_val_date + pd.Timedelta(days=spec.lookbehind_days)

        if isinstance(spec, OutcomeSpec):
            max_val_date = spec.values_df[self.timestamp_col_name].max()  # type: ignore
            return max_val_date - pd.Timedelta(days=spec.lookahead_days)

        raise ValueError(f"Spec type {type(spec)} not recognised.")

    @print_df_dimensions_diff
    def _drop_pred_time_if_insufficient_look_distance(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop prediction times if there is insufficient look distance.

        A prediction time has insufficient look distance if the feature spec
            tries to look beyond the boundary of the data.
        For example, if a predictor specifies two years of lookbehind,
            but you only have one year of data prior to the prediction time.

        Takes a dataframe as input to conform to a standard filtering interface,
            which we can easily decorate.
        """
        spec_batch = (
            self.unprocessed_specs.outcome_specs
            + self.unprocessed_specs.predictor_specs
        )

        # Find the latest cutoff date for predictors
        cutoff_date_behind = pd.Timestamp("1700-01-01")

        # Find the earliest cutoff date for outcomes
        cutoff_date_ahead = pd.Timestamp("2200-01-01")

        for spec in spec_batch:
            spec_cutoff_date = self._get_cutoff_date_from_spec(spec=spec)

            if isinstance(spec, OutcomeSpec):
                cutoff_date_ahead = min(cutoff_date_ahead, spec_cutoff_date)
            elif isinstance(spec, PredictorSpec):
                cutoff_date_behind = max(cutoff_date_behind, spec_cutoff_date)

        # Drop all prediction times that are outside the cutoffs window
        output_df = df[
            (df[self.timestamp_col_name] >= cutoff_date_behind)
            & (df[self.timestamp_col_name] <= cutoff_date_ahead)
        ]

        if output_df.shape[0] == 0:
            raise ValueError(
                "No records left after dropping records outside look distance",
            )

        return output_df

    def _process_temporal_specs(self):
        """Process outcome specs."""

        for spec in self.unprocessed_specs.outcome_specs:
            # Handle incident specs separately, since their operations can be vectorised,
            # making them much faster
            if hasattr(spec, "incident") and spec.incident:
                self._add_incident_outcome(
                    outcome_spec=spec,
                )

        # Remove processed specs. Beware of using .remove on a list of specs, as it causes errors.
        self.unprocessed_specs.outcome_specs = [
            s
            for s in self.unprocessed_specs.outcome_specs
            if hasattr(s, "incident") and not s.incident
        ]

        temporal_batch: List[TemporalSpec] = self.unprocessed_specs.outcome_specs  # type: ignore
        temporal_batch += self.unprocessed_specs.predictor_specs

        if self.drop_pred_times_with_insufficient_look_distance:
            self._df = self._drop_pred_time_if_insufficient_look_distance(df=self._df)

        if len(temporal_batch) > 0:
            self._add_temporal_batch(temporal_batch=temporal_batch)

        # Remove the processed specs
        self.unprocessed_specs.outcome_specs = []
        self.unprocessed_specs.predictor_specs = []

    def _check_that_spec_df_has_required_columns(self, spec: _AnySpec):
        """Check that df has required columns."""
        # Find all attributes in self that contain col_name
        required_columns = [self.entity_id_col_name]

        if not isinstance(spec, StaticSpec):
            required_columns += [self.timestamp_col_name]

        for col in required_columns:
            if col not in spec.values_df.columns:  # type: ignore
                raise ValueError(f"Missing required column: {col}")

    def _check_that_spec_df_timestamp_col_is_correctly_formatted(
        self,
        spec: TemporalSpec,
    ):
        """Check that timestamp column is correctly formatted. Attempt to coerce if possible."""
        timestamp_col_type = spec.values_df[self.timestamp_col_name].dtype  # type: ignore

        if timestamp_col_type not in ("Timestamp", "datetime64[ns]"):
            # Convert column dtype to datetime64[ns] if it isn't already
            log.info(
                f"{spec.feature_name}: Converting timestamp column to datetime64[ns]",
            )

            spec.values_df[self.timestamp_col_name] = pd.to_datetime(  # type: ignore
                spec.values_df[self.timestamp_col_name],  # type: ignore
            )

            min_timestamp = min(spec.values_df[self.timestamp_col_name])  # type: ignore

            if min_timestamp < pd.Timestamp("1971-01-01"):
                log.warning(
                    f"{spec.feature_name}: Minimum timestamp is {min_timestamp} - perhaps ints were coerced to timestamps?",
                )

    def add_spec(
        self,
        spec: Union[List[_AnySpec], _AnySpec],
    ):
        """Add a specification to the flattened dataset.

        This adds it to a queue of unprocessed specs, which are not processed
        until you call the .compute() or .get_df() methods. This allows us to
        more effectiely parallelise the processing of the specs.

        Most of the complexity lies in the OutcomeSpec and PredictorSpec objects.
        For further documentation, see those objects and the tutorial.
        """
        if isinstance(spec, _AnySpec):
            specs_to_process: List[_AnySpec] = [spec]
        else:
            specs_to_process = spec

        for spec_i in specs_to_process:
            allowed_spec_types = (OutcomeSpec, PredictorSpec, StaticSpec)

            if not isinstance(spec_i, allowed_spec_types):
                raise ValueError(
                    f"Input is not allowed. Must be one of: {allowed_spec_types}",
                )

            self._check_that_spec_df_has_required_columns(spec=spec_i)

            if isinstance(spec_i, TemporalSpec):
                self._check_that_spec_df_timestamp_col_is_correctly_formatted(
                    spec=spec_i,
                )

            if isinstance(spec_i, OutcomeSpec):
                self.unprocessed_specs.outcome_specs.append(spec_i)
            elif isinstance(spec_i, PredictorSpec):
                self.unprocessed_specs.predictor_specs.append(spec_i)
            elif isinstance(spec_i, StaticSpec):
                self.unprocessed_specs.static_specs.append(spec_i)

    def add_age(
        self,
        date_of_birth_df: DataFrame,
        date_of_birth_col_name: Optional[str] = "date_of_birth",
        output_prefix: str = "pred",
    ):
        """Add age at prediction time as predictor.

        Has its own function because of its very frequent use.

        Args:
            date_of_birth_df (DataFrame): Two columns, one matching self.entity_id_col_name and one containing date_of_birth.
            date_of_birth_col_name (str, optional): Name of the date_of_birth column in date_of_birth_df.
                Defaults to "date_of_birth".
            output_prefix (str, optional): Prefix for the output column. Defaults to "pred".
        """
        if date_of_birth_df[date_of_birth_col_name].dtype != "<M8[ns]":
            try:
                date_of_birth_df[date_of_birth_col_name] = pd.to_datetime(
                    date_of_birth_df[date_of_birth_col_name],
                    format="%Y-%m-%d",
                )
            except ValueError as e:
                raise ValueError(
                    f"Conversion of {date_of_birth_col_name} to datetime failed, doesn't match format %Y-%m-%d. Recommend converting to datetime before adding.",
                ) from e

        output_age_col_name = f"{output_prefix}_age_in_years"

        tmp_prefix = "tmp"
        self._add_static_info(
            static_spec=_AnySpec(
                values_df=date_of_birth_df,
                input_col_name_override=date_of_birth_col_name,
                prefix=tmp_prefix,
                # We typically don't want to use date of birth as a predictor,
                # but might want to use transformations - e.g. "year of birth" or "age at prediction time".
                feature_name=date_of_birth_col_name,
            ),
        )

        tmp_date_of_birth_col_name = f"{tmp_prefix}_{date_of_birth_col_name}"

        self._df[output_age_col_name] = (
            (
                self._df[self.timestamp_col_name] - self._df[tmp_date_of_birth_col_name]
            ).dt.days
            / (365.25)
        ).round(2)

        # Remove date of birth column
        self._df.drop(columns=tmp_date_of_birth_col_name, inplace=True)  # noqa

    def compute(self):
        """Compute the flattened dataset."""
        if len(self.unprocessed_specs) == 0:
            log.warning("No unprocessed specs, skipping")
            return

        self._process_temporal_specs()
        self._process_static_specs()

    def get_df(self) -> DataFrame:
        """Get the flattened dataframe. Computes if any unprocessed specs are present.

        Returns:
            DataFrame: Flattened dataframe.
        """
        if len(self.unprocessed_specs) > 0:
            log.info("There were unprocessed specs, computing...")
            self.compute()

        # Process
        return self._df
