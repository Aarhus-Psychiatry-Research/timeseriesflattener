"""Flattens timeseries.

Takes a time-series and flattens it into a set of prediction times describing values.
"""
import datetime
import datetime as dt
import logging
import random
import time
from datetime import timedelta
from multiprocessing import Pool
from typing import Callable, List, Optional, Sequence, Union

import coloredlogs
import numpy as np
import pandas as pd
import polars as pl
import tqdm
from pyarrow import timestamp
from pydantic import BaseModel as PydanticBaseModel

from timeseriesflattener.aggregation_fns import AggregationFunType
from timeseriesflattener.column_handler import ColumnHandler
from timeseriesflattener.feature_cache.abstract_feature_cache import FeatureCache
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
    TemporalSpec,
    TextPredictorSpec,
)
from timeseriesflattener.flattened_ds_validator import ValidateInitFlattenedDataset
from timeseriesflattener.misc_utils import print_df_dimensions_diff

log = logging.getLogger(__name__)


class SpecCollection(PydanticBaseModel):
    """A collection of specs."""

    outcome_specs: List[OutcomeSpec] = []
    predictor_specs: List[Union[PredictorSpec, TextPredictorSpec]] = []
    static_specs: List[AnySpec] = []

    class Config:
        arbitrary_types_allowed = True

    def __len__(self) -> int:
        """Return number of specs in collection."""
        return (
            len(self.outcome_specs) + len(self.predictor_specs) + len(self.static_specs)
        )


class TimeseriesFlattener:
    """Turn a set of time-series into tabular prediction-time data."""

    def __init__(
        self,
        prediction_times_df: pl.DataFrame,
        entity_id_col_name: str = "entity_id",
        timestamp_col_name: str = "timestamp",
        predictor_col_name_prefix: str = "pred",
        outcome_col_name_prefix: str = "outc",
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
            prediction_times_df (DataFrame): pl.LazyFrame with prediction times, required cols: patient_id, .
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

        self.timestamp_col_name = timestamp_col_name
        self.entity_id_col_name = entity_id_col_name
        self.predictor_col_name_prefix = predictor_col_name_prefix
        self.outcome_col_name_prefix = outcome_col_name_prefix
        self.pred_time_uuid_col_name = "prediction_time_uuid"
        self.unprocessed_specs: SpecCollection = SpecCollection()

        if "value" in prediction_times_df.columns:
            raise ValueError(
                "Column 'value' should not occur in prediction_times_df, only timestamps and ids.",
            )

        self._df = prediction_times_df

        # Create pred_time_uuid_column
        self._df = self._df.with_columns(
            pl.concat_str(
                [
                    pl.col(self.entity_id_col_name).cast(str),
                    pl.col(self.timestamp_col_name).dt.strftime("-%Y-%m-%d-%H-%M-%S"),
                ]
            ).alias(self.pred_time_uuid_col_name)
        )

        if log_to_stdout:
            # Setup logging to stdout by default
            coloredlogs.install(
                level=logging.INFO,
                fmt="%(asctime)s [%(levelname)s] %(message)s",
            )

    @staticmethod
    def _add_back_prediction_times_without_value(
        df: pl.LazyFrame,
        pred_times_with_uuid: pl.LazyFrame,
        pred_time_uuid_colname: str,
    ) -> pl.LazyFrame:
        """Ensure all prediction times are represented in the returned

        dataframe.

        Args:
            df (DataFrame): pl.LazyFrame with prediction times but without uuid.
            pred_times_with_uuid (DataFrame): pl.LazyFrame with prediction times and uuid.
            pred_time_uuid_colname (str): Name of uuid column in both df and pred_times_with_uuid.

        Returns:
            pl.LazyFrame: A merged dataframe with all prediction times.
        """
        return pred_times_with_uuid.join(
            df,
            how="left",
            on=pred_time_uuid_colname,
            suffix="_temp",
        )

    @staticmethod
    def _aggregate_values_within_interval_days(
        aggregation_fn: AggregationFunType,
        df: pl.LazyFrame,
        pred_time_uuid_colname: str,
        val_timestamp_col_name: str,
    ) -> pl.LazyFrame:
        """Apply the aggregation function to prediction_times where there

        are multiple values within the interval_days lookahead.

        Args:
            aggregation_fn (Callable): Takes a grouped df and collapses each group to one record (e.g. sum, count etc.).
            df (DataFrame): Source dataframe with all prediction time x val combinations.
            pred_time_uuid_colname (str): Name of uuid column in df.
            val_timestamp_col_name (str): Name of timestamp column in df.

        Returns:
            pl.LazyFrame: pl.LazyFrame with one row pr. prediction time.
        """
        # Convert timestamp val to numeric that can be used for aggregation functions
        # Numeric value amounts to days passed since 1/1/1970
        # Sort by timestamp_pred in case aggregation needs dates
        grouped_df = df.sort(by=val_timestamp_col_name).groupby(
            pred_time_uuid_colname,
        )

        df = aggregation_fn(grouped_df)

        return df

    @staticmethod
    def _drop_records_outside_interval_days(
        df: pl.LazyFrame,
        direction: str,
        interval_days: float,
        timestamp_pred_colname: str,
        timestamp_value_colname: str,
    ) -> pl.LazyFrame:
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
        df = df.with_columns(
            (
                (
                    pl.col(timestamp_value_colname) - pl.col(timestamp_pred_colname)
                ).dt.seconds()
                / 86_400  # Divide by 86.400 seconds/day
            ).alias("time_from_pred_to_val_in_days")
        )

        if direction == "ahead":
            df = df.with_columns(
                (
                    (pl.col("time_from_pred_to_val_in_days") <= interval_days)
                    & (pl.col("time_from_pred_to_val_in_days") > 0)
                ).alias("is_in_interval")
            )

        elif direction == "behind":
            df = df.with_columns(
                (
                    (pl.col("time_from_pred_to_val_in_days") <= -interval_days)
                    & (pl.col("time_from_pred_to_val_in_days") > 0)
                ).alias("is_in_interval")
            )
        else:
            raise ValueError("direction can only be 'ahead' or 'behind'")

        return df.filter(pl.col("is_in_interval")).drop(
            ["is_in_interval", "time_from_pred_to_val_in_days"]
        )

    @staticmethod
    def _flatten_temporal_values_to_df(
        prediction_times_with_uuid_df: pl.LazyFrame,
        output_spec: TemporalSpec,
        entity_id_col_name: str,
        pred_time_uuid_col_name: str,
        timestamp_col_name: str,
    ) -> pl.LazyFrame:
        """Create a dataframe with flattened values (either predictor or

        outcome depending on the value of "direction").

        Args:
            prediction_times_with_uuid_df (DataFrame): pl.LazyFrame with id_col and
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
        df = (
            prediction_times_with_uuid_df.with_columns(pl.all().suffix("_pred"))
            .join(
                output_spec.timeseries_df,
                how="left",
                on=entity_id_col_name,
                suffix="_val",
                validate="m:m",
            )
            .drop(entity_id_col_name)
        )

        timestamp_val_col_name = f"{timestamp_col_name}_val"
        timestamp_pred_col_name = f"{timestamp_col_name}_pred"

        # Drop prediction times without event times within interval days
        if isinstance(output_spec, OutcomeSpec):
            direction = "ahead"
            interval_days = output_spec.lookahead_days
        elif isinstance(output_spec, (PredictorSpec, TextPredictorSpec)):
            direction = "behind"
            interval_days = output_spec.lookbehind_days
        else:
            raise ValueError(f"Unknown output_spec type {type(output_spec)}")

        df = TimeseriesFlattener._drop_records_outside_interval_days(
            df,
            direction=direction,
            interval_days=interval_days,
            timestamp_pred_colname=timestamp_pred_col_name,
            timestamp_value_colname=timestamp_val_col_name,
        )

        df = TimeseriesFlattener._aggregate_values_within_interval_days(
            aggregation_fn=output_spec.aggregation_fn,
            df=df,
            pred_time_uuid_colname=pred_time_uuid_col_name,
            val_timestamp_col_name=timestamp_val_col_name,
        )

        # If aggregation generates empty values,
        # e.g. when there is only one prediction_time within look_ahead window for slope calculation,
        # replace with NaN

        output_col_name = output_spec.get_output_col_name()

        # Find value_cols and add fallback to them
        df = df.rename({"value": output_col_name}).with_columns(
            pl.col(output_col_name).fill_null(pl.lit(output_spec.fallback))
        )

        # Add back prediction times that don't have a value, and fill them with fallback
        df = TimeseriesFlattener._add_back_prediction_times_without_value(
            df=df,
            pred_times_with_uuid=prediction_times_with_uuid_df,
            pred_time_uuid_colname=pred_time_uuid_col_name,
        )

        if output_spec.fallback is not None:
            df = df.fill_null(
                value=output_spec.fallback,
            )

        return df.select([output_col_name, pred_time_uuid_col_name])

    def _get_temporal_feature(
        self,
        feature_spec: TemporalSpec,
    ) -> pl.LazyFrame:
        """Get feature. Either load from cache, or generate if necessary.

        Args:
            feature_spec (TemporalSpec): Specification of the feature.

        Returns:
            pd.pl.LazyFrame: Feature
        """
        prediction_times_with_uuid_df = self._df.select(
            [
                self.pred_time_uuid_col_name,
                self.entity_id_col_name,
                self.timestamp_col_name,
            ],
        )

        df = self._flatten_temporal_values_to_df(
            prediction_times_with_uuid_df=prediction_times_with_uuid_df.lazy(),
            entity_id_col_name=self.entity_id_col_name,
            pred_time_uuid_col_name=self.pred_time_uuid_col_name,
            output_spec=feature_spec,
            timestamp_col_name=self.timestamp_col_name,
        )

        return df

    # TODO: Add checking of alignment of dataframes

    def _collect_temporal_batch(
        self,
        temporal_batch: List[TemporalSpec],
    ) -> pl.DataFrame:
        """Add predictors to the flattened dataframe from a list."""
        # Shuffle predictor specs to avoid IO contention
        random.shuffle(temporal_batch)

        flattened_predictor_dfs = [
            self._get_temporal_feature(feature_spec) for feature_spec in temporal_batch
        ]

        combined_predictor_dfs = pl.concat(
            flattened_predictor_dfs, how="align"
        ).profile()

        return pl.concat([self._df, combined_predictor_dfs], how="horizontal")

    def _add_static_info(
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
        possible_value_cols = [
            col
            for col in static_spec.timeseries_df.columns
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

        output_col_name = static_spec.get_output_col_name()

        df = pl.LazyFrame(
            {
                self.entity_id_col_name: static_spec.timeseries_df.select(
                    self.entity_id_col_name
                ),
                output_col_name: static_spec.timeseries_df.select(value_col_name),
            },
        )

        self._df = self._df.join(
            df.collect(),
            how="left",
            on=self.entity_id_col_name,
            validate="m:1",
            suffix="",
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

        df = self._df.with_columns(pl.all().suffix("_prediction")).join(
            outcome_spec.timeseries_df.collect(),
            how="left",
            on=self.entity_id_col_name,
            suffix="_outcome",
            validate="m:1",
        )

        df = df.filter(
            pl.col(outcome_timestamp_col_name) > pl.col(prediction_timestamp_col_name)
        )

        if outcome_spec.is_dichotomous():
            df = df.with_columns(
                (
                    pl.col(outcome_timestamp_col_name)
                    + timedelta(days=outcome_spec.lookahead_days)
                    > pl.col(prediction_timestamp_col_name)
                )
                .cast(int)
                .alias(outcome_spec.get_output_col_name())
            )

        df = df.rename(
            {prediction_timestamp_col_name: "timestamp"},
        )
        df = df.drop([outcome_timestamp_col_name, "value"])

        self._df = df

    def _get_cutoff_date_from_spec(self, spec: TemporalSpec) -> datetime.datetime:
        """Get the cutoff date from a spec.

        A cutoff date is the earliest date that a prediction time can get data from the values_df.
        We do not want to include those prediction times, as we might make incorrect inferences.
        For example, if a spec says to look 5 years into the future, but we only have one year of data,
        there will necessarily be fewer outcomes - without that reflecting reality. This means our model won't generalise.

        Returns:
            pd.Timestamp: A cutoff date.
        """

        timestamp_series = spec.timeseries_df.collect().get_column(
            self.timestamp_col_name
        )

        if isinstance(spec, PredictorSpec):
            min_val_date: datetime.datetime = timestamp_series.min()  # type: ignore
            return min_val_date + datetime.timedelta(days=spec.lookbehind_days)

        if isinstance(spec, OutcomeSpec):
            max_val_date: datetime.datetime = timestamp_series.max()  # type: ignore
            return max_val_date - datetime.timedelta(days=spec.lookahead_days)

        raise ValueError(f"Spec type {type(spec)} not recognised.")

    @print_df_dimensions_diff
    def _drop_pred_time_if_insufficient_look_distance(
        self,
        df: pl.LazyFrame,
    ) -> pl.LazyFrame:
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
        output_df = df.filter(
            (pl.col(self.timestamp_col_name) <= cutoff_date_behind)
            & (pl.col(self.timestamp_col_name) >= cutoff_date_ahead)
        )

        return output_df

    def _process_temporal_specs(self, batch_size: int = 1):
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

        temporal_batch = self.unprocessed_specs.outcome_specs
        temporal_batch += self.unprocessed_specs.predictor_specs

        while len(temporal_batch) > 0:
            # Pop batch_size specs from the batch
            log.info(f"Processing {len(temporal_batch)} specs.")
            popped_specs = [temporal_batch.pop() for _ in range(batch_size)]
            log.info(f"{len(temporal_batch)} specs remaining")
            self._df = self._collect_temporal_batch(temporal_batch=popped_specs)

        # Remove the processed specs
        self.unprocessed_specs.outcome_specs = []
        self.unprocessed_specs.predictor_specs = []

    def add_spec(
        self,
        spec: Union[Sequence[AnySpec], AnySpec],
    ):
        """Add a specification to the flattened dataset.

        This adds it to a queue of unprocessed specs, which are not processed
        until you call the .compute() or .get_df() methods. This allows us to
        more effectiely parallelise the processing of the specs.

        Most of the complexity lies in the OutcomeSpec and PredictorSpec objects.
        For further documentation, see those objects and the tutorial.
        """
        specs_to_process = [spec] if not isinstance(spec, Sequence) else spec

        for spec_i in specs_to_process:
            allowed_spec_types = (
                OutcomeSpec,
                PredictorSpec,
                StaticSpec,
                TextPredictorSpec,
            )

            if not isinstance(spec_i, allowed_spec_types):
                raise ValueError(
                    f"Input is not allowed. Must be one of: {allowed_spec_types}",
                )

            if isinstance(spec_i, OutcomeSpec):
                self.unprocessed_specs.outcome_specs.append(spec_i)
            elif isinstance(spec_i, (PredictorSpec, TextPredictorSpec)):
                self.unprocessed_specs.predictor_specs.append(spec_i)
            elif isinstance(spec_i, StaticSpec):
                self.unprocessed_specs.static_specs.append(spec_i)

    def add_age(
        self,
        date_of_birth_df: pl.LazyFrame,
        date_of_birth_col_name: str = "date_of_birth",
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
        output_age_col_name = f"{output_prefix}_age_in_years"

        tmp_prefix = "tmp"
        self._add_static_info(
            static_spec=StaticSpec(
                timeseries_df=date_of_birth_df,
                prefix=tmp_prefix,
                feature_base_name=date_of_birth_col_name,
                # We typically don't want to use date of birth as a predictor,
                # but might want to use transformations - e.g. "year of birth" or "age at prediction time".
            ),
        )

        tmp_date_of_birth_col_name = f"{tmp_prefix}_{date_of_birth_col_name}"

        self._df = self._df.with_columns(
            (
                (
                    pl.col(self.timestamp_col_name) - pl.col(tmp_date_of_birth_col_name)
                ).dt.days()
                / 365.25
            ).alias(output_age_col_name)
        ).drop(columns=tmp_date_of_birth_col_name)

    def compute(self):
        """Compute the flattened dataset."""
        if len(self.unprocessed_specs) == 0:
            log.warning("No unprocessed specs, skipping")
            return

        self._process_temporal_specs()
        self._process_static_specs()

    def get_df(self) -> pl.DataFrame:
        """Get the flattened dataframe. Computes if any unprocessed specs are present.

        Returns:
            pl.LazyFrame: Flattened dataframe.
        """
        if len(self.unprocessed_specs) > 0:
            log.info("There were unprocessed specs, computing...")
            self.compute()

        # Process
        return self._df
