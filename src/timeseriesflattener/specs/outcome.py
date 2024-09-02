from __future__ import annotations

import datetime as dt
from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from timeseriesflattener.specs import _lookdistance_to_timedelta_days

from ..aggregators import (
    AggregatorName,
    strings_to_aggregators,
    validate_compatible_fallback_type_for_aggregator,
)
from ..validators import validate_col_name_columns_exist
from .timestamp import TimestampValueFrame
from .value import ValueFrame, lookdistance_to_normalised_lookperiod

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..aggregators import Aggregator


@dataclass
class OutcomeSpec:
    """Specification for an outcome. If your outcome is binary/boolean, you can use BooleanOutcomeSpec instead."""

    value_frame: ValueFrame
    lookahead_distances: InitVar[Sequence[dt.timedelta | tuple[dt.timedelta, dt.timedelta]]]
    aggregators: Sequence[Aggregator]
    fallback: int | float | str | None
    column_prefix: str = "outc"

    def __post_init__(
        self, lookahead_distances: Sequence[dt.timedelta | tuple[dt.timedelta, dt.timedelta]]
    ):
        self.normalised_lookperiod = [
            lookdistance_to_normalised_lookperiod(lookdistance=lookdistance, direction="ahead")
            for lookdistance in lookahead_distances
        ]
        validate_col_name_columns_exist(obj=self)
        for aggregator in self.aggregators:
            validate_compatible_fallback_type_for_aggregator(
                aggregator=aggregator, fallback=self.fallback
            )

    @property
    def df(self) -> pl.DataFrame:
        return self.value_frame.df

    @staticmethod
    def from_primitives(
        df: pl.DataFrame,
        entity_id_col_name: str,
        lookahead_days: Sequence[float | tuple[float, float]],
        aggregators: Sequence[AggregatorName],
        value_timestamp_col_name: str = "timestamp",
        column_prefix: str = "outc",
    ) -> OutcomeSpec:
        """Create an OutcomeSpec from primitives."""
        lookahead_distances = [_lookdistance_to_timedelta_days(d) for d in lookahead_days]

        return OutcomeSpec(
            value_frame=ValueFrame(
                init_df=df,
                entity_id_col_name=entity_id_col_name,
                value_timestamp_col_name=value_timestamp_col_name,
            ),
            lookahead_distances=lookahead_distances,
            aggregators=strings_to_aggregators(aggregators, value_timestamp_col_name),
            fallback=0,
            column_prefix=column_prefix,
        )


@dataclass
class BooleanOutcomeSpec:
    """Specification for a boolean outcome, e.g. whether a patient received a treatment or not.

    The init_frame must contain columns:
        entity_id_col_name: The name of the column containing the entity ids. Must be a string, and the column's values must be strings which are unique.
        value_timestamp_col_name: The name of the column containing the timestamps of when the event occurs. Must be a string, and the column's values must be datetimes.
    """

    init_frame: InitVar[TimestampValueFrame]
    lookahead_distances: Sequence[dt.timedelta | tuple[dt.timedelta, dt.timedelta]]
    aggregators: Sequence[Aggregator]
    output_name: str
    column_prefix: str = "outc"

    def __post_init__(self, init_frame: TimestampValueFrame):
        self.normalised_lookperiod = [
            lookdistance_to_normalised_lookperiod(lookdistance=lookdistance, direction="ahead")
            for lookdistance in self.lookahead_distances
        ]

        self.fallback = 0
        for aggregator in self.aggregators:
            validate_compatible_fallback_type_for_aggregator(
                aggregator=aggregator, fallback=self.fallback
            )

        self.value_frame = ValueFrame(
            init_df=init_frame.df.with_columns((pl.lit(1)).alias(self.output_name)),
            entity_id_col_name=init_frame.entity_id_col_name,
            value_timestamp_col_name=init_frame.value_timestamp_col_name,
        )

    @property
    def df(self) -> pl.DataFrame:
        return self.value_frame.df

    @staticmethod
    def from_primitives(
        df: pl.DataFrame | pd.DataFrame,
        entity_id_col_name: str,
        lookahead_days: Sequence[float | tuple[float, float]],
        aggregators: Sequence[AggregatorName],
        value_timestamp_col_name: str = "timestamp",
        column_prefix: str = "outc",
    ) -> BooleanOutcomeSpec:
        """Create an OutcomeSpec from primitives."""
        lookahead_distances = [_lookdistance_to_timedelta_days(d) for d in lookahead_days]

        return BooleanOutcomeSpec(
            init_frame=TimestampValueFrame(
                init_df=df,
                value_timestamp_col_name=value_timestamp_col_name,
                entity_id_col_name=entity_id_col_name,
            ),
            lookahead_distances=lookahead_distances,
            aggregators=strings_to_aggregators(aggregators, value_timestamp_col_name),
            output_name=column_prefix,
            column_prefix=column_prefix,
        )
