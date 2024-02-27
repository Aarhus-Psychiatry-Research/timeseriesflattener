from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING

import polars as pl

from .._frame_validator import _validate_col_name_columns_exist
from .meta import LookDistances, ValueFrame, ValueType, _lookdistance_to_normalised_lookperiod

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..aggregators import Aggregator
    from .timestamp_frame import TimestampValueFrame


@dataclass
class OutcomeSpec:
    """Specification for an outcome. If your outcome is binary/boolean, you can use BooleanOutcomeSpec instead."""

    value_frame: ValueFrame
    lookahead_distances: InitVar[LookDistances]
    aggregators: Sequence[Aggregator]
    fallback: ValueType
    column_prefix: str = "outc"

    def __post_init__(self, lookahead_distances: LookDistances):
        self.normalised_lookperiod = [
            _lookdistance_to_normalised_lookperiod(lookdistance=lookdistance, direction="ahead")
            for lookdistance in lookahead_distances
        ]
        _validate_col_name_columns_exist(obj=self)

    @property
    def df(self) -> pl.LazyFrame:
        return self.value_frame.df


@dataclass
class BooleanOutcomeSpec:
    """Specification for a boolean outcome, e.g. whether a patient received a treatment or not.

    The init_frame must contain columns:
        entity_id_col_name: The name of the column containing the entity ids. Must be a string, and the column's values must be strings which are unique.
        value_timestamp_col_name: The name of the column containing the timestamps of when the event occurs. Must be a string, and the column's values must be datetimes.
    """

    init_frame: InitVar[TimestampValueFrame]
    lookahead_distances: LookDistances
    aggregators: Sequence[Aggregator]
    output_name: str
    column_prefix: str = "outc"

    def __post_init__(self, init_frame: TimestampValueFrame):
        self.normalised_lookperiod = [
            _lookdistance_to_normalised_lookperiod(lookdistance=lookdistance, direction="ahead")
            for lookdistance in self.lookahead_distances
        ]

        self.fallback = 0

        self.value_frame = ValueFrame(
            init_df=init_frame.df.with_columns((pl.lit(1)).alias(self.output_name)),
            entity_id_col_name=init_frame.entity_id_col_name,
            value_timestamp_col_name=init_frame.value_timestamp_col_name,
        )

    @property
    def df(self) -> pl.LazyFrame:
        return self.value_frame.df
