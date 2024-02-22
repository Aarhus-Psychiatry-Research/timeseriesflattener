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
    init_frame: InitVar[TimestampValueFrame]
    lookahead_distances: LookDistances
    aggregators: Sequence[Aggregator]
    column_prefix: str = "outc"

    def __post_init__(self, init_frame: TimestampValueFrame):
        self.normalised_lookperiod = [
            _lookdistance_to_normalised_lookperiod(lookdistance=lookdistance, direction="ahead")
            for lookdistance in self.lookahead_distances
        ]

        self.fallback = 0

        self.value_frame = ValueFrame(
            init_df=init_frame.df.with_columns((pl.lit(1)).alias("value")),
            value_col_name="value",
            entity_id_col_name=init_frame.entity_id_col_name,
            value_timestamp_col_name=init_frame.value_timestamp_col_name,
        )

    @property
    def df(self) -> pl.LazyFrame:
        return self.value_frame.df
