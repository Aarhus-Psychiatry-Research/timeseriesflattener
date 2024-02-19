from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING, Literal, Union

import pandas as pd
import polars as pl

from timeseriesflattenerv2.feature_specs.default_column_names import default_entity_id_col_name

from .._frame_validator import _validate_col_name_columns_exist
from ..frame_utilities.anyframe_to_lazyframe import _anyframe_to_lazyframe

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


ValueType = Union[int, float, str, None]
InitDF_T = Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame]


LookDistance = dt.timedelta


LookDistances: TypeAlias = Sequence[Union[LookDistance, tuple[LookDistance, LookDistance]]]


@dataclass
class ValueFrame:
    """A frame that contains the values of a time series."""

    init_df: InitVar[InitDF_T]
    value_col_name: str
    entity_id_col_name: str = default_entity_id_col_name
    value_timestamp_col_name: str = "timestamp"

    def __post_init__(self, init_df: InitDF_T):
        self.df = _anyframe_to_lazyframe(init_df)
        _validate_col_name_columns_exist(obj=self)

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()


@dataclass(frozen=True)
class LookPeriod:
    first: LookDistance
    last: LookDistance

    def __post_init__(self):
        if self.first >= self.last:
            raise ValueError(
                f"Invalid LookPeriod. The first value ({self.first}) must be smaller than the large value ({self.last})."
            )


def _lookdistance_to_normalised_lookperiod(
    lookdistance: LookDistance | tuple[LookDistance, LookDistance],
    direction: Literal["ahead", "behind"],
) -> LookPeriod:
    is_ahead = direction == "ahead"
    if isinstance(lookdistance, LookDistance):
        return LookPeriod(
            first=dt.timedelta(days=0) if is_ahead else -lookdistance,
            last=lookdistance if is_ahead else dt.timedelta(0),
        )
    return LookPeriod(
        first=lookdistance[0] if is_ahead else -lookdistance[1],
        last=lookdistance[1] if is_ahead else -lookdistance[0],
    )
