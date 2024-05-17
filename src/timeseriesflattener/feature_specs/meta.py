from __future__ import annotations

import datetime as dt
from dataclasses import InitVar, dataclass
from typing import Literal

import pandas as pd
import polars as pl

from .._frame_validator import _validate_col_name_columns_exist
from ..frame_utilities.anyframe_to_lazyframe import _anyframe_to_lazyframe


@dataclass
class ValueFrame:
    """A frame that contains the values of a time series.

    Must contain columns:
        entity_id_col_name: The name of the column containing the entity ids. Must be a string, and the column's values must be strings which are unique.
        value_timestamp_col_name: The name of the column containing the timestamps. Must be a string, and the column's values must be datetimes.
        Additional columns containing the values of the time series. The name of the columns will be used for feature naming.
    """

    init_df: InitVar[pl.LazyFrame | pl.DataFrame | pd.DataFrame]
    entity_id_col_name: str = "entity_id"
    value_timestamp_col_name: str = "timestamp"
    coerce_to_lazy: InitVar[bool] = True

    def __post_init__(
        self, init_df: pl.LazyFrame | pl.DataFrame | pd.DataFrame, coerce_to_lazy: bool
    ):
        if coerce_to_lazy:
            self.df = _anyframe_to_lazyframe(init_df)
        else:
            self.df: pl.LazyFrame = init_df

        _validate_col_name_columns_exist(obj=self)
        self.value_col_names = [
            col
            for col in self.df.columns
            if col not in [self.entity_id_col_name, self.value_timestamp_col_name]
        ]

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()


@dataclass(frozen=True)
class LookPeriod:
    first: dt.timedelta
    last: dt.timedelta

    def __post_init__(self):
        if self.first >= self.last:
            raise ValueError(
                f"Invalid LookPeriod. The first value ({self.first}) must be smaller than the large value ({self.last})."
            )


def _lookdistance_to_normalised_lookperiod(
    lookdistance: dt.timedelta | tuple[dt.timedelta, dt.timedelta],
    direction: Literal["ahead", "behind"],
) -> LookPeriod:
    is_ahead = direction == "ahead"
    if isinstance(lookdistance, dt.timedelta):
        return LookPeriod(
            first=dt.timedelta(days=0) if is_ahead else -lookdistance,
            last=lookdistance if is_ahead else dt.timedelta(0),
        )
    return LookPeriod(
        first=lookdistance[0] if is_ahead else -lookdistance[1],
        last=lookdistance[1] if is_ahead else -lookdistance[0],
    )
