from __future__ import annotations

import datetime as dt
from dataclasses import InitVar, dataclass
from typing import Literal

import pandas as pd
import polars as pl

from ..validators import validate_col_name_columns_exist
from ..utils import anyframe_to_pl_frame


@dataclass
class ValueFrame:
    """A frame that contains the values of a time series.

    Must contain columns:
        entity_id_col_name: The name of the column containing the entity ids. Must be a string, and the column's values must be strings which are unique.
        value_timestamp_col_name: The name of the column containing the timestamps. Must be a string, and the column's values must be datetimes.
        Additional columns containing the values of the time series. The name of the columns will be used for feature naming.
    """

    init_df: InitVar[pl.DataFrame | pd.DataFrame]
    entity_id_col_name: str = "entity_id"
    value_timestamp_col_name: str = "timestamp"

    def __post_init__(self, init_df: pl.DataFrame | pd.DataFrame):
        self.df = anyframe_to_pl_frame(init_df)

        validate_col_name_columns_exist(obj=self)
        self.value_col_names = [
            col
            for col in self.df.columns
            if col not in [self.entity_id_col_name, self.value_timestamp_col_name]
        ]


@dataclass(frozen=True)
class LookPeriod:
    first: dt.timedelta
    last: dt.timedelta

    def __post_init__(self):
        if self.first >= self.last:
            raise ValueError(
                f"Invalid LookPeriod. The first value ({self.first}) must be smaller than the large value ({self.last})."
            )


def lookdistance_to_normalised_lookperiod(
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
