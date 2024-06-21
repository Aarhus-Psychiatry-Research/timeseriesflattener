from __future__ import annotations

from dataclasses import InitVar, dataclass

import pandas as pd
import polars as pl
from timeseriesflattener.validators import validate_col_name_columns_exist

from ..utils import anyframe_to_pl_frame


@dataclass
class TimestampValueFrame:
    """Timestamps, useful for computing e.g. age.

    Must contain columns:
        entity_id_col_name: The name of the column containing the entity ids. Must be a string, and the column's values must be strings which are unique.
        value_timestamp_col_name: The name of the column containing the timestamps. Must be a string, and the column's values must be datetimes.
    """

    init_df: InitVar[pl.DataFrame | pd.DataFrame]
    entity_id_col_name: str = "entity_id"
    value_timestamp_col_name: str = "timestamp"

    def __post_init__(self, init_df: pl.DataFrame | pd.DataFrame):
        self.df = anyframe_to_pl_frame(init_df)
        validate_col_name_columns_exist(obj=self)

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df
