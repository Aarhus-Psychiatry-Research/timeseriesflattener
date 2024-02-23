from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING

import polars as pl

from .._frame_validator import _validate_col_name_columns_exist
from ..frame_utilities.anyframe_to_lazyframe import _anyframe_to_lazyframe
from .default_column_names import default_entity_id_col_name

if TYPE_CHECKING:
    from .meta import InitDF_T


@dataclass
class TimestampValueFrame:
    """Timestamps, useful for computing e.g. age.

    Must contain columns:
        entity_id_col_name: The name of the column containing the entity ids. Must be a string, and the column's values must be strings which are unique.
        value_timestamp_col_name: The name of the column containing the timestamps. Must be a string, and the column's values must be datetimes.
    """

    init_df: InitVar[InitDF_T]
    entity_id_col_name: str = default_entity_id_col_name
    value_timestamp_col_name: str = "timestamp"

    def __post_init__(self, init_df: InitDF_T):
        self.df = _anyframe_to_lazyframe(init_df)
        _validate_col_name_columns_exist(obj=self)

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()
