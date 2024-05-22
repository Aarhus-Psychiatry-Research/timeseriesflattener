from __future__ import annotations

from dataclasses import InitVar, dataclass

import pandas as pd
import polars as pl

from .._frame_validator import _validate_col_name_columns_exist
from ..frame_utilities.anyframe_to_lazyframe import _anyframe_to_lazyframe


@dataclass
class StaticFrame:
    init_df: InitVar[pl.LazyFrame | pl.DataFrame | pd.DataFrame]

    entity_id_col_name: str = "entity_id"

    def __post_init__(self, init_df: pl.LazyFrame | pl.DataFrame | pd.DataFrame):
        self.df = _anyframe_to_lazyframe(init_df)
        _validate_col_name_columns_exist(obj=self)
        self.value_col_names = [col for col in self.df.columns if col != self.entity_id_col_name]

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()


@dataclass(frozen=True)
class StaticSpec:
    """Specification for a static feature, e.g. the sex of a person.

    The value_frame must contain columns:
        entity_id_col_name: The name of the column containing the entity ids. Must be a string, and the column's values must be strings which are unique.
        additional columns containing the values of the static feature. The name of the columns will be used for feature naming.
    """

    value_frame: StaticFrame
    column_prefix: str
    fallback: int | float | str | None
