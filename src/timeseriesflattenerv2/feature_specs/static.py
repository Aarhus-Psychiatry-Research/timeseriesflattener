from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING

import polars as pl

from .._frame_validator import _validate_col_name_columns_exist
from ..frame_utilities.anyframe_to_lazyframe import _anyframe_to_lazyframe
from .default_column_names import default_entity_id_col_name
from .meta import ValueType

if TYPE_CHECKING:
    from .meta import InitDF_T


@dataclass
class StaticFrame:
    init_df: InitVar["InitDF_T"]
    value_col_name: str
    entity_id_col_name: str = default_entity_id_col_name

    def __post_init__(self, init_df: "InitDF_T"):
        self.df = _anyframe_to_lazyframe(init_df)
        _validate_col_name_columns_exist(obj=self)

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()


@dataclass(frozen=True)
class StaticSpec:
    value_frame: StaticFrame
    column_prefix: str
    fallback: ValueType
