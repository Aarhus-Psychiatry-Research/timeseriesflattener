from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING

import polars as pl

from ._frame_validator import _validate_col_name_columns_exist
from .feature_specs.default_column_names import (
    default_pred_time_uuid_col_name,
    default_timestamp_col_name,
)
from .frame_utilities.anyframe_to_lazyframe import _anyframe_to_lazyframe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .feature_specs.meta import ValueType

if TYPE_CHECKING:
    import datetime as dt


@dataclass(frozen=True)
class TimeMaskedFrame:
    """A frame that has had all values outside its lookbehind and lookahead distances masked."""

    init_df: pl.LazyFrame
    value_col_name: str
    timestamp_col_name: str = default_timestamp_col_name
    pred_time_uuid_col_name: str = default_pred_time_uuid_col_name
    validate_cols_exist: bool = True

    def __post_init__(self):
        if self.validate_cols_exist:
            _validate_col_name_columns_exist(obj=self)

    @property
    def df(self) -> pl.LazyFrame:
        return self.init_df

    def collect(self) -> pl.DataFrame:
        return self.init_df.collect()


@dataclass
class AggregatedValueFrame:
    df: pl.LazyFrame
    value_col_name: str
    pred_time_uuid_col_name: str = default_pred_time_uuid_col_name

    def __post_init__(self):
        _validate_col_name_columns_exist(obj=self)

    def fill_nulls(self, fallback: ValueType) -> AggregatedValueFrame:
        filled = self.df.with_columns(
            pl.col(self.value_col_name)
            .fill_null(fallback)
            .alias(f"{self.value_col_name}_fallback_{fallback}")
        ).drop([self.value_col_name])

        return AggregatedValueFrame(
            df=filled,
            pred_time_uuid_col_name=self.pred_time_uuid_col_name,
            value_col_name=self.value_col_name,
        )

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()


@dataclass
class TimeDeltaFrame:
    df: pl.LazyFrame
    value_col_name: str
    value_timestamp_col_name: str
    pred_time_uuid_col_name: str = default_pred_time_uuid_col_name
    timedelta_col_name: str = "time_from_prediction_to_value"

    def __post_init__(self):
        _validate_col_name_columns_exist(obj=self)

    def get_timedeltas(self) -> Sequence[dt.datetime]:
        return self.collect().get_column(self.timedelta_col_name).to_list()

    def collect(self) -> pl.DataFrame:
        return self.df.collect()


@dataclass
class AggregatedFrame:
    init_df: InitVar[pl.LazyFrame]
    entity_id_col_name: str
    timestamp_col_name: str
    pred_time_uuid_col_name: str

    def __post_init__(self, init_df: pl.LazyFrame):
        self.df = _anyframe_to_lazyframe(init_df)

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()


@dataclass(frozen=True)
class ProcessedFrame:
    df: pl.LazyFrame
    pred_time_uuid_col_name: str

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()
