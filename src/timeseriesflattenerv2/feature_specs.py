import datetime as dt
from dataclasses import InitVar, dataclass
from typing import Protocol, Sequence, Union

import pandas as pd
import polars as pl
from polars.dataframe.group_by import GroupBy
from polars.lazyframe.group_by import LazyGroupBy

ValueType = Union[int, float, str, None]
LookDistance = dt.timedelta

default_entity_id_col_name = "entity_id"
default_pred_time_uuid_col_name = "pred_time_uuid"
default_pred_time_col_name = "pred_timestamp"
default_timestamp_col_name = "timestamp"


@dataclass
class PredictionTimeFrame:
    init_df: InitVar[pl.LazyFrame | pd.DataFrame]
    entity_id_col_name: str = default_entity_id_col_name
    timestamp_col_name: str = default_pred_time_col_name
    pred_time_uuid_col_name: str = default_pred_time_uuid_col_name

    def __post_init__(self, init_df: pl.LazyFrame | pd.DataFrame):
        if isinstance(init_df, pd.DataFrame):
            self.df: pl.LazyFrame = pl.from_pandas(init_df).lazy()
        else:
            self.df: pl.LazyFrame = init_df

        self.df = self.df.with_columns(
            pl.concat_str(
                pl.col(self.entity_id_col_name), pl.lit("-"), pl.col(self.timestamp_col_name)
            )
            .str.strip_chars()
            .alias(self.pred_time_uuid_col_name)
        )

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()

    def required_columns(self) -> Sequence[str]:
        return [self.entity_id_col_name]


@dataclass
class ValueFrame:
    """A frame that contains the values of a time series."""

    init_df: InitVar[pl.LazyFrame | pd.DataFrame]
    value_col_name: str
    entity_id_col_name: str = default_entity_id_col_name
    value_timestamp_col_name: str = "timestamp"

    def __post_init__(self, init_df: pl.LazyFrame | pd.DataFrame):
        if isinstance(init_df, pd.DataFrame):
            self.df: pl.LazyFrame = pl.from_pandas(init_df).lazy()
        else:
            self.df: pl.LazyFrame = init_df

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()


@dataclass(frozen=True)
class SlicedFrame:
    """A frame that has been sliced by a lookdirection."""

    init_df: pl.LazyFrame
    value_col_name: str
    pred_time_uuid_col_name: str = default_pred_time_uuid_col_name

    @property
    def df(self) -> pl.LazyFrame:
        return self.init_df


@dataclass
class AggregatedValueFrame:
    df: pl.LazyFrame
    value_col_name: str
    pred_time_uuid_col_name: str = default_pred_time_uuid_col_name

    def fill_nulls(self, fallback: ValueType) -> "AggregatedValueFrame":
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


class Aggregator(Protocol):
    def apply(self, value_frame: LazyGroupBy | GroupBy, column_name: str) -> AggregatedValueFrame:
        ...


@dataclass(frozen=True)
class PredictorSpec:
    value_frame: ValueFrame
    lookbehind_distances: Sequence[LookDistance]
    aggregators: Sequence[Aggregator]
    fallback: ValueType
    column_prefix: str = "pred"


@dataclass(frozen=True)
class OutcomeSpec:
    value_frame: ValueFrame
    lookahead_distances: Sequence[LookDistance]
    aggregators: Sequence[Aggregator]
    fallback: ValueType
    column_prefix: str = "outc"


@dataclass
class TimedeltaFrame:
    df: pl.LazyFrame
    value_col_name: str
    pred_time_uuid_col_name: str = default_pred_time_uuid_col_name
    timedelta_col_name: str = "time_from_prediction_to_value"

    def get_timedeltas(self) -> Sequence[dt.datetime]:
        return self.collect().get_column(self.timedelta_col_name).to_list()

    def collect(self) -> pl.DataFrame:
        return self.df.collect()


ValueSpecification = Union[PredictorSpec, OutcomeSpec]


@dataclass(frozen=True)
class AggregatedFrame:
    df: pl.LazyFrame
    pred_time_uuid_col_name: str
    timestamp_col_name: str

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()
