import datetime as dt
from dataclasses import InitVar, dataclass
from typing import NewType, Protocol, Sequence, Union

import pandas as pd
import polars as pl

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

    def collect(self) -> pl.DataFrame:
        return self.init_df.collect()


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


AggregatorInstance = NewType("AggregatorInstance", pl.Expr)


class Aggregator(Protocol):
    def __call__(self, column_name: str) -> pl.Expr:
        ...


PolarsInt = [pl.Int64, pl.UInt64, pl.Int32, pl.UInt32, pl.Int16, pl.UInt16, pl.Int8, pl.UInt8]
PolarsFloat = [pl.Float64, pl.Float32]


def _downcast_column(
    df: pl.DataFrame, col_name: str, downcast_types: Sequence[pl.DataType]
) -> pl.DataFrame:
    for dtype in downcast_types[-1::-1]:
        try:
            return df.with_columns(pl.col(col_name).cast(dtype))
        except pl.ComputeError:
            print(f"Failed to downcast with {dtype}, trying next type.")
            continue
    return df


def _downcast_dispatcher(
    df: pl.DataFrame, col_name: str, type_categories: Sequence[Sequence[pl.DataType]]
) -> pl.DataFrame:
    dtype = df[col_name].dtype
    for category in type_categories:
        if dtype in category:
            df = _downcast_column(df, col_name, category)
    return df


def _downcast_dataframe(df: pl.LazyFrame) -> pl.LazyFrame:
    collected_df = df.collect()
    for col in df.columns:
        collected_df = _downcast_dispatcher(collected_df, col, [PolarsInt, PolarsFloat])

    return collected_df.lazy()


@dataclass
class PredictorSpec:
    value_frame: ValueFrame
    lookbehind_distances: Sequence[LookDistance]
    aggregators: Sequence[Aggregator]
    fallback: ValueType
    column_prefix: str = "pred"
    attempt_downcast: InitVar[bool] = False

    def __post_init__(self, attempt_downcast: bool) -> None:
        if attempt_downcast:
            downcast_frame = _downcast_dataframe(self.value_frame.df)
            self.value_frame = ValueFrame(downcast_frame, self.value_frame.value_col_name)


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
