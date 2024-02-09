import datetime as dt
from dataclasses import dataclass
from typing import Protocol, Sequence, Union

import polars as pl

Fallback = Union[int, float, str]
LookDistance = dt.timedelta

# TODO: Add validation that all entity_id and timestamp columns are the same

default_entity_id_col_name = "entity_id"
default_pred_time_uuid_col_name = "pred_time_uuid"
default_pred_time_col_name = "pred_timestamp"


@dataclass
class PredictionTimeFrame:
    df: pl.LazyFrame
    entity_id_col_name: str = default_entity_id_col_name
    timestamp_col_name: str = default_pred_time_col_name
    pred_time_uuid_col_name: str = default_pred_time_uuid_col_name

    def __post_init__(self):
        self.df = self.df.with_columns(
            pl.concat_str(
                pl.col(self.entity_id_col_name), pl.lit("-"), pl.col(self.timestamp_col_name)
            ).alias(self.pred_time_uuid_col_name)
        )

    def to_lazyframe_with_uuid(self) -> pl.LazyFrame:
        return self.df


@dataclass(frozen=True)
class ValueFrame:
    """A frame that contains the values of a time series."""

    df: pl.LazyFrame
    value_type: str
    entity_id_col_name: str = default_entity_id_col_name
    value_timestamp_col_name: str = "value_timestamp"


@dataclass(frozen=True)
class SlicedFrame:
    """A frame that has been sliced by a lookdirection."""

    df: pl.LazyFrame
    pred_time_uuid_col_name: str = default_pred_time_uuid_col_name
    value_col_name: str = "value"


@dataclass(frozen=True)
class AggregatedValueFrame:
    df: pl.LazyFrame
    pred_time_uuid_col_name: str = default_pred_time_uuid_col_name
    value_col_name: str = "value"


class Aggregator(Protocol):
    name: str

    def apply(self, value_frame: SlicedFrame, column_name: str) -> AggregatedValueFrame:
        ...


@dataclass(frozen=True)
class PredictorSpec:
    value_frame: ValueFrame
    lookbehind_distances: Sequence[LookDistance]
    aggregators: Sequence[Aggregator]
    fallbacks: Sequence[Fallback]


@dataclass(frozen=True)
class OutcomeSpec:
    value_frame: ValueFrame
    lookahead_distances: Sequence[LookDistance]
    aggregators: Sequence[Aggregator]
    fallbacks: Sequence[Fallback]


@dataclass(frozen=True)
class TimedeltaFrame:
    df: pl.LazyFrame
    pred_time_uuid_col_name: str = default_pred_time_uuid_col_name
    timedelta_col_name: str = "time_from_prediction_to_value"
    value_col_name: str = "value"

    def get_timedeltas(self) -> Sequence[dt.datetime]:
        return self.df.collect().get_column(self.timedelta_col_name).to_list()


ValueSpecification = Union[PredictorSpec, OutcomeSpec]


@dataclass(frozen=True)
class AggregatedFrame:
    pred_time_uuid_col_name: str
    timestamp_col_name: str
