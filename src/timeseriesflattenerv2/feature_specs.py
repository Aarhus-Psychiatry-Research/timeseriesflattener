import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass
from typing import Literal, NewType, Sequence, TypeAlias, Union

import pandas as pd
import polars as pl
from iterpy.iter import Iter

ValueType = Union[int, float, str, None]
LookDistance = dt.timedelta

default_entity_id_col_name = "entity_id"
default_pred_time_uuid_col_name = "pred_time_uuid"
default_pred_time_col_name = "pred_timestamp"
default_timestamp_col_name = "timestamp"

InitDF_T = pl.LazyFrame | pl.DataFrame | pd.DataFrame


def _anyframe_to_lazyframe(init_df: InitDF_T) -> pl.LazyFrame:
    if isinstance(init_df, pl.LazyFrame):
        return init_df
    if isinstance(init_df, pl.DataFrame):
        return init_df.lazy()
    if isinstance(init_df, pd.DataFrame):
        return pl.from_pandas(init_df).lazy()
    raise ValueError(f"Unsupported type: {type(init_df)}.")


FrameTypes: TypeAlias = "PredictionTimeFrame | ValueFrame | TimeMaskedFrame | AggregatedValueFrame | TimedeltaFrame | TimestampValueFrame | PredictorSpec | OutcomeSpec | BooleanOutcomeSpec"


def _validate_col_name_columns_exist(obj: FrameTypes):
    missing_columns = (
        Iter(dir(obj))
        .filter(lambda attr_name: attr_name.endswith("_col_name"))
        .map(lambda attr_name: getattr(obj, attr_name))
        .filter(lambda col_name: col_name not in obj.df.columns)
        .to_list()
    )

    if len(missing_columns) > 0:
        raise SpecColumnError(
            f"""Missing columns: {missing_columns} in dataframe.
Current columns are: {obj.df.columns}."""
        )


@dataclass
class PredictionTimeFrame:
    init_df: InitVar[InitDF_T]
    entity_id_col_name: str = default_entity_id_col_name
    timestamp_col_name: str = default_pred_time_col_name
    pred_time_uuid_col_name: str = default_pred_time_uuid_col_name

    def __post_init__(self, init_df: InitDF_T):
        self.df = _anyframe_to_lazyframe(init_df)
        self.df = self.df.with_columns(
            pl.concat_str(
                pl.col(self.entity_id_col_name), pl.lit("-"), pl.col(self.timestamp_col_name)
            )
            .str.strip_chars()
            .alias(self.pred_time_uuid_col_name)
        )

        _validate_col_name_columns_exist(obj=self)

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()

    def required_columns(self) -> Sequence[str]:
        return [self.entity_id_col_name]


@dataclass(frozen=True)
class SpecColumnError(Exception):
    description: str


@dataclass
class ValueFrame:
    """A frame that contains the values of a time series."""

    init_df: InitVar[InitDF_T]
    value_col_name: str
    entity_id_col_name: str = default_entity_id_col_name
    value_timestamp_col_name: str = "timestamp"

    def __post_init__(self, init_df: InitDF_T):
        self.df = _anyframe_to_lazyframe(init_df)
        _validate_col_name_columns_exist(obj=self)

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()


@dataclass
class TimestampValueFrame:
    """A frame that contains the values of a time series."""

    init_df: InitVar[InitDF_T]
    value_timestamp_col_name: str = "timestamp"
    entity_id_col_name: str = default_entity_id_col_name

    def __post_init__(self, init_df: InitDF_T):
        self.df = _anyframe_to_lazyframe(init_df)
        _validate_col_name_columns_exist(obj=self)

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()


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


class Aggregator(ABC):
    name: str

    @abstractmethod
    def __call__(self, column_name: str) -> pl.Expr:
        ...

    def new_col_name(self, previous_col_name: str) -> str:
        return f"{previous_col_name}_{self.name}"


@dataclass(frozen=True)
class LookPeriod:
    first: LookDistance
    last: LookDistance

    def __post_init__(self):
        if self.first >= self.last:
            raise ValueError(
                f"Invalid LookPeriod. The first value ({self.first}) must be smaller than the large value ({self.last})."
            )


def _lookdistance_to_normalised_lookperiod(
    lookdistance: LookDistance | tuple[LookDistance, LookDistance],
    direction: Literal["ahead", "behind"],
) -> LookPeriod:
    is_ahead = direction == "ahead"
    if isinstance(lookdistance, LookDistance):
        return LookPeriod(
            first=dt.timedelta(days=0) if is_ahead else -lookdistance,
            last=lookdistance if is_ahead else dt.timedelta(0),
        )
    return LookPeriod(
        first=lookdistance[0] if is_ahead else -lookdistance[1],
        last=lookdistance[1] if is_ahead else -lookdistance[0],
    )


LookDistances = Sequence[LookDistance | tuple[LookDistance, LookDistance]]


@dataclass
class PredictorSpec:
    value_frame: ValueFrame
    lookbehind_distances: InitVar[LookDistances]
    aggregators: Sequence[Aggregator]
    fallback: ValueType
    column_prefix: str = "pred"

    def __post_init__(self, lookbehind_distances: LookDistances):
        self.normalised_lookperiod = [
            _lookdistance_to_normalised_lookperiod(lookdistance=lookdistance, direction="behind")
            for lookdistance in lookbehind_distances
        ]
        _validate_col_name_columns_exist(obj=self)

    @property
    def df(self) -> pl.LazyFrame:
        return self.value_frame.df


@dataclass
class OutcomeSpec:
    value_frame: ValueFrame
    lookahead_distances: InitVar[LookDistances]
    aggregators: Sequence[Aggregator]
    fallback: ValueType
    column_prefix: str = "outc"

    def __post_init__(self, lookahead_distances: LookDistances):
        self.normalised_lookperiod = [
            _lookdistance_to_normalised_lookperiod(lookdistance=lookdistance, direction="ahead")
            for lookdistance in lookahead_distances
        ]
        _validate_col_name_columns_exist(obj=self)

    @property
    def df(self) -> pl.LazyFrame:
        return self.value_frame.df


@dataclass
class BooleanOutcomeSpec:
    init_frame: InitVar[TimestampValueFrame]
    lookahead_distances: LookDistances
    aggregators: Sequence[Aggregator]
    fallback: ValueType
    column_prefix: str = "outc"

    def __post_init__(self, init_frame: TimestampValueFrame):
        self.normalised_lookperiod = [
            _lookdistance_to_normalised_lookperiod(lookdistance=lookdistance, direction="ahead")
            for lookdistance in self.lookahead_distances
        ]

        self.value_frame = ValueFrame(
            init_df=init_frame.df.with_columns((pl.lit(1)).alias("value")),
            value_col_name="value",
            entity_id_col_name=init_frame.entity_id_col_name,
            value_timestamp_col_name=init_frame.value_timestamp_col_name,
        )

    @property
    def df(self) -> pl.LazyFrame:
        return self.value_frame.df


@dataclass
class TimedeltaFrame:
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


ValueSpecification = Union[PredictorSpec, OutcomeSpec, BooleanOutcomeSpec]


@dataclass(frozen=True)
class AggregatedFrame:
    df: pl.LazyFrame
    pred_time_uuid_col_name: str
    timestamp_col_name: str

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
