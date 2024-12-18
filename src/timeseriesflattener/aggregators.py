from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Literal, Sequence

import polars as pl
from attr import dataclass


def validate_compatible_fallback_type_for_aggregator(
    aggregator: Aggregator, fallback: str | int | float | None
) -> None:
    try:
        pl.Series([aggregator.output_type()]).fill_null(fallback)
    except:
        raise ValueError(
            f"Invalid fallback value {fallback} for aggregator {aggregator.__class__.__name__}. Fallback of type {type(fallback)} is not compatible with the aggregator's output type of {type(aggregator.output_type)}."
        )


AggregatorName = Literal[
    "bool",
    "change_per_day",
    "count",
    "unique_count",
    "has_values",
    "max",
    "mean",
    "min",
    "slope",
    "sum",
    "variance",
]


def strings_to_aggregators(
    aggregator_names: Sequence[AggregatorName], timestamp_col_name: str
) -> Sequence[Aggregator]:
    return [
        string_to_aggregator(name, timestamp_col_name=timestamp_col_name)
        for name in aggregator_names
    ]


def string_to_aggregator(aggregator_name: AggregatorName, timestamp_col_name: str) -> Aggregator:
    str2aggr: dict[AggregatorName, Aggregator] = {
        "bool": HasValuesAggregator(),
        "change_per_day": SlopeAggregator(timestamp_col_name=timestamp_col_name),
        "count": CountAggregator(),
        "unique_count": UniqueCountAggregator(),
        "has_values": HasValuesAggregator(),
        "max": MaxAggregator(),
        "mean": MeanAggregator(),
        "min": MinAggregator(),
        "slope": SlopeAggregator(timestamp_col_name=timestamp_col_name),
        "sum": SumAggregator(),
        "variance": VarianceAggregator(),
    }

    return str2aggr[aggregator_name]


class Aggregator(ABC):
    name: str
    output_type: type[float | int | bool]

    @abstractmethod
    def __call__(self, column_name: str) -> pl.Expr: ...

    def new_col_name(self, previous_col_name: str) -> str:
        return f"{previous_col_name}_{self.name}"


class MinAggregator(Aggregator):
    """Returns the minimum value in the look window."""

    name: str = "min"
    output_type = float

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).min().alias(self.new_col_name(column_name))


class MaxAggregator(Aggregator):
    """Returns the maximum value in the look window."""

    name: str = "max"
    output_type = float

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).max().alias(self.new_col_name(column_name))


class MeanAggregator(Aggregator):
    """Returns the mean value in the look window."""

    name: str = "mean"
    output_type = float

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).mean().alias(self.new_col_name(column_name))


class CountAggregator(Aggregator):
    """Returns the count of non-null values in the look window."""

    name: str = "count"
    output_type = int

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).count().alias(self.new_col_name(column_name))


class UniqueCountAggregator(Aggregator):
    """Returns the count of non-null values in the look window."""

    name: str = "unique_count"
    output_type = int

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).n_unique().alias(self.new_col_name(column_name))


@dataclass(frozen=True)
class EarliestAggregator(Aggregator):
    """Returns the earliest value in the look window."""

    timestamp_col_name: str
    name: str = "earliest"
    output_type = float

    def __call__(self, column_name: str) -> pl.Expr:
        return (
            pl.col(column_name)
            .filter(pl.col(self.timestamp_col_name) == pl.col(self.timestamp_col_name).min())
            .first()
            .alias(self.new_col_name(column_name))
        )


@dataclass(frozen=True)
class LatestAggregator(Aggregator):
    """Returns the latest value in the look window"""

    timestamp_col_name: str
    name: str = "latest"
    output_type = float

    def __call__(self, column_name: str) -> pl.Expr:
        return (
            pl.col(column_name)
            .filter(pl.col(self.timestamp_col_name) == pl.col(self.timestamp_col_name).max())
            .first()
            .alias(self.new_col_name(column_name))
        )


class SumAggregator(Aggregator):
    """Returns the sum of all values in the look window."""

    name: str = "sum"
    output_type = float

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).sum().alias(self.new_col_name(column_name))


class VarianceAggregator(Aggregator):
    """Returns the variance of the values in the look window"""

    name: str = "var"
    output_type = float

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).var().alias(self.new_col_name(column_name))


class HasValuesAggregator(Aggregator):
    """Examines whether any values exist in the look window. If so, returns True, else False."""

    name: str = "bool"
    output_type = bool

    def __call__(self, column_name: str) -> pl.Expr:
        return (
            pl.when(pl.col(column_name).is_not_null())
            .then(1)
            .otherwise(0)
            .cast(pl.Boolean)
            .any()
            .alias(self.new_col_name(column_name))
        )


@dataclass(frozen=True)
class SlopeAggregator(Aggregator):
    """Returns the slope (i.e. the correlation between the timestamp and the value) in the look window."""

    timestamp_col_name: str
    name: str = "slope"
    output_type = float

    def __call__(self, column_name: str) -> pl.Expr:
        # Convert to days for the slope. Arbitrarily chosen to be the number of days since 1970-01-01.
        x_col = (pl.col(self.timestamp_col_name) - dt.datetime(1970, 1, 1)).dt.total_days()
        y_col = pl.col(column_name)

        numerator = pl.corr(x_col, y_col, propagate_nans=True) * y_col.std()
        denominator = x_col.std()
        return (numerator / denominator).alias(self.new_col_name(column_name))
