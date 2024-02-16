import datetime as dt

import polars as pl
from attr import dataclass

from timeseriesflattenerv2.feature_specs import Aggregator


class MinAggregator(Aggregator):
    name: str = "min"

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).min().alias(self.new_col_name(column_name))


class MaxAggregator(Aggregator):
    name: str = "max"

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).max().alias(self.new_col_name(column_name))


class MeanAggregator(Aggregator):
    name: str = "mean"

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).mean().alias(self.new_col_name(column_name))


class CountAggregator(Aggregator):
    name: str = "count"

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).count().alias(self.new_col_name(column_name))


@dataclass(frozen=True)
class EarliestAggregator(Aggregator):
    timestamp_col_name: str
    name: str = "earliest"

    def __call__(self, column_name: str) -> pl.Expr:
        return (
            pl.col(column_name)
            .filter(pl.col(self.timestamp_col_name) == pl.col(self.timestamp_col_name).min())
            .first()
            .alias(self.new_col_name(column_name))
        )


@dataclass(frozen=True)
class LatestAggregator(Aggregator):
    timestamp_col_name: str
    name: str = "latest"

    def __call__(self, column_name: str) -> pl.Expr:
        return (
            pl.col(column_name)
            .filter(pl.col(self.timestamp_col_name) == pl.col(self.timestamp_col_name).max())
            .first()
            .alias(self.new_col_name(column_name))
        )


class SumAggregator(Aggregator):
    name: str = "sum"

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).sum().alias(self.new_col_name(column_name))


class VarianceAggregator(Aggregator):
    name: str = "var"

    def __call__(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).var().alias(self.new_col_name(column_name))


class HasValuesAggregator(Aggregator):
    """Examines whether any values exist in the column. If so, returns 1, else 0."""

    name: str = "bool"

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
    timestamp_col_name: str
    name: str = "slope"

    def __call__(self, column_name: str) -> pl.Expr:
        # Convert to days for the slope. Arbitrarily chosen to be the number of days since 1970-01-01.
        x_col = (pl.col(self.timestamp_col_name) - dt.datetime(1970, 1, 1)).dt.days()
        y_col = pl.col(column_name)

        numerator = pl.corr(x_col, y_col, propagate_nans=True) * y_col.std()
        denominator = x_col.std()
        return (numerator / denominator).alias(self.new_col_name(column_name))
