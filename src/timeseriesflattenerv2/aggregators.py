import polars as pl

from timeseriesflattenerv2.feature_specs import Aggregator


class MinAggregator(Aggregator):
    def __call__(self, column_name: str) -> pl.Expr:
        value_col_name = f"{column_name}_min"
        return pl.col(column_name).min().alias(value_col_name)


class MaxAggregator(Aggregator):
    def __call__(self, column_name: str) -> pl.Expr:
        value_col_name = f"{column_name}_max"
        return pl.col(column_name).max().alias(value_col_name)


class MeanAggregator(Aggregator):
    def __call__(self, column_name: str) -> pl.Expr:
        value_col_name = f"{column_name}_mean"
        return pl.col(column_name).mean().alias(value_col_name)


class CountAggregator(Aggregator):
    def __call__(self, column_name: str) -> pl.Expr:
        value_col_name = f"{column_name}_count"
        return pl.col(column_name).count().alias(value_col_name)
