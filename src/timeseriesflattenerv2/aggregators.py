import polars as pl

from timeseriesflattenerv2.feature_specs import Aggregator


class MeanAggregator(Aggregator):
    def __call__(self, column_name: str) -> pl.Expr:
        value_col_name = f"{column_name}_mean"
        return pl.col(column_name).mean().alias(value_col_name)


class MaxAggregator(Aggregator):
    def __call__(self, column_name: str) -> pl.Expr:
        value_col_name = f"{column_name}_max"
        return pl.col(column_name).max().alias(value_col_name)
