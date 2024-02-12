from dataclasses import dataclass

import polars as pl
from polars.lazyframe.group_by import LazyGroupBy

from timeseriesflattenerv2.feature_specs import Aggregator

from .feature_specs import AggregatedValueFrame


@dataclass
class MeanAggregator(Aggregator):
    def apply(self, grouped_frame: LazyGroupBy, column_name: str) -> AggregatedValueFrame:
        value_col_name = f"{column_name}_mean"
        df = grouped_frame.agg(pl.col(column_name).mean().alias(value_col_name))
        return AggregatedValueFrame(df=df, value_col_name=value_col_name)


@dataclass
class MaxAggregator(Aggregator):
    def apply(self, grouped_frame: LazyGroupBy, column_name: str) -> AggregatedValueFrame:
        value_col_name = f"{column_name}_max"
        df = grouped_frame.agg(pl.col(column_name).max().alias(value_col_name))

        return AggregatedValueFrame(df=df, value_col_name=value_col_name)
