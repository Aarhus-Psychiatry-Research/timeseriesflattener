from dataclasses import dataclass

import polars as pl

from .feature_specs import AggregatedValueFrame, Aggregator, SlicedFrame


@dataclass
class MeanAggregator(Aggregator):
    name: str = "mean"

    def apply(self, sliced_frame: SlicedFrame, column_name: str) -> AggregatedValueFrame:
        df = sliced_frame.df.group_by(
            sliced_frame.pred_time_uuid_col_name, maintain_order=True
        ).agg(pl.col(column_name).mean())
        # TODO: Figure out how to standardise the output column names

        return AggregatedValueFrame(df=df)
