import datetime as dt
from dataclasses import dataclass

import polars as pl
from timeseriesflattener.testing.utils_for_testing import str_to_pl_df

from . import flattener
from .feature_specs import (
    AggregatedValueFrame,
    Aggregator,
    PredictionTimeFrame,
    PredictorSpec,
    SlicedFrame,
    ValueFrame,
)


@dataclass
class MeanAggregator(Aggregator):
    name: str = "mean"

    def apply(self, sliced_frame: SlicedFrame, column_name: str) -> AggregatedValueFrame:
        df = sliced_frame.df.group_by(pl.col(sliced_frame.pred_time_uuid_col_name)).agg(
            pl.col(column_name).mean().alias(column_name)
        )
        # TODO: Figure out how to standardise the output column names

        return AggregatedValueFrame(df=df)


def test_flattener():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp
        1,2021-01-03"""
    )

    value_frame = str_to_pl_df(
        """entity_id,value,value_timestamp
        1,1,2021-01-01
        1,2,2021-01-02
        1,3,2021-01-03"""
    )

    result = flattener.Flattener(
        predictiontime_frame=PredictionTimeFrame(df=pred_frame.lazy())
    ).aggregate_timeseries(
        specs=[
            PredictorSpec(
                value_frame=ValueFrame(df=value_frame.lazy(), value_type="test_value"),
                lookbehind_distances=[dt.timedelta(days=1)],
                aggregators=[MeanAggregator()],
                fallbacks=["NaN"],
            )
        ]
    )

    assert isinstance(result, AggregatedValueFrame)


def test_get_timedelta_frame():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp
        1,2021-01-03"""
    )

    value_frame = str_to_pl_df(
        """entity_id,value,value_timestamp
        1,1,2021-01-01
        1,2,2021-01-02
        1,3,2021-01-03"""
    )

    expected_timedeltas = [dt.timedelta(days=-2), dt.timedelta(days=-1), dt.timedelta(days=0)]

    result = flattener._get_timedelta_frame(
        predictiontime_frame=PredictionTimeFrame(df=pred_frame.lazy()),
        value_frame=ValueFrame(df=value_frame.lazy(), value_type="test_value"),
    )

    assert result.get_timedeltas() == expected_timedeltas
