import datetime as dt

import polars as pl
import polars.testing as polars_testing
from timeseriesflattener.testing.utils_for_testing import str_to_pl_df

from timeseriesflattenerv2.aggregators import MaxAggregator, MeanAggregator

from . import flattener
from .feature_specs import (
    AggregatedFrame,
    PredictionTimeFrame,
    PredictorSpec,
    SlicedFrame,
    ValueFrame,
)


def assert_frame_equal(left: pl.DataFrame, right: pl.DataFrame):
    polars_testing.assert_frame_equal(left, right, check_dtype=False, check_column_order=False)


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
                fallback="NaN",
            )
        ]
    )

    assert isinstance(result, AggregatedFrame)


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


def test_aggregate_within_slice():
    sliced_frame = SlicedFrame(
        df=str_to_pl_df(
            """pred_time_uuid,value
1-2021-01-03,1
1-2021-01-03,2
2-2021-01-03,2
2-2021-01-03,4"""
        ).lazy()
    )

    aggregated_values = flattener._aggregate_within_slice(
        sliced_frame=sliced_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean
1-2021-01-03,1.5
2-2021-01-03,3"""
    )

    assert_frame_equal(aggregated_values[0].df.collect(), expected)


def test_aggregate_over_fallback():
    sliced_frame = SlicedFrame(
        df=pl.LazyFrame({"pred_time_uuid": ["1-2021-01-03", "1-2021-01-03"], "value": [None, None]})
    )

    aggregated_values = flattener._aggregate_within_slice(
        sliced_frame=sliced_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean
1-2021-01-03,0"""
    )

    assert_frame_equal(aggregated_values[0].df.collect(), expected)


def test_multiple_aggregatrs():
    sliced_frame = SlicedFrame(
        df=str_to_pl_df(
            """pred_time_uuid,value
1-2021-01-03,1
1-2021-01-03,2
2-2021-01-03,2
2-2021-01-03,4"""
        ).lazy()
    )

    aggregated_values = flattener._aggregate_within_slice(
        sliced_frame=sliced_frame, aggregators=[MeanAggregator(), MaxAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean,value_max
1-2021-01-03,1.5,2
2-2021-01-03,3,4"""
    )

    assert_frame_equal(
        flattener.horizontally_concatenate_dfs(
            [agg.df for agg in aggregated_values], pred_time_uuid_col_name="pred_time_uuid"
        ).collect(),
        expected,
    )
