import datetime as dt

import polars as pl
from timeseriesflattener.testing.utils_for_testing import str_to_pl_df

import timeseriesflattenerv2._process_spec as process_spec

from ._horisontally_concat import horizontally_concatenate_dfs
from .aggregators import MaxAggregator, MeanAggregator
from .feature_specs import PredictionTimeFrame, SlicedFrame, TimedeltaFrame, ValueFrame
from .test_flattener import assert_frame_equal


def test_aggregate_over_fallback():
    sliced_frame = SlicedFrame(
        init_df=pl.LazyFrame(
            {"pred_time_uuid": ["1-2021-01-03", "1-2021-01-03"], "value": [None, None]}
        ),
        value_col_name="value",
    )

    aggregated_values = process_spec._aggregate_within_slice(
        sliced_frame=sliced_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean_fallback_0
1-2021-01-03,0"""
    )

    assert_frame_equal(aggregated_values[0].df.collect(), expected)


def test_aggregate_with_null():
    sliced_frame = SlicedFrame(
        init_df=pl.LazyFrame(
            {"pred_time_uuid": ["1-2021-01-03", "1-2021-01-03"], "value": [1, None]}
        ),
        value_col_name="value",
    )

    aggregated_values = process_spec._aggregate_within_slice(
        sliced_frame=sliced_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean_fallback_0
1-2021-01-03,1"""
    )

    assert_frame_equal(aggregated_values[0].df.collect(), expected)


def test_aggregate_within_slice():
    sliced_frame = SlicedFrame(
        init_df=str_to_pl_df(
            """pred_time_uuid,value
1-2021-01-03,1
1-2021-01-03,2
2-2021-01-03,2
2-2021-01-03,4"""
        ).lazy(),
        value_col_name="value",
    )

    aggregated_values = process_spec._aggregate_within_slice(
        sliced_frame=sliced_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean_fallback_0
1-2021-01-03,1.5
2-2021-01-03,3"""
    )

    assert_frame_equal(aggregated_values[0].df.collect(), expected)


def test_get_timedelta_frame():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp
        1,2021-01-03"""
    )

    value_frame = str_to_pl_df(
        """entity_id,value,timestamp
        1,1,2021-01-01
        1,2,2021-01-02
        1,3,2021-01-03"""
    )

    expected_timedeltas = [dt.timedelta(days=-2), dt.timedelta(days=-1), dt.timedelta(days=0)]

    result = process_spec._get_timedelta_frame(
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame.lazy()),
        value_frame=ValueFrame(init_df=value_frame.lazy(), value_col_name="value"),
    )

    assert result.get_timedeltas() == expected_timedeltas


def test_slice_without_any_within_window():
    timedelta_frame = TimedeltaFrame(
        df=pl.LazyFrame(
            {
                "pred_time_uuid": [1, 1, 2, 2],
                "time_from_prediction_to_value": [
                    dt.timedelta(days=1),  # Outside the lookbehind
                    dt.timedelta(days=-1),  # Inside the lookbehind
                    dt.timedelta(days=-2.1),  # Outside the lookbehind
                    dt.timedelta(days=-2.1),  # Outside the lookbehind
                ],
                "is_null": [None, 0, None, None],
            }
        ),
        value_col_name="is_null",
    )

    result = process_spec._slice_frame(
        timedelta_frame=timedelta_frame,
        lookdistance=dt.timedelta(days=-2),
        column_prefix="pred",
        value_col_name="value",
    ).collect()

    from polars.testing import assert_series_equal

    assert_series_equal(
        result.get_column("pred_value_within_2_days"),
        timedelta_frame.df.collect().get_column("is_null"),
        check_names=False,
        check_dtype=False,
    )


def test_multiple_aggregators():
    sliced_frame = SlicedFrame(
        init_df=str_to_pl_df(
            """pred_time_uuid,value
1-2021-01-03,1
1-2021-01-03,2
2-2021-01-03,2
2-2021-01-03,4"""
        ).lazy(),
        value_col_name="value",
    )

    aggregated_values = process_spec._aggregate_within_slice(
        sliced_frame=sliced_frame, aggregators=[MeanAggregator(), MaxAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean_fallback_0,value_max_fallback_0
1-2021-01-03,1.5,2
2-2021-01-03,3,4"""
    )

    assert_frame_equal(
        horizontally_concatenate_dfs(
            [agg.df for agg in aggregated_values], pred_time_uuid_col_name="pred_time_uuid"
        ).collect(),
        expected,
    )
