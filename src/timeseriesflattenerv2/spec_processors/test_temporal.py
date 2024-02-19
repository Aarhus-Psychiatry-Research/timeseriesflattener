from __future__ import annotations

import datetime as dt

import polars as pl
from timeseriesflattener.testing.utils_for_testing import str_to_pl_df

import timeseriesflattenerv2.spec_processors.temporal as process_spec
import timeseriesflattenerv2.spec_processors.timedelta

from .._intermediary_frames import TimeDeltaFrame, TimeMaskedFrame
from ..aggregators import MaxAggregator, MeanAggregator
from ..feature_specs.meta import LookPeriod, ValueFrame
from ..feature_specs.prediction_times import PredictionTimeFrame
from ..feature_specs.timedelta import TimeDeltaSpec
from ..feature_specs.timestamp_frame import TimestampValueFrame
from ..test_flattener import assert_frame_equal


def test_aggregate_over_fallback():
    masked_frame = TimeMaskedFrame(
        validate_cols_exist=False,
        init_df=pl.LazyFrame(
            {
                "pred_time_uuid": ["1-2021-01-03", "1-2021-01-03"],
                "value": [None, None],
                "timestamp": ["2021-01-01", "2021-01-02"],
            }
        ),
        value_col_name="value",
    )

    aggregated_values = process_spec._aggregate_masked_frame(
        masked_frame=masked_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean_fallback_0
1-2021-01-03,0"""
    )

    assert_frame_equal(aggregated_values.collect(), expected)


def test_aggregate_with_null():
    masked_frame = TimeMaskedFrame(
        validate_cols_exist=False,
        init_df=pl.LazyFrame(
            {
                "pred_time_uuid": ["1-2021-01-03", "1-2021-01-03"],
                "value": [1, None],
                "timestamp": ["2021-01-01", "2021-01-02"],
            }
        ),
        value_col_name="value",
    )

    aggregated_values = process_spec._aggregate_masked_frame(
        masked_frame=masked_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean_fallback_0
1-2021-01-03,1"""
    )

    assert_frame_equal(aggregated_values.collect(), expected)


def test_aggregate_within_slice():
    masked_frame = TimeMaskedFrame(
        validate_cols_exist=False,
        init_df=str_to_pl_df(
            """pred_time_uuid,value
1-2021-01-03,1
1-2021-01-03,2
2-2021-01-03,2
2-2021-01-03,4"""
        ).lazy(),
        value_col_name="value",
    )

    aggregated_values = process_spec._aggregate_masked_frame(
        masked_frame=masked_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean_fallback_0
1-2021-01-03,1.5
2-2021-01-03,3"""
    )

    assert_frame_equal(aggregated_values.collect(), expected)


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
    timedelta_frame = TimeDeltaFrame(
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
                "value_timestamp": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
            }
        ),
        value_col_name="is_null",
        value_timestamp_col_name="value_timestamp",
    )

    result = process_spec._mask_outside_lookperiod(
        timedelta_frame=timedelta_frame,
        lookperiod=LookPeriod(first=dt.timedelta(days=-2), last=dt.timedelta(days=0)),
        column_prefix="pred",
        value_col_name="value",
    ).collect()

    from polars.testing import assert_series_equal

    assert_series_equal(
        result.get_column("pred_value_within_0_to_2_days"),
        timedelta_frame.df.collect().get_column("is_null"),
        check_names=False,
        check_dtype=False,
    )


def test_multiple_aggregators():
    masked_frame = TimeMaskedFrame(
        validate_cols_exist=False,
        init_df=str_to_pl_df(
            """pred_time_uuid,value
1-2021-01-03,1
1-2021-01-03,2
2-2021-01-03,2
2-2021-01-03,4"""
        ).lazy(),
        value_col_name="value",
    )

    aggregated_values = process_spec._aggregate_masked_frame(
        masked_frame=masked_frame, aggregators=[MeanAggregator(), MaxAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean_fallback_0,value_max_fallback_0
1-2021-01-03,1.5,2
2-2021-01-03,3,4"""
    )

    assert_frame_equal(aggregated_values.collect(), expected)


def test_process_time_from_event_spec():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp
        1,2021-01-01
        2,2021-01-01
        """
    )

    value_frame = str_to_pl_df(
        """entity_id,timestamp
        1,2020-01-01"""
    )

    result = timeseriesflattenerv2.spec_processors.timedelta.process_timedelta_spec(
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame),
        spec=TimeDeltaSpec(
            init_frame=TimestampValueFrame(init_df=value_frame),
            output_name="age",
            fallback=0,
            time_format="years",
        ),
    )

    expected = str_to_pl_df(
        """pred_time_uuid,pred_age_years_fallback_0
1-2021-01-01 00:00:00.000000,1.002053388090349
2-2021-01-01 00:00:00.000000,0
       """
    )

    assert_frame_equal(result.collect(), expected)
