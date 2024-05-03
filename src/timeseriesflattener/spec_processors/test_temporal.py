from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
from timeseriesflattener.testing.utils_for_testing import str_to_pl_df

import timeseriesflattener.spec_processors.temporal as process_spec
import timeseriesflattener.spec_processors.timedelta
from timeseriesflattener.feature_specs.predictor import PredictorSpec

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
                "prediction_time_uuid": ["1-2021-01-03", "1-2021-01-03"],
                "value": [None, None],
                "timestamp": ["2021-01-01", "2021-01-02"],
            }
        ),
        value_col_names=["value"],
    )

    aggregated_values = process_spec._aggregate_masked_frame(
        masked_frame=masked_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """prediction_time_uuid,value_mean_fallback_0
1-2021-01-03,0"""
    )

    assert_frame_equal(aggregated_values.collect(), expected)


def test_aggregate_with_null():
    masked_frame = TimeMaskedFrame(
        validate_cols_exist=False,
        init_df=pl.LazyFrame(
            {
                "prediction_time_uuid": ["1-2021-01-03", "1-2021-01-03"],
                "value": [1, None],
                "timestamp": ["2021-01-01", "2021-01-02"],
            }
        ),
        value_col_names=["value"],
    )

    aggregated_values = process_spec._aggregate_masked_frame(
        masked_frame=masked_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """prediction_time_uuid,value_mean_fallback_0
1-2021-01-03,1"""
    )

    assert_frame_equal(aggregated_values.collect(), expected)


def test_aggregate_within_slice():
    masked_frame = TimeMaskedFrame(
        validate_cols_exist=False,
        init_df=str_to_pl_df(
            """prediction_time_uuid,value
1-2021-01-03,1
1-2021-01-03,2
2-2021-01-03,2
2-2021-01-03,4"""
        ).lazy(),
        value_col_names=["value"],
    )

    aggregated_values = process_spec._aggregate_masked_frame(
        masked_frame=masked_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """prediction_time_uuid,value_mean_fallback_0
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
        value_frame=ValueFrame(init_df=value_frame.lazy()),
    )

    assert result.get_timedeltas() == expected_timedeltas


def test_get_timedelta_frame_same_timestamp_col_names():
    pred_frame = str_to_pl_df(
        """entity_id,timestamp
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
        predictiontime_frame=PredictionTimeFrame(
            init_df=pred_frame.lazy(), timestamp_col_name="timestamp"
        ),
        value_frame=ValueFrame(init_df=value_frame.lazy()),
    )

    assert result.get_timedeltas() == expected_timedeltas


def test_slice_without_any_within_window():
    timedelta_frame = TimeDeltaFrame(
        df=pl.LazyFrame(
            {
                "prediction_time_uuid": [1, 1, 2, 2],
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
        value_col_names=["is_null"],
        value_timestamp_col_name="value_timestamp",
    )

    result = process_spec._mask_outside_lookperiod(
        timedelta_frame=timedelta_frame,
        lookperiod=LookPeriod(first=dt.timedelta(days=-2), last=dt.timedelta(days=0)),
        column_prefix="pred",
        value_col_names=["is_null"],
    ).collect()

    from polars.testing import assert_series_equal

    assert_series_equal(
        result.get_column("pred_is_null_within_0_to_2_days"),
        timedelta_frame.df.collect().get_column("is_null"),
        check_names=False,
        check_dtype=False,
    )


def test_multiple_aggregators():
    masked_frame = TimeMaskedFrame(
        validate_cols_exist=False,
        init_df=str_to_pl_df(
            """prediction_time_uuid,value
1-2021-01-03,1
1-2021-01-03,2
2-2021-01-03,2
2-2021-01-03,4"""
        ).lazy(),
        value_col_names=["value"],
    )

    aggregated_values = process_spec._aggregate_masked_frame(
        masked_frame=masked_frame, aggregators=[MeanAggregator(), MaxAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """prediction_time_uuid,value_mean_fallback_0,value_max_fallback_0
1-2021-01-03,1.5,2
2-2021-01-03,3,4"""
    )

    assert_frame_equal(aggregated_values.collect(), expected)


def test_masking_multiple_values_multiple_aggregators():
    masked_frame = TimeMaskedFrame(
        validate_cols_exist=False,
        init_df=str_to_pl_df(
            """prediction_time_uuid,value_1,value_2
1-2021-01-03,1,np.nan
1-2021-01-03,2,np.nan
2-2021-01-03,2,np.nan
2-2021-01-03,4,np.nan"""
        ).lazy(),
        value_col_names=["value_1", "value_2"],
    )

    aggregated_values = process_spec._aggregate_masked_frame(
        masked_frame=masked_frame, aggregators=[MeanAggregator(), MaxAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """prediction_time_uuid,value_1_mean_fallback_0,value_2_mean_fallback_0,value_1_max_fallback_0,value_2_max_fallback_0
1-2021-01-03,1.5,0,2,0
2-2021-01-03,3,0,4,0"""
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

    result = timeseriesflattener.spec_processors.timedelta.process_timedelta_spec(
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame),
        spec=TimeDeltaSpec(
            init_frame=TimestampValueFrame(init_df=value_frame),
            output_name="age",
            fallback=0,
            time_format="years",
        ),
    )

    expected = str_to_pl_df(
        """prediction_time_uuid,pred_age_years_fallback_0
1-2021-01-01 00:00:00.000000,1.002053388090349
2-2021-01-01 00:00:00.000000,0
       """
    )

    assert_frame_equal(result.collect(), expected)


def test_process_temporal_spec_multiple_values():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp
        1,2021-01-01"""
    )
    value_frame = str_to_pl_df(
        """entity_id,timestamp,value_1,value_2
        1,2021-01-01,1,2"""
    )

    result = process_spec.process_temporal_spec(
        spec=PredictorSpec(
            value_frame=ValueFrame(init_df=value_frame.lazy()),
            lookbehind_distances=[dt.timedelta(days=1)],
            aggregators=[MeanAggregator()],
            fallback=0,
        ),
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame.lazy()),
    )

    expected = str_to_pl_df(
        """prediction_time_uuid,pred_value_1_within_0_to_1_days_mean_fallback_0,pred_value_2_within_0_to_1_days_mean_fallback_0
1-2021-01-01 00:00:00.000000,1,2"""
    )
    assert_frame_equal(result.collect(), expected)


def test_sliding_window():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp
                              1,2011-01-01,
                              1,2014-01-01,
                              1,2016-01-01,
                              1,2018-01-01,
                              1,2020-01-01,
                              1,2022-01-01,"""  # 2012 year without prediction times
    )

    value_frame = str_to_pl_df(
        """entity_id,timestamp,value
                                1,2011-01-01,1
                                1,2012-01-01,2
                                1,2013-01-01,3
                                1,2014-01-01,4
                                1,2015-01-01,5
                                1,2016-01-01,6
                                1,2019-01-01,9
                                1,2021-01-01,11 
                                1,2021-01-01,12"""  # 2021 year with multiple values
    )  # 2022 year with no values

    result = process_spec.process_temporal_spec(
        spec=PredictorSpec(
            value_frame=ValueFrame(init_df=value_frame.lazy()),
            lookbehind_distances=[
                dt.timedelta(days=10),
                dt.timedelta(days=365),
            ],  # test multiple lookperiods
            aggregators=[MeanAggregator()],
            fallback=0,
        ),
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame.lazy()),
        step_size=dt.timedelta(days=365),
    )

    expected = str_to_pl_df(
        """prediction_time_uuid,pred_value_within_0_to_10_days_mean_fallback_0,pred_value_within_0_to_365_days_mean_fallback_0
1-2011-01-01 00:00:00.000000,1.0,1.0
1-2014-01-01 00:00:00.000000,4.0,3.5
1-2016-01-01 00:00:00.000000,6.0,5.5
1-2018-01-01 00:00:00.000000,0.0,0.0
1-2020-01-01 00:00:00.000000,0.0,9.0
1-2022-01-01 00:00:00.000000,0.0,11.5"""
    )

    assert_frame_equal(result.collect(), expected)
