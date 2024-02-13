import datetime as dt

import numpy as np
import polars as pl
import polars.testing as polars_testing
import pytest
from timeseriesflattener.testing.utils_for_testing import str_to_pl_df

from timeseriesflattenerv2.aggregators import MaxAggregator, MeanAggregator

from . import flattener
from .feature_specs import PredictionTimeFrame, PredictorSpec, SlicedFrame, ValueFrame

FakePredictiontimeFrame = PredictionTimeFrame(
    init_df=pl.LazyFrame({"entity_id": [1], "pred_timestamp": ["2021-01-03"]})
)
FakeValueFrame = ValueFrame(
    init_df=pl.LazyFrame({"entity_id": [1], "value": [1], "timestamp": ["2021-01-01"]}),
    value_col_name="value",
)
FakePredictorSpec = PredictorSpec(
    value_frame=ValueFrame(
        init_df=pl.LazyFrame(
            {"entity_id": [1], "FakeValueColName": [1], "timestamp": ["2021-01-01"]}
        ),
        value_col_name="FakeValueColName",
    ),
    lookbehind_distances=[dt.timedelta(days=1)],
    aggregators=[MeanAggregator()],
    fallback=np.nan,
)


def assert_frame_equal(result: pl.DataFrame, expected: pl.DataFrame):
    polars_testing.assert_frame_equal(result, expected, check_dtype=False, check_column_order=False)


def test_flattener():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp
        1,2021-01-03"""
    )

    value_frame = str_to_pl_df(
        """entity_id,value,timestamp
        1,1,2021-01-01
        1,2,2021-01-02
        1,4,2021-01-03"""
    )

    result = flattener.Flattener(
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame.lazy())
    ).aggregate_timeseries(
        specs=[
            PredictorSpec(
                value_frame=ValueFrame(init_df=value_frame.lazy(), value_col_name="value"),
                lookbehind_distances=[dt.timedelta(days=1)],
                aggregators=[MeanAggregator()],
                fallback=np.nan,
            )
        ]
    )

    expected = str_to_pl_df(
        """pred_time_uuid,pred_value_within_1_days_mean_fallback_nan
1-2021-01-03 00:00:00.000000,3.0"""
    )

    assert_frame_equal(result.df.collect(), expected)


def test_eager_flattener():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp
        1,2021-01-03"""
    )

    value_frame = str_to_pl_df(
        """entity_id,value,timestamp
        1,1,2021-01-01
        1,2,2021-01-02
        1,4,2021-01-03"""
    )

    result = flattener.Flattener(
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame.lazy()), lazy=False
    ).aggregate_timeseries(
        specs=[
            PredictorSpec(
                value_frame=ValueFrame(init_df=value_frame.lazy(), value_col_name="value"),
                lookbehind_distances=[dt.timedelta(days=1)],
                aggregators=[MeanAggregator()],
                fallback=np.nan,
            )
        ]
    )

    expected = str_to_pl_df(
        """pred_time_uuid,pred_value_within_1_days_mean_fallback_nan
1-2021-01-03 00:00:00.000000,3.0"""
    )

    assert_frame_equal(result.df, expected)  # type: ignore


def test_keep_prediction_times_without_predictors():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp
        1,2021-01-03"""
    )

    value_frame = str_to_pl_df(
        """entity_id,value,timestamp
        2,1,2021-01-01"""
    )

    result = flattener.Flattener(
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame.lazy())
    ).aggregate_timeseries(
        specs=[
            PredictorSpec(
                value_frame=ValueFrame(init_df=value_frame.lazy(), value_col_name="value"),
                lookbehind_distances=[dt.timedelta(days=1)],
                aggregators=[MeanAggregator()],
                fallback=123,
            )
        ]
    )

    expected = str_to_pl_df(
        """pred_time_uuid,pred_value_within_1_days_mean_fallback_123
1-2021-01-03 00:00:00.000000,123.0"""
    )

    assert_frame_equal(result.df.collect(), expected)


def test_flattener_multiple_features():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp
        1,2021-01-03"""
    )

    value_frame = str_to_pl_df(
        """entity_id,value,timestamp
        1,1,2021-01-01
        1,2,2021-01-02
        1,4,2021-01-03"""
    )

    result = flattener.Flattener(
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame.lazy())
    ).aggregate_timeseries(
        specs=[
            PredictorSpec(
                value_frame=ValueFrame(
                    init_df=value_frame.rename({"value": "value_1"}).lazy(),
                    value_col_name="value_1",
                ),
                lookbehind_distances=[dt.timedelta(days=1)],
                aggregators=[MeanAggregator()],
                fallback=np.nan,
            ),
            PredictorSpec(
                value_frame=ValueFrame(
                    init_df=value_frame.rename({"value": "value_2"}).lazy(),
                    value_col_name="value_2",
                ),
                lookbehind_distances=[dt.timedelta(days=1)],
                aggregators=[MeanAggregator()],
                fallback=np.nan,
            ),
        ]
    )

    expected = str_to_pl_df(
        """pred_time_uuid,pred_value_1_within_1_days_mean_fallback_nan,pred_value_2_within_1_days_mean_fallback_nan
1-2021-01-03 00:00:00.000000,3.0,3.0"""
    )

    assert_frame_equal(result.df.collect(), expected)


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

    result = flattener._get_timedelta_frame(
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame.lazy()),
        value_frame=ValueFrame(init_df=value_frame.lazy(), value_col_name="value"),
    )

    assert result.get_timedeltas() == expected_timedeltas


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

    aggregated_values = flattener._aggregate_within_slice(
        sliced_frame=sliced_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean_fallback_0
1-2021-01-03,1.5
2-2021-01-03,3"""
    )

    assert_frame_equal(aggregated_values[0].df.collect(), expected)


def test_aggregate_over_fallback():
    sliced_frame = SlicedFrame(
        init_df=pl.LazyFrame(
            {"pred_time_uuid": ["1-2021-01-03", "1-2021-01-03"], "value": [None, None]}
        ),
        value_col_name="value",
    )

    aggregated_values = flattener._aggregate_within_slice(
        sliced_frame=sliced_frame, aggregators=[MeanAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean_fallback_0
1-2021-01-03,0"""
    )

    assert_frame_equal(aggregated_values[0].df.collect(), expected)


def test_multiple_aggregatrs():
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

    aggregated_values = flattener._aggregate_within_slice(
        sliced_frame=sliced_frame, aggregators=[MeanAggregator(), MaxAggregator()], fallback=0
    )

    expected = str_to_pl_df(
        """pred_time_uuid,value_mean_fallback_0,value_max_fallback_0
1-2021-01-03,1.5,2
2-2021-01-03,3,4"""
    )

    assert_frame_equal(
        flattener.horizontally_concatenate_dfs(
            [agg.df for agg in aggregated_values], pred_time_uuid_col_name="pred_time_uuid"
        ).collect(),
        expected,
    )


def test_error_if_conflicting_value_col_names():
    with pytest.raises(flattener.SpecError, match=".*unique.*"):
        flattener.Flattener(predictiontime_frame=FakePredictiontimeFrame).aggregate_timeseries(
            specs=[FakePredictorSpec, FakePredictorSpec]
        )
