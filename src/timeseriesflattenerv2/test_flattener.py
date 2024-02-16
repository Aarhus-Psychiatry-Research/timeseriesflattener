import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl
import polars.testing as polars_testing
import pytest
from timeseriesflattener.testing.utils_for_testing import str_to_pl_df

from timeseriesflattenerv2.aggregators import EarliestAggregator, MeanAggregator

from . import flattener
from .feature_specs import (
    OutcomeSpec,
    PredictionTimeFrame,
    PredictorSpec,
    SpecColumnError,
    ValueFrame,
)

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


@dataclass(frozen=True)
class FlattenerExample:
    should: str  # What the example is testing
    lazy: bool = True
    n_workers: int | None = None


@pytest.mark.parametrize(
    ("example"),
    [
        FlattenerExample(should="work with lazy flattening", lazy=True),
        FlattenerExample(should="work with eager flattening", lazy=False),
        FlattenerExample(should="work with multiprocessing", n_workers=2),
    ],
    ids=lambda example: example.should,
)
def test_flattener(example: FlattenerExample):
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
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame.lazy()),
        compute_lazily=example.lazy,
        n_workers=example.n_workers,
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
        """pred_time_uuid,pred_value_within_0_to_1_days_mean_fallback_nan
1-2021-01-03 00:00:00.000000,3.0"""
    )

    assert_frame_equal(result.collect(), expected)


def test_keep_prediction_times_without_predictors():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp,pred_time_uuid
        1,2021-01-03"""
    )

    value_frame = str_to_pl_df(
        """entity_id,value,timestamp
        1,1,2021-01-01"""
    )

    result = flattener.Flattener(
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame.lazy())
    ).aggregate_timeseries(
        specs=[
            PredictorSpec(
                value_frame=ValueFrame(init_df=value_frame.lazy(), value_col_name="value"),
                lookbehind_distances=[dt.timedelta(days=1)],
                aggregators=[MeanAggregator(), EarliestAggregator(timestamp_col_name="timestamp")],
                fallback=123,
            )
        ]
    )

    expected = pl.DataFrame(
        {
            "pred_time_uuid": ["1-2021-01-03 00:00:00.000000"],
            "pred_value_within_0_to_1_days_mean_fallback_123": [123],
            "pred_value_within_0_to_1_days_earliest_fallback_123": [123],
        }
    )

    assert_frame_equal(result.collect(), expected)


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
        """pred_time_uuid,pred_value_1_within_0_to_1_days_mean_fallback_nan,pred_value_2_within_0_to_1_days_mean_fallback_nan
1-2021-01-03 00:00:00.000000,3.0,3.0"""
    )

    assert_frame_equal(result.collect(), expected)


def test_error_if_conflicting_value_col_names():
    with pytest.raises(flattener.SpecError, match=".*unique.*"):
        flattener.Flattener(predictiontime_frame=FakePredictiontimeFrame).aggregate_timeseries(
            specs=[FakePredictorSpec, FakePredictorSpec]
        )


def test_error_if_missing_entity_id_column():
    with pytest.raises(flattener.SpecError, match=".*is missing in the .* specification.*"):
        flattener.Flattener(
            predictiontime_frame=PredictionTimeFrame(
                init_df=pl.LazyFrame(
                    {
                        "no_entity_id": [1, 2, 3],
                        "pred_timestamp": ["2013-01-01", "2013-01-01", "2013-01-01"],
                    }
                ),
                entity_id_col_name="no_entity_id",
            )
        ).aggregate_timeseries(specs=[FakePredictorSpec])


def test_error_if_missing_column_in_valueframe():
    with pytest.raises(SpecColumnError, match="Missing columns: *"):
        ValueFrame(
            init_df=pl.LazyFrame({"value": [1], "timestamp": ["2021-01-01"]}),
            value_col_name="value",
        )


def test_predictor_with_interval_lookperiod():
    prediction_times_df_str = """entity_id,pred_timestamp,
                            1,2022-01-01 00:00:00
                            """
    predictor_df_str = """entity_id,timestamp,value,
                        1,2021-12-30 00:00:01, 2
                        1,2021-12-15 00:00:00, 1
                        """
    result = flattener.Flattener(
        predictiontime_frame=PredictionTimeFrame(init_df=str_to_pl_df(prediction_times_df_str))
    ).aggregate_timeseries(
        specs=[
            PredictorSpec(
                value_frame=ValueFrame(
                    init_df=str_to_pl_df(predictor_df_str), value_col_name="value"
                ),
                lookbehind_distances=[(dt.timedelta(days=5), dt.timedelta(days=30))],
                fallback=np.NaN,
                aggregators=[MeanAggregator()],
            )
        ]
    )
    expected = str_to_pl_df(
        """pred_time_uuid,pred_value_within_5_to_30_days_mean_fallback_nan
1-2022-01-01 00:00:00.000000,1"""
    )
    assert_frame_equal(result.collect(), expected)


def test_outcome_with_interval_lookperiod():
    prediction_times_df_str = """entity_id,pred_timestamp,
                            1,2022-01-01 00:00:00
                            """
    outcome_df_str = """entity_id,timestamp,value,
                        1,2022-01-02 00:00:01, 2
                        1,2022-01-15 00:00:00, 1
                        """
    result = flattener.Flattener(
        predictiontime_frame=PredictionTimeFrame(init_df=str_to_pl_df(prediction_times_df_str))
    ).aggregate_timeseries(
        specs=[
            OutcomeSpec(
                value_frame=ValueFrame(
                    init_df=str_to_pl_df(outcome_df_str), value_col_name="value"
                ),
                lookahead_distances=[(dt.timedelta(days=5), dt.timedelta(days=30))],
                fallback=np.NaN,
                aggregators=[MeanAggregator()],
            )
        ]
    )
    expected = str_to_pl_df(
        """pred_time_uuid,outc_value_within_5_to_30_days_mean_fallback_nan
1-2022-01-01 00:00:00.000000,1"""
    )
    assert_frame_equal(result.collect(), expected)
