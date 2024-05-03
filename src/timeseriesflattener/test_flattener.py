from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import polars.testing as polars_testing
import pytest
from timeseriesflattener.testing.utils_for_testing import str_to_pl_df

from timeseriesflattener.aggregators import EarliestAggregator, MeanAggregator

from . import flattener
from ._frame_validator import SpecColumnError
from .feature_specs.meta import ValueFrame
from .feature_specs.outcome import OutcomeSpec
from .feature_specs.prediction_times import PredictionTimeFrame
from .feature_specs.predictor import PredictorSpec
from .feature_specs.static import StaticSpec, StaticFrame

if TYPE_CHECKING:
    from collections.abc import Sequence

FakePredictiontimeFrame = PredictionTimeFrame(
    init_df=pl.LazyFrame({"entity_id": [1], "pred_timestamp": ["2021-01-03"]})
)
FakeValueFrame = ValueFrame(
    init_df=pl.LazyFrame({"entity_id": [1], "value": [1], "timestamp": ["2021-01-01"]})
)
FakePredictorSpec = PredictorSpec(
    value_frame=ValueFrame(
        init_df=pl.LazyFrame(
            {"entity_id": [1], "FakeValueColName": [1], "timestamp": ["2021-01-01"]}
        )
    ),
    lookbehind_distances=[dt.timedelta(days=1)],
    aggregators=[MeanAggregator()],
    fallback=np.nan,
)


def assert_frame_equal(
    result: pl.DataFrame, expected: pl.DataFrame, ignore_colums: Sequence[str] = ()
):
    polars_testing.assert_frame_equal(
        result.drop(ignore_colums),
        expected.drop(ignore_colums),
        check_dtype=False,
        check_column_order=False,
    )


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
                value_frame=ValueFrame(init_df=value_frame.lazy()),
                lookbehind_distances=[dt.timedelta(days=1)],
                aggregators=[MeanAggregator()],
                fallback=np.nan,
            )
        ]
    )

    expected = str_to_pl_df(
        """entity_id,pred_timestamp,prediction_time_uuid,pred_value_within_0_to_1_days_mean_fallback_nan
1,2021-01-03 00:00:00.000000,1-2021-01-03 00:00:00.000000,3.0"""
    )

    assert_frame_equal(result.collect(), expected)


def test_keep_prediction_times_without_predictors():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp,prediction_time_uuid
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
                value_frame=ValueFrame(init_df=value_frame.lazy()),
                lookbehind_distances=[dt.timedelta(days=1)],
                aggregators=[MeanAggregator(), EarliestAggregator(timestamp_col_name="timestamp")],
                fallback=123,
            )
        ]
    )

    expected = pl.DataFrame(
        {
            "prediction_time_uuid": ["1-2021-01-03 00:00:00.000000"],
            "pred_value_within_0_to_1_days_mean_fallback_123": [123],
            "pred_value_within_0_to_1_days_earliest_fallback_123": [123],
        }
    )

    assert_frame_equal(result.collect(), expected, ignore_colums=["entity_id", "pred_timestamp"])


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
                value_frame=ValueFrame(init_df=value_frame.rename({"value": "value_1"}).lazy()),
                lookbehind_distances=[dt.timedelta(days=1)],
                aggregators=[MeanAggregator()],
                fallback=np.nan,
            ),
            PredictorSpec(
                value_frame=ValueFrame(init_df=value_frame.rename({"value": "value_2"}).lazy()),
                lookbehind_distances=[dt.timedelta(days=1)],
                aggregators=[MeanAggregator()],
                fallback=np.nan,
            ),
        ]
    )

    expected = str_to_pl_df(
        """prediction_time_uuid,pred_value_1_within_0_to_1_days_mean_fallback_nan,pred_value_2_within_0_to_1_days_mean_fallback_nan
1-2021-01-03 00:00:00.000000,3.0,3.0"""
    )

    assert_frame_equal(result.collect(), expected, ignore_colums=["entity_id", "pred_timestamp"])


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
        ValueFrame(init_df=pl.LazyFrame({"value": [1], "timestamp": ["2021-01-01"]}))


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
                value_frame=ValueFrame(init_df=str_to_pl_df(predictor_df_str)),
                lookbehind_distances=[(dt.timedelta(days=5), dt.timedelta(days=30))],
                fallback=np.NaN,
                aggregators=[MeanAggregator()],
            )
        ]
    )
    expected = str_to_pl_df(
        """prediction_time_uuid,pred_value_within_5_to_30_days_mean_fallback_nan
1-2022-01-01 00:00:00.000000,1"""
    )
    assert_frame_equal(result.collect(), expected, ignore_colums=["entity_id", "pred_timestamp"])


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
                value_frame=ValueFrame(init_df=str_to_pl_df(outcome_df_str)),
                lookahead_distances=[(dt.timedelta(days=5), dt.timedelta(days=30))],
                fallback=np.NaN,
                aggregators=[MeanAggregator()],
            )
        ]
    )
    expected = str_to_pl_df(
        """prediction_time_uuid,outc_value_within_5_to_30_days_mean_fallback_nan
1-2022-01-01 00:00:00.000000,1"""
    )
    assert_frame_equal(result.collect(), expected, ignore_colums=["entity_id", "pred_timestamp"])


def test_add_static_spec():
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
                value_frame=ValueFrame(init_df=str_to_pl_df(outcome_df_str)),
                lookahead_distances=[(dt.timedelta(days=5), dt.timedelta(days=30))],
                fallback=np.NaN,
                aggregators=[MeanAggregator()],
            )
        ]
    )
    expected = str_to_pl_df(
        """prediction_time_uuid,outc_value_within_5_to_30_days_mean_fallback_nan
1-2022-01-01 00:00:00.000000,1"""
    )
    assert_frame_equal(result.collect(), expected, ignore_colums=["entity_id", "pred_timestamp"])


def test_add_features_with_non_default_entity_id_col_name():
    prediction_times_df_str = """dw_ek_borger,pred_timestamp,
                            1,2022-01-01 00:00:00
                            """
    outcome_df_str = """dw_ek_borger,timestamp,value,
                        1,2022-01-02 00:00:01, 2
                        1,2022-01-15 00:00:00, 1
                        """
    result = flattener.Flattener(
        predictiontime_frame=PredictionTimeFrame(
            init_df=str_to_pl_df(prediction_times_df_str), entity_id_col_name="dw_ek_borger"
        )
    ).aggregate_timeseries(
        specs=[
            OutcomeSpec(
                value_frame=ValueFrame(
                    init_df=str_to_pl_df(outcome_df_str), entity_id_col_name="dw_ek_borger"
                ),
                lookahead_distances=[(dt.timedelta(days=5), dt.timedelta(days=30))],
                fallback=np.NaN,
                aggregators=[MeanAggregator()],
            )
        ]
    )
    expected = str_to_pl_df(
        """prediction_time_uuid,outc_value_within_5_to_30_days_mean_fallback_nan
1-2022-01-01 00:00:00.000000,1"""
    )
    assert_frame_equal(result.collect(), expected, ignore_colums=["dw_ek_borger", "pred_timestamp"])


@pytest.mark.parametrize("step_size", [None, dt.timedelta(days=30)])
def test_multiple_features_with_unordered_prediction_times(step_size):
    prediction_times_df_str = """entity_id,pred_timestamp,
                            2,2022-01-02 00:00:00
                            1,2022-01-01 00:00:00
                            1,2020-01-01 00:00:00
                            """
    pred_df_str = """entity_id,timestamp,value,
                        1,2021-12-31 00:00:01, 1
                        """
    static_df_str = """entity_id,static
                        1,1
                        2,2
                        """
    result = flattener.Flattener(
        predictiontime_frame=PredictionTimeFrame(init_df=str_to_pl_df(prediction_times_df_str))
    ).aggregate_timeseries(
        specs=[
            PredictorSpec(
                value_frame=ValueFrame(init_df=str_to_pl_df(pred_df_str)),
                lookbehind_distances=[dt.timedelta(days=1)],
                fallback=0,
                aggregators=[MeanAggregator()],
            ),
            StaticSpec(
                value_frame=StaticFrame(init_df=str_to_pl_df(static_df_str)),
                column_prefix="pred",
                fallback=0,
            ),
        ],
        step_size=step_size,
    )
    expected = str_to_pl_df(
        """prediction_time_uuid,pred_value_within_0_to_1_days_mean_fallback_0,pred_static_fallback_0
2-2022-01-02 00:00:00.000000,0.0,2
1-2022-01-01 00:00:00.000000,1.0,1
1-2020-01-01 00:00:00.000000,0.0,1
"""
    ).sort("prediction_time_uuid")
    assert_frame_equal(
        result.df.collect().sort("prediction_time_uuid"),
        expected,
        ignore_colums=["entity_id", "pred_timestamp"],
    )
