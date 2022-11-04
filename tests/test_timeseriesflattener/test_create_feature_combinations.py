"""Tests for feature_combination creation."""

# pylint: disable=missing-function-docstring

from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    PredictorGroupSpec,
    PredictorSpec,
)


def test_skip_all_if_no_need_to_process():
    assert (
        len(
            PredictorGroupSpec(
                values_df=["prediction_times_df"],
                source_values_col_name=["val"],
                interval_days=[1],
                resolve_multiple=["max"],
                fallback=[0],
                allowed_nan_value_prop=[0.5],
            ).create_combinations()
        )
        == 1
    )


def test_skip_one_if_no_need_to_process():
    created_combinations = PredictorGroupSpec(
        values_df=["prediction_times_df"],
        source_values_col_name=["val"],
        interval_days=[1, 2],
        resolve_multiple=["max", "min"],
        fallback=[0],
        allowed_nan_value_prop=[0],
    ).create_combinations()

    expected_combinations = [
        PredictorSpec(
            values_df="prediction_times_df",
            interval_days=1,
            resolve_multiple="max",
            fallback=0,
            allowed_nan_value_prop=0,
            source_values_col_name="val",
        ),
        PredictorSpec(
            values_df="prediction_times_df",
            interval_days=2,
            resolve_multiple="max",
            fallback=0,
            allowed_nan_value_prop=0,
            source_values_col_name="val",
        ),
        PredictorSpec(
            values_df="prediction_times_df",
            interval_days=1,
            resolve_multiple="min",
            fallback=0,
            allowed_nan_value_prop=0,
            source_values_col_name="val",
        ),
        PredictorSpec(
            values_df="prediction_times_df",
            interval_days=2,
            resolve_multiple="min",
            fallback=0,
            allowed_nan_value_prop=0,
            source_values_col_name="val",
        ),
    ]

    for combination in created_combinations:
        assert combination in expected_combinations
