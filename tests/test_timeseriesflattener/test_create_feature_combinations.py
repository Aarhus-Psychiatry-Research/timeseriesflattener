"""Tests for feature_combination creation."""

# pylint: disable=missing-function-docstring
import numpy as np

from psycop_feature_generation.loaders.synth.raw.load_synth_data import (
    synth_predictor_float,
)
from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    PredictorGroupSpec,
    PredictorSpec,
)


def test_skip_all_if_no_need_to_process():
    assert (
        len(
            PredictorGroupSpec(
                values_df=synth_predictor_float(),
                input_col_name_override="val",
                interval_days=[1],
                resolve_multiple_fn=["max"],
                fallback=[0],
                allowed_nan_value_prop=[0.5],
                feature_name="value",
            ).create_combinations(),
        )
        == 1
    )


def test_skip_one_if_no_need_to_process():
    created_combinations = PredictorGroupSpec(
        values_df=synth_predictor_float(),
        input_col_name_override="val",
        interval_days=[1, 2],
        resolve_multiple_fn=["max", "min"],
        fallback=[0],
        allowed_nan_value_prop=[0],
        feature_name="value",
    ).create_combinations()

    expected_combinations = [
        PredictorSpec(
            values_df=synth_predictor_float(),
            interval_days=1,
            resolve_multiple_fn="max",
            fallback=0,
            allowed_nan_value_prop=0,
            input_col_name_override="val",
            feature_name="value",
        ),
        PredictorSpec(
            values_df=synth_predictor_float(),
            interval_days=2,
            resolve_multiple_fn="max",
            fallback=0,
            allowed_nan_value_prop=0,
            input_col_name_override="val",
            feature_name="value",
        ),
        PredictorSpec(
            values_df=synth_predictor_float(),
            interval_days=1,
            resolve_multiple_fn="min",
            fallback=0,
            allowed_nan_value_prop=0,
            input_col_name_override="val",
            feature_name="value",
        ),
        PredictorSpec(
            values_df=synth_predictor_float(),
            interval_days=2,
            resolve_multiple_fn="min",
            fallback=0,
            allowed_nan_value_prop=0,
            input_col_name_override="val",
            feature_name="value",
        ),
    ]

    for combination in created_combinations:
        assert combination in expected_combinations
