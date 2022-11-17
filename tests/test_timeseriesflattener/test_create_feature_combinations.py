"""Tests for feature_combination creation."""

# pylint: disable=missing-function-docstring

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
                values_loader=["synth_predictor_float"],
                input_col_name_override="val",
                interval_days=[1],
                resolve_multiple_fn=["max"],
                fallback=[0],
                allowed_nan_value_prop=[0.5],
            ).create_combinations(),
        )
        == 1
    )


def test_skip_one_if_no_need_to_process():
    created_combinations = PredictorGroupSpec(
        values_loader=["synth_predictor_float"],
        input_col_name_override="val",
        interval_days=[1, 2],
        resolve_multiple_fn=["max", "min"],
        fallback=[0],
        allowed_nan_value_prop=[0],
    ).create_combinations()

    assert len(created_combinations) == 4
