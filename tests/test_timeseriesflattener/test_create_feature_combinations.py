"""Tests for feature_combination creation."""

# pylint: disable=missing-function-docstring

import numpy as np

from timeseriesflattener.feature_spec_objects import PredictorGroupSpec
from timeseriesflattener.resolve_multiple_functions import maximum
from timeseriesflattener.testing.load_synth_data import (  # noqa
    load_synth_predictor_float,
)
from timeseriesflattener.utils import data_loaders  # noqa

# Avoid ruff removing as unused
used_loaders = [
    load_synth_predictor_float,
]


def test_skip_all_if_no_need_to_process():
    assert (
        len(
            PredictorGroupSpec(
                values_loader=["synth_predictor_float"],
                input_col_name_override="value",
                lookbehind_days=[1],
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
        input_col_name_override="value",
        lookbehind_days=[1, 2],
        resolve_multiple_fn=["max", "min"],
        fallback=[0],
        allowed_nan_value_prop=[0],
    ).create_combinations()

    assert len(created_combinations) == 4


def test_resolve_multiple_fn_to_str():
    pred_spec_batch = PredictorGroupSpec(
        values_loader=["synth_predictor_float"],
        lookbehind_days=[365, 730],
        fallback=[np.nan],
        resolve_multiple_fn=[maximum],
    ).create_combinations()

    assert "maximum" in pred_spec_batch[0].get_col_str()
