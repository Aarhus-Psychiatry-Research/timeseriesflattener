"""Larger tests for the `flattened_dataset.py` module."""
import numpy as np
import pandas as pd
import pytest

from timeseriesflattener.feature_spec_objects import (
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
)
from timeseriesflattener.flattened_dataset import TimeseriesFlattener
from timeseriesflattener.resolve_multiple_functions import latest, mean
from timeseriesflattener.testing.utils_for_testing import (
    synth_outcome,
    synth_prediction_times,
)

# To avoid ruff auto-removing unused imports
used_funcs = [synth_prediction_times, synth_outcome]

# pylint: disable=missing-function-docstring


def test_add_spec(synth_prediction_times: pd.DataFrame, synth_outcome: pd.DataFrame):
    # Create an instance of the class that contains the `add_spec` method
    dataset = TimeseriesFlattener(
        prediction_times_df=synth_prediction_times,
        drop_pred_times_with_insufficient_look_distance=False,
    )

    # Create sample specs
    outcome_spec = OutcomeSpec(
        values_df=synth_outcome,
        feature_name="outcome",
        lookahead_days=1,
        resolve_multiple_fn=mean,
        fallback=0,
        incident=False,
    )
    predictor_spec = PredictorSpec(
        values_df=synth_outcome,
        feature_name="predictor",
        lookbehind_days=1,
        resolve_multiple_fn=mean,
        fallback=np.nan,
    )
    static_spec = StaticSpec(
        values_df=synth_outcome,
        feature_name="static",
        prefix="pred",
    )

    # Test adding a single spec
    dataset.add_spec(outcome_spec)
    assert dataset.unprocessed_specs.outcome_specs == [outcome_spec]

    # Test adding multiple specs
    dataset.add_spec([predictor_spec, static_spec])
    assert dataset.unprocessed_specs.predictor_specs == [predictor_spec]
    assert dataset.unprocessed_specs.static_specs == [static_spec]

    # Test adding an invalid spec type
    with pytest.raises(ValueError):
        dataset.add_spec("invalid spec")


def test_compute_specs(
    synth_prediction_times: pd.DataFrame,
    synth_outcome: pd.DataFrame,
):
    # Create an instance of the class that contains the `add_spec` method
    dataset = TimeseriesFlattener(
        prediction_times_df=synth_prediction_times,
        drop_pred_times_with_insufficient_look_distance=False,
    )

    # Create sample specs
    outcome_spec = OutcomeSpec(
        values_df=synth_outcome,
        feature_name="outcome",
        lookahead_days=1,
        resolve_multiple_fn=mean,
        fallback=0,
        incident=False,
    )
    predictor_spec = PredictorSpec(
        values_df=synth_outcome,
        feature_name="predictor",
        lookbehind_days=1,
        resolve_multiple_fn=mean,
        fallback=np.nan,
    )
    static_spec = StaticSpec(
        values_df=synth_outcome[["value", "dw_ek_borger"]],
        feature_name="static",
        prefix="pred",
    )

    # Test adding a single spec
    dataset.add_spec([outcome_spec, predictor_spec, static_spec])

    df = dataset.get_df()

    assert isinstance(df, pd.DataFrame)


def test_drop_pred_time_if_insufficient_look_distance():
    # Create a sample DataFrame with some test data
    pred_time_df = pd.DataFrame(
        {
            "dw_ek_borger": [1, 1, 1, 1],
            "timestamp": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        },
    )

    ts_flattener = TimeseriesFlattener(
        prediction_times_df=pred_time_df,
        drop_pred_times_with_insufficient_look_distance=True,
    )

    pred_val_df = pd.DataFrame(
        {
            "dw_ek_borger": [1],
            "timestamp": ["2022-01-01"],
            "value": [1],
        },
    )

    # Create a sample set of specs
    predictor_spec = PredictorSpec(
        values_df=pred_val_df,
        lookbehind_days=1,
        resolve_multiple_fn=latest,
        fallback=np.nan,
        feature_name="test_feature",
    )

    out_val_df = pd.DataFrame(
        {
            "dw_ek_borger": [1],
            "timestamp": ["2022-01-05"],
            "value": [4],
        },
    )

    outcome_spec = OutcomeSpec(
        values_df=out_val_df,
        lookahead_days=2,
        resolve_multiple_fn=latest,
        fallback=np.nan,
        feature_name="test_feature",
        incident=False,
    )

    ts_flattener.add_spec(spec=[predictor_spec, outcome_spec])

    out_df = ts_flattener.get_df()

    # Assert that the correct rows were dropped from the DataFrame
    expected_df = pd.DataFrame({"timestamp": ["2022-01-02", "2022-01-03"]})
    # Convert to datetime to avoid a warning
    expected_df = expected_df.astype({"timestamp": "datetime64[ns]"})
    pd.testing.assert_series_equal(out_df["timestamp"], expected_df["timestamp"])
