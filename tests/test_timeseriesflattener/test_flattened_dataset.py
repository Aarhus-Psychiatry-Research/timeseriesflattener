import numpy as np
import pandas as pd
import pytest

from timeseriesflattener.feature_spec_objects import (
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
)
from timeseriesflattener.flattened_dataset import TimeseriesFlattener
from timeseriesflattener.resolve_multiple_functions import mean
from timeseriesflattener.testing.utils_for_testing import (
    synth_outcome,
    synth_prediction_times,
)

# To avoid ruff auto-removing unused imports
used_funcs = [synth_prediction_times, synth_outcome]


def test_add_spec(synth_prediction_times: pd.DataFrame, synth_outcome: pd.DataFrame):
    # Create an instance of the class that contains the `add_spec` method
    dataset = TimeseriesFlattener(
        prediction_times_df=synth_prediction_times,
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
        values_df=synth_outcome, feature_name="static", prefix="pred"
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
