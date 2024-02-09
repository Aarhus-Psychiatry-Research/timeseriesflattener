"""Test that feature spec objects work as intended."""


from typing import List

import numpy as np
import pytest

from timeseriesflattener.aggregation_fns import maximum
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    OutcomeGroupSpec,
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import PredictorSpec
from timeseriesflattener.testing.utils_for_testing import str_to_df


def test_skip_all_if_no_need_to_process(empty_named_df: NamedDataframe):
    """Test that no combinations are created if no need to process."""
    assert (
        len(
            PredictorGroupSpec(
                named_dataframes=[empty_named_df],
                lookbehind_days=[1],
                aggregation_fns=[maximum],
                fallback=[0],
            ).create_combinations()
        )
        == 1
    )


def test_skip_one_if_no_need_to_process(empty_named_df: NamedDataframe):
    """Test that one combination is skipped if no need to process."""
    created_combinations = PredictorGroupSpec(
        named_dataframes=[empty_named_df],
        lookbehind_days=[1, 2],
        aggregation_fns=[maximum],
        fallback=[0],
    ).create_combinations()

    assert len(created_combinations) == 2


def test_aggregation_fn_to_str(empty_named_df: NamedDataframe):
    """Test that aggregation_fn is converted to str correctly."""
    pred_spec_batch = PredictorGroupSpec(
        named_dataframes=[empty_named_df],
        lookbehind_days=[365, 730],
        fallback=[np.nan],
        aggregation_fns=[maximum],
    ).create_combinations()

    assert "maximum" in pred_spec_batch[0].get_output_col_name()


def test_lookbehind_days_handles_floats(empty_named_df: NamedDataframe):
    """Test that lookbheind days does not coerce floats into ints."""
    pred_spec_batch = PredictorGroupSpec(
        named_dataframes=[empty_named_df],
        lookbehind_days=[2, 0.5],
        fallback=[np.nan],
        aggregation_fns=[maximum],
    ).create_combinations()

    assert pred_spec_batch[1].lookbehind_days == 0.5


def get_lines_with_diff(text1: str, text2: str) -> List[str]:
    """Find all lines in text1 which are different from text2."""
    # Remove whitespace and periods
    text_1 = text1.replace(" ", "").replace(".", "")
    text_2 = text2.replace(" ", "").replace(".", "")

    lines1 = text_1.splitlines()
    lines2 = text_2.splitlines()
    return [line for line in lines1 if line not in lines2]


def test_create_combinations_outcome_specs(empty_named_df: NamedDataframe):
    """Test that create_combinations() creates the correct outcome_specs."""
    outc_spec_batch = OutcomeGroupSpec(
        named_dataframes=[empty_named_df],
        lookahead_days=[1, 2, (1, 2)],
        aggregation_fns=[maximum],
        fallback=[0],
        incident=[True],
    ).create_combinations()
    assert len(outc_spec_batch) == 3


def test_invalid_lookbehind():
    prediction_times_df_str = """entity_id,timestamp,
                                1,2021-12-30 00:00:00
                                """
    spec = PredictorSpec(
        timeseries_df=str_to_df(prediction_times_df_str),
        lookbehind_days=(1, 0),
        aggregation_fn=maximum,
        fallback=2,
        feature_base_name="value",
    )
    with pytest.raises(ValueError, match=r".*Invalid.*"):
        assert spec.lookbehind_period
