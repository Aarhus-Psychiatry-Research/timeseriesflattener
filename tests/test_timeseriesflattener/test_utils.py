"""Test that feature spec objects work as intended."""
import pandas as pd
import pytest

from timeseriesflattener.feature_spec_objects import AnySpec
from timeseriesflattener.testing.load_synth_data import synth_predictor_binary  # noqa
from timeseriesflattener.testing.utils_for_testing import long_df
from timeseriesflattener.utils import split_df_and_register_to_dict, split_df_dict


def test_split_df_and_register_in_dict(long_df: pd.DataFrame):
    """Test that the split_df_and_register_to_dict function works as intended."""

    split_df_and_register_to_dict(df=long_df)

    assert len(split_df_dict) == 2
    assert split_df_dict["value_name_1"].shape == (10000, 3)
    assert split_df_dict["value_name_2"].shape == (10000, 3)


def test_resolve_from_df_dict(long_df: pd.DataFrame):
    """Test that a split_df_and_register_to_dict resolves from the  correctly."""

    split_df_and_register_to_dict(df=long_df)

    spec = AnySpec(
        values_name="value_name_1",
        feature_name="test",
        prefix="test",
    )

    assert len(spec.values_df) == 10000
