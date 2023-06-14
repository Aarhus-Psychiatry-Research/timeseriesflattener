"""Test that feature spec objects work as intended."""
import pandas as pd

from timeseriesflattener.feature_specs.single_specs import AnySpec
from timeseriesflattener.misc_utils import split_df_and_register_to_dict, split_dfs


def test_split_df_and_register_in_dict(long_df_with_multiple_values: pd.DataFrame):
    """Test that the split_df_and_register_to_dict function works as intended."""

    split_df_and_register_to_dict(df=long_df_with_multiple_values)

    assert len(split_dfs) == 2
    assert split_dfs["value_name_1"].shape == (10000, 3)
    assert split_dfs["value_name_2"].shape == (10000, 3)


def test_resolve_from_df_dict(long_df_with_multiple_values: pd.DataFrame):
    """Test that a split_df_and_register_to_dict resolves from the  correctly."""

    split_df_and_register_to_dict(df=long_df_with_multiple_values)

    spec = AnySpec(
        values_name="value_name_1",
        feature_base_name="test",
        prefix="test",
    )

    assert len(spec.base_values_df) == 10000  # type: ignore
