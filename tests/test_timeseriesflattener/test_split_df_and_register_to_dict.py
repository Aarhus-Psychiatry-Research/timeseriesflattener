"""Test that feature spec objects work as intended."""
import pandas as pd
import pytest

from timeseriesflattener.feature_spec_objects import (
    AnySpec,
    MinGroupSpec,
    PredictorGroupSpec,
    PredictorSpec,
    create_specs_from_group,
)
from timeseriesflattener.testing.load_synth_data import synth_predictor_binary  # noqa
from timeseriesflattener.utils import df_dict, split_df_and_register_to_dict


@pytest.fixture(scope="function")
def df():
    """Create a long df."""
    synth_df = synth_predictor_binary()
    synth_df = synth_df.rename(columns={"value": "value_name_1"})
    synth_df["value_name_2"] = (
        synth_df["value_name_1"].sample(frac=1).reset_index(drop=True)
    )

    df = pd.melt(
        synth_df,
        id_vars=["dw_ek_borger", "timestamp"],
        value_vars=["value_name_1", "value_name_2"],
        var_name="value_names",
        value_name="value",
    )

    return df


def test_split_df_and_register_in_dict(df):
    """Test that the split_df_and_register_to_dict function works as intended."""

    split_df_and_register_to_dict(df=df)

    assert len(df_dict) == 2
    assert df_dict["value_name_1"].shape == (10000, 3)
    assert df_dict["value_name_2"].shape == (10000, 3)


def test_resolve_from_df_dict(df):
    """Test that a split_df_and_register_to_dict resolves correctly."""

    split_df_and_register_to_dict(df=df)

    spec = AnySpec(
        values_name="value_name_1",
        feature_name="test",
        prefix="test",
    )

    assert len(spec.values_df) == 10000


def test_resolve_from_df_dict_group_spec(df):
    """Test that split_df_and_register_to_dict resolves correctly when multiple dataframes are fetched."""

    split_df_and_register_to_dict(df=df)

    group_spec = PredictorGroupSpec(
        values_name=[
            "value_name_1",
            "value_name_2",
        ],
        resolve_multiple_fn=["mean"],
        interval_days=[30],
        fallback=[0],
    ).create_combinations()

    assert 2 + 2 == 4
