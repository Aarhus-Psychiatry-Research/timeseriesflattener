"""Test that feature spec objects work as intended."""
import pandas as pd
import pytest

from timeseriesflattener.feature_spec_objects import AnySpec
from timeseriesflattener.testing.load_synth_data import synth_predictor_binary  # noqa
from timeseriesflattener.utils import hydrate_dict_from_long_df, long_df_registry


@pytest.fixture(scope="function")
def long_df():
    """Create a long df."""
    df = synth_predictor_binary()
    df = df.rename(columns={"value": "value_key_1"})
    df["value_key_2"] = df["value_key_1"].sample(frac=1).reset_index(drop=True)

    long_df = pd.melt(
        df,
        id_vars=["dw_ek_borger", "timestamp"],
        value_vars=["value_key_1", "value_key_2"],
        var_name="value_keys",
        value_name="value",
    )

    return long_df


def test_hydrate_dict_from_long_df(long_df):
    """Test that the hydrate_registry_from_long_df function works as intended."""

    hydrate_dict_from_long_df(df=long_df)

    assert len(long_df_registry) == 2
    assert long_df_registry["value_key_1"].shape == (10000, 3)
    assert long_df_registry["value_key_2"].shape == (10000, 3)


def test_resolve_from_long_df_dict(long_df):
    """Test that a hydrated long df dict resolves correctly."""

    hydrate_dict_from_long_df(df=long_df)

    spec = AnySpec(
        values_df_dict="value_key_1",
        feature_name="test",
        prefix="test",
    )

    assert len(spec.values_df) == 10000
