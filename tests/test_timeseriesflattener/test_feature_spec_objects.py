"""Test that feature spec objects work as intended."""
import pandas as pd
import pytest

from psycop_feature_generation.loaders.synth.raw.load_synth_data import (  # pylint: disable=unused-import
    synth_predictor_float,
)
from psycop_feature_generation.timeseriesflattener.feature_spec_objects import AnySpec


def test_anyspec_init():
    """Test that AnySpec initialises correctly."""
    values_loader_name = "synth_predictor_float"

    spec = AnySpec(
        values_loader=values_loader_name,
        prefix="test",
    )

    assert isinstance(spec.values_df, pd.DataFrame)
    assert spec.feature_name == values_loader_name


def test_loader_kwargs():
    spec = AnySpec(
        values_loader="synth_predictor_float",
        prefix="test",
        loader_kwargs={"n_rows": 10},
    )

    assert len(spec.values_df) == 10


def test_anyspec_output_col_name_override():
    spec = AnySpec(
        values_loader="synth_predictor_float",
        prefix="test",
        output_col_name_override="test",
    )

    assert "test" in spec.values_df.columns


def test_anyspec_incorrect_values_loader_str():
    with pytest.raises(ValueError, match=r".*in registry.*"):
        AnySpec(values_loader="I don't exist", prefix="test")


if __name__ == "__main__":
    test_anyspec_output_col_name_override()
