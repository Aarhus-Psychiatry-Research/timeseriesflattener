"""Test that feature spec objects work as intended."""
import pandas as pd
import pytest

from loaders.synth.raw.load_synth_data import (  # pylint: disable=unused-import
    synth_predictor_float,
)
from timeseriesflattener.feature_spec_objects import AnySpec


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
    """Test that loader kwargs are passed correctly."""
    spec = AnySpec(
        values_loader="synth_predictor_float",
        prefix="test",
        loader_kwargs={"n_rows": 10},
    )

    assert len(spec.values_df) == 10


def test_anyspec_incorrect_values_loader_str():
    """Test that AnySpec raises an error if the values loader is not a key in
    the loader registry."""
    with pytest.raises(ValueError, match=r".*in registry.*"):
        AnySpec(values_loader="I don't exist", prefix="test")
