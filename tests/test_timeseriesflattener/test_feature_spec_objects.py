import pandas as pd
import pytest

from psycop_feature_generation.loaders.synth.raw.load_synth_data import (
    synth_predictor_float,
)
from psycop_feature_generation.timeseriesflattener.feature_spec_objects import AnySpec


def test_anyspec_init():
    """Test that AnySpec initialises correctly."""
    values_loader_name = "synth_predictor_float"

    df = synth_predictor_float()

    spec = AnySpec(
        values_loader=values_loader_name,
        prefix="test",
    )

    assert type(spec.df) == pd.DataFrame
    assert spec.feature_name == values_loader_name


def test_anyspec_incorrect_values_loader_str():
    with pytest.raises(ValueError, match="to a callable"):
        AnySpec(values_loader="I don't exist", prefix="test")
