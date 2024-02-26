"""Fixtures for v1 testing. Delete when v1 is deprecated."""
from __future__ import annotations

import pytest
from pandas import DataFrame
from timeseriesflattener.testing.load_synth_data import (
    load_synth_outcome,
    load_synth_prediction_times,
    load_synth_text,
)
from timeseriesflattener.testing.utils_for_testing import load_long_df_with_multiple_values
from timeseriesflattener.v1.feature_specs.group_specs import NamedDataframe


@pytest.fixture()
def synth_prediction_times() -> DataFrame:
    """Load the prediction times."""
    return load_synth_prediction_times().to_pandas()


@pytest.fixture()
def synth_predictor() -> DataFrame:
    """Load the synth outcome times."""
    return load_synth_outcome().to_pandas()


@pytest.fixture()
def synth_outcome() -> DataFrame:
    """Load the synth outcome times."""
    return load_synth_outcome().to_pandas()


@pytest.fixture()
def long_df_with_multiple_values() -> DataFrame:
    """Load the long df."""
    return load_long_df_with_multiple_values()


@pytest.fixture()
def synth_text_data() -> DataFrame:
    """Load the synth text data."""
    return load_synth_text().to_pandas()


@pytest.fixture()
def empty_df() -> DataFrame:
    """Create an empty dataframe."""
    return DataFrame()


@pytest.fixture()
def empty_named_df() -> NamedDataframe:
    return NamedDataframe(df=DataFrame(), name="empty")
