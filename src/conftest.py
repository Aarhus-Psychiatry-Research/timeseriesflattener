import pytest
from pandas import DataFrame
from timeseriesflattener.feature_specs.group_specs import NamedDataframe
from timeseriesflattener.testing.load_synth_data import (
    load_synth_outcome,
    load_synth_prediction_times,
    load_synth_text,
)
from timeseriesflattener.testing.utils_for_testing import load_long_df_with_multiple_values


@pytest.fixture()
def synth_prediction_times() -> DataFrame:
    """Load the prediction times."""
    return load_synth_prediction_times()


@pytest.fixture()
def synth_predictor() -> DataFrame:
    """Load the synth outcome times."""
    return load_synth_outcome(n_rows=1_000)


@pytest.fixture()
def synth_outcome() -> DataFrame:
    """Load the synth outcome times."""
    return load_synth_outcome()


@pytest.fixture()
def long_df_with_multiple_values() -> DataFrame:
    """Load the long df."""
    return load_long_df_with_multiple_values()


@pytest.fixture()
def synth_text_data() -> DataFrame:
    """Load the synth text data."""
    return load_synth_text()


@pytest.fixture()
def empty_df() -> DataFrame:
    """Create an empty dataframe."""
    return DataFrame()


@pytest.fixture()
def empty_named_df() -> NamedDataframe:
    return NamedDataframe(df=DataFrame(), name="empty")
