import pytest
from pandas import DataFrame
from timeseriesflattener.testing.load_synth_data import (
    load_synth_outcome,
    load_synth_prediction_times,
    load_synth_text,
)
from timeseriesflattener.testing.utils_for_testing import (
    load_long_df_with_multiple_values,
)


def pytest_addoption(parser):  # noqa
    parser.addoption(
        "--skiphuggingface",
        action="store_true",
        default=False,
        help="run tests that use huggingface models",
    )


def pytest_configure(config):  # noqa
    config.addinivalue_line(
        "markers",
        "huggingface: mark test as using huggingface models",
    )


def pytest_collection_modifyitems(config, items):  # noqa
    if config.getoption("--skiphuggingface"):
        # --skiphuggingface given in cli: skip huggingface tests
        skip_hf = pytest.mark.skip(reason="remove --skiphuggingface option to run")
        for item in items:
            if "huggingface" in item.keywords:
                item.add_marker(skip_hf)
        return


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
