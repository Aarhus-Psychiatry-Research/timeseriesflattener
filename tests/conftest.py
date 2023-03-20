# pylint: disable-all

from typing import Optional

import pandas as pd
import pytest
from pandas import DataFrame
from timeseriesflattener.testing.load_synth_data import load_raw_test_csv
from timeseriesflattener.testing.utils_for_testing import (
    load_long_df_with_multiple_values,
)
from timeseriesflattener.utils import data_loaders


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


@data_loaders.register("synth_predictor_float")
def load_synth_predictor_float(
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_raw_float_1.csv", n_rows=n_rows)


@data_loaders.register("synth_sex")
def load_synth_sex(
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load synth sex data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_sex.csv", n_rows=n_rows)


@data_loaders.register("synth_predictor_binary")
def synth_predictor_binary(
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_raw_binary_1.csv", n_rows=n_rows)


@data_loaders.register("synth_outcome")
def load_synth_outcome(
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    # Get first row for each id
    df = load_raw_test_csv("synth_raw_binary_2.csv", n_rows=n_rows)
    df = df.groupby("entity_id").last().reset_index()

    # Drop all rows with a value equal to 1
    df = df[df["value"] == 1]
    return df


@data_loaders.register("synth_prediction_times")
def load_synth_prediction_times(
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_prediction_times.csv", n_rows=n_rows)


@data_loaders.register("synth_text")
def load_synth_text(
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load synth text data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_text_data.csv", n_rows=n_rows)
