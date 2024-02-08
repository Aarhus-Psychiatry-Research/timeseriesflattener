"""Loaders for synth data."""

import logging
from typing import Optional

import pandas as pd
from timeseriesflattener.misc_utils import PROJECT_ROOT

log = logging.getLogger(__name__)

TEST_DATA_PATH = PROJECT_ROOT / "src" / "timeseriesflattener" / "testing" / "test_data"


def load_raw_test_csv(filename: str, n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load raw csv.

    Args:
        filename (str): Name of the file to load.
        n_rows (int, optional): Number of rows to load. Defaults to None.
    """
    df = pd.read_csv(TEST_DATA_PATH / "raw" / filename, nrows=n_rows)

    # Convert timestamp col to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def load_synth_predictor_float(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_raw_float_1.csv", n_rows=n_rows)


def load_synth_sex(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load synth sex data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_sex.csv", n_rows=n_rows)


def synth_predictor_binary(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_raw_binary_1.csv", n_rows=n_rows)


def load_synth_outcome(n_rows: Optional[int] = None) -> pd.DataFrame:
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


def load_synth_prediction_times(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_prediction_times.csv", n_rows=n_rows)


def load_synth_text(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load synth text data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    df = load_raw_test_csv("synth_text_data.csv", n_rows=n_rows)
    df["value"] = df["text"]
    df = df.drop(columns=["text"])
    return df
