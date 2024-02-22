"""Loaders for synth data."""
from __future__ import annotations

import logging

import polars as pl
from timeseriesflattener.misc_utils import PROJECT_ROOT

log = logging.getLogger(__name__)

TEST_DATA_PATH = PROJECT_ROOT / "src" / "timeseriesflattener" / "testing" / "test_data"


def load_raw_test_csv(filename: str) -> pl.DataFrame:
    """Load raw csv.

    Args:
        filename (str): Name of the file to load.
        n_rows (int, optional): Number of rows to load. Defaults to None.
    """
    df = pl.read_csv(TEST_DATA_PATH / "raw" / filename)

    if "timestamp" in df.columns:
        df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime))

    return df


def load_synth_predictor_float() -> pl.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pl.DataFrame
    """
    return load_raw_test_csv("synth_raw_float_1.csv")


def load_synth_sex() -> pl.DataFrame:
    """Load synth sex data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pl.DataFrame
    """
    return load_raw_test_csv("synth_sex.csv")


def synth_predictor_binary() -> pl.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pl.DataFrame
    """
    return load_raw_test_csv("synth_raw_binary_1.csv")


def load_synth_outcome() -> pl.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pl.DataFrame
    """
    # Get first row for each id
    df = load_raw_test_csv("synth_raw_binary_2.csv")
    df = df.groupby("entity_id").last()

    # Drop all rows with a value equal to 1
    df = df.filter(pl.col("value") == 1)
    return df


def load_synth_prediction_times() -> pl.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pl.DataFrame
    """
    return load_raw_test_csv("synth_prediction_times.csv")


def load_synth_text() -> pl.DataFrame:
    """Load synth text data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pl.DataFrame
    """
    df = load_raw_test_csv("synth_text_data.csv")
    df["value"] = df["text"]
    df = df.drop(["text"])
    return df
