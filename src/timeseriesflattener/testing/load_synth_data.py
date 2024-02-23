"""Loaders for synth data."""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEST_DATA_PATH = PROJECT_ROOT / "src" / "timeseriesflattener" / "testing" / "test_data"


def load_raw_test_csv(filename: str) -> pl.DataFrame:
    df = pl.read_csv(TEST_DATA_PATH / "raw" / filename)

    if "timestamp" in df.columns:
        df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime))

    return df


def load_synth_predictor_float() -> pl.DataFrame:
    return load_raw_test_csv("synth_raw_float_1.csv")


def load_synth_sex() -> pl.DataFrame:
    return load_raw_test_csv("synth_sex.csv")


def synth_predictor_binary() -> pl.DataFrame:
    return load_raw_test_csv("synth_raw_binary_1.csv")


def load_synth_outcome() -> pl.DataFrame:
    # Get first row for each id
    df = load_raw_test_csv("synth_raw_binary_2.csv")
    df = df.groupby("entity_id").last()

    # Drop all rows with a value equal to 1
    df = df.filter(pl.col("value") == 1)
    return df


def load_synth_prediction_times() -> pl.DataFrame:
    return load_raw_test_csv("synth_prediction_times.csv")


def load_synth_text() -> pl.DataFrame:
    df = load_raw_test_csv("synth_text_data.csv").rename({"text": "value"})
    return df


def load_synth_birthdays() -> pl.DataFrame:
    return load_raw_test_csv("synth_birthdays.csv").with_columns(
        pl.col("birthday").str.strptime(pl.Datetime)
    )
