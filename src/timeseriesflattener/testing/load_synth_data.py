"""Loaders for synth data."""

import logging
from typing import Optional

import pandas as pd
from timeseriesflattener.utils import PROJECT_ROOT

log = logging.getLogger(__name__)


def load_raw_test_csv(filename: str, n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load raw csv.

    Args:
        filename (str): Name of the file to load.
        n_rows (int, optional): Number of rows to load. Defaults to None.
    """
    df = pd.read_csv(
        PROJECT_ROOT / "tests" / "test_data" / "raw" / filename,
        nrows=n_rows,
    )

    # Convert timestamp col to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df
