"""Utilites for generating synthetic data."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def replace_vals_with_na(
    df: pd.DataFrame, na_prob: float, na_ignore_cols: list[str] | None = None
) -> pd.DataFrame:
    """Replace values with NAs.

    Args:
        df (pd.DataFrame): The dataframe to replace values in.
        na_prob (float): The probability of replacing a value with NA.
        na_ignore_cols (Optional[list[str]]): The columns to ignore when replacing values.

    Returns:
        pd.DataFrame: The dataframe with values replaced with NAs.
    """
    mask = np.random.choice([True, False], size=df.shape, p=[na_prob, 1 - na_prob])
    df_ = df.mask(mask)

    # For all columns in df.columns if column is not in na_ignore_cols
    for col in df.columns:
        if na_ignore_cols and col in na_ignore_cols:
            continue
        df[col] = df_[col]

    return df
