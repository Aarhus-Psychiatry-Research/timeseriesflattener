"""Loaders for patient IDs."""

from typing import Optional

import pandas as pd

from psycop_feature_generation.loaders.raw.sql_load import sql_load


def load_ids(split: str, n_rows: Optional[int] = None) -> pd.DataFrame:
    """Loads ids for a given split.

    Args:
        split (str): Which split to load IDs from. Takes either "train", "test" or "val". # noqa: DAR102
        n_rows: Number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame: Only dw_ek_borger column with ids
    """
    view = f"[psycop_{split}_ids]"

    sql = f"SELECT * FROM [fct].{view}"

    df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)

    return df.reset_index(drop=True)
