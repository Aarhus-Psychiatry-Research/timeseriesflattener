"""Loaders for T2D outcomes."""

# pylint: disable=missing-function-docstring

from typing import Optional

import pandas as pd
from wasabi import msg

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.utils import data_loaders


@data_loaders.register("t2d")
def t2d(n_rows: Optional[int] = None) -> pd.DataFrame:
    msg.info("Loading t2d event times")

    df = sql_load(
        "SELECT dw_ek_borger, timestamp FROM [fct].[psycop_t2d_first_diabetes_t2d] WHERE timestamp IS NOT NULL",
        database="USR_PS_FORSK",
        chunksize=None,
        format_timestamp_cols_to_datetime=True,
        n_rows=n_rows,
    )
    df["value"] = 1

    # 2 duplicates, dropping
    df = df.drop_duplicates(keep="first")

    msg.good("Finished loading t2d event times")
    return df.reset_index(drop=True)


@data_loaders.register("any_diabetes")
def any_diabetes(n_rows: Optional[int] = None):
    df = sql_load(
        "SELECT * FROM [fct].[psycop_t2d_first_diabetes_any] WHERE timestamp IS NOT NULL",
        database="USR_PS_FORSK",
        chunksize=None,
        n_rows=n_rows,
    )

    df = df[["dw_ek_borger", "datotid_first_diabetes_any"]]
    df["value"] = 1

    df.rename(columns={"datotid_first_diabetes_any": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    msg.good("Finished loading any_diabetes event times")
    output = df[["dw_ek_borger", "timestamp", "value"]]
    return output.reset_index(drop=True)
