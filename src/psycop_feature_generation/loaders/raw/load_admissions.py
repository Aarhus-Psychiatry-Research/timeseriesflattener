"""Load admissions."""

from typing import Optional

import pandas as pd
from wasabi import msg

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.utils import data_loaders


@data_loaders.register("admissions")
def admissions(
    where_clause: Optional[str] = None,
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load admissions. Outputs a value column containng lenght of admission in
    day.

    Args:
        where_clause (Optional[str], optional): SHAK code determining which rows to keep. Defaults to None.
        n_rows (Optional[int], optional): Number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with all physical visits to psychiatry. Has columns dw_ek_borger, timestamp and value (length of admissions in days).
    """

    # SHAK = 6600 ≈ in psychiatry
    d = {
        "LPR3": {
            "view": "[FOR_LPR3kontakter_psyk_somatik_inkl_2021_feb2022]",
            "datetime_col": "datotid_lpr3kontaktstart",
            "value_col": "datotid_lpr3kontaktslut",
            "location_col": "shakkode_lpr3kontaktansvarlig",
            "where": "AND pt_type = 'Indlæggelse'",
        },
        "LPR2_admissions": {
            "view": "[FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021_feb2022]",
            "datetime_col": "datotid_indlaeggelse",
            "value_col": "datotid_udskrivning",
            "location_col": "shakKode_kontaktansvarlig",
            "where": "",
        },
    }

    dfs = []

    for meta in d.values():
        cols = f"{meta['datetime_col']}, {meta['value_col']}, dw_ek_borger"

        sql = f"SELECT {cols} FROM [fct].{meta['view']} WHERE {meta['datetime_col']} IS NOT NULL AND {meta['value_col']} IS NOT NULL {meta['where']}"

        if where_clause is not None:
            sql += f" AND left({meta['location_col']}, 4) = {where_clause}"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)
        df.rename(
            columns={meta["datetime_col"]: "timestamp", meta["value_col"]: "value"},
            inplace=True,
        )

        dfs.append(df)

    # Concat the list of dfs
    output_df = pd.concat(dfs)

    # 0,8% of visits are duplicates. Unsure if overlap between sources or errors in source data. Removing.
    output_df = output_df.drop_duplicates(
        subset=["timestamp", "dw_ek_borger"],
        keep="first",
    )

    # Change value column to length of admission in days
    output_df["value"] = (
        output_df["value"] - output_df["timestamp"]
    ).dt.total_seconds() / 86400

    msg.good("Loaded all admissions")

    return output_df.reset_index(drop=True)
