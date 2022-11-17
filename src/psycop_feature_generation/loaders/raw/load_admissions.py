"""Load admissions."""
from typing import Optional

import pandas as pd
from wasabi import msg

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.utils import data_loaders


@data_loaders.register("admissions")
def admissions(
    shak_code: Optional[int] = None,
    shak_sql_operator: Optional[str] = "=",
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load admissions. Outputs a value column containing length of admission
    in days.

    Args:
        shak_code (Optional[int], optional): SHAK code indicating where to keep/not keep visits from (e.g. 6600). Combines with
            shak_sql_operator, e.g. "!= 6600". Defaults to None, in which case all admissions are kept.
        shak_sql_operator (Optional[str], optional): Operator to use with shak_code. Defaults to "=".
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

        if shak_code is not None:
            sql += f" AND left({meta['location_col']}, {len(str(shak_code))}) {shak_sql_operator} {str(shak_code)}"

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

    msg.good("Loaded admissions data")

    return output_df.reset_index(drop=True)


@data_loaders.register("admissions_to_psychiatry")
def admissions_to_psychiatry(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load admissions to psychiatry."""
    return admissions(shak_code=6600, shak_sql_operator="=", n_rows=n_rows)


@data_loaders.register("admissions_to_somatic")
def admissions_to_somatic(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load admissions to somatic."""
    return admissions(shak_code=6600, shak_sql_operator="!=", n_rows=n_rows)
