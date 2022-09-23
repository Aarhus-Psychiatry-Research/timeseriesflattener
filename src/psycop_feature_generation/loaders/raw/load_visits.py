"""Loaders for visits to psychiatry."""

from typing import Optional

import pandas as pd
from wasabi import msg

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.utils import data_loaders


@data_loaders.register("physical_visits_to_psychiatry")
def physical_visits_to_psychiatry(
    where_clause: Optional[str] = None,
    where_separator: Optional[str] = "AND",
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load physical visits.

    Args:
        where_clause (Optional[str], optional): Extra where-clauses to add to the SQL call. E.g. dw_ek_borger = 1. Defaults to None. # noqa: DAR102
        where_separator (Optional[str], optional): Separator between where-clauses. Defaults to "AND".
        n_rows (Optional[int], optional): Number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with all physical visits to psychiatry. Has columns dw_ek_borger and timestamp.
    """
    # SHAK = 6600 ≈ in psychiatry
    d = {
        "LPR3": {
            "view": "[FOR_LPR3kontakter_psyk_somatik_inkl_2021_feb2022]",
            "datetime_col": "datotid_lpr3kontaktstart",
            "location_col": "shakkode_lpr3kontaktansvarlig",
            "where": "AND pt_type in ('Ambulant', 'Akut ambulant', 'Indlæggelse')",
        },
        "LPR2_outpatient": {
            "view": "[FOR_besoeg_psyk_somatik_LPR2_inkl_2021_feb2022]",
            "datetime_col": "datotid_start",
            "location_col": "shakafskode",
            "where": "AND psykambbesoeg = 1",
        },
        "LPR2_acute_outpatient": {
            "view": "[FOR_akutambulantekontakter_psyk_somatik_LPR2_inkl_2021_feb2022]",
            "datetime_col": "datotid_start",
            "location_col": "afsnit_stam",
            "where": "",
        },
        "LPR2_admissions": {
            "view": "[FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021_feb2022]",
            "datetime_col": "datotid_indlaeggelse",
            "location_col": "shakKode_kontaktansvarlig",
            "where": "",
        },
    }

    dfs = []

    for meta in d.values():
        cols = f"{meta['datetime_col']}, dw_ek_borger"

        sql = f"SELECT {cols} FROM [fct].{meta['view']} WHERE {meta['datetime_col']} IS NOT NULL AND left({meta['location_col']}, 4) = '6600' {meta['where']}"

        if where_clause is not None:
            sql += f" {where_separator} {where_clause}"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)
        df.rename(columns={meta["datetime_col"]: "timestamp"}, inplace=True)

        dfs.append(df)

    # Concat the list of dfs
    output_df = pd.concat(dfs)

    # 0,8% of visits are duplicates. Unsure if overlap between sources or errors in source data. Removing.
    output_df = output_df.drop_duplicates(
        subset=["timestamp", "dw_ek_borger"],
        keep="first",
    )
    msg.good("Loaded physical visits")

    return output_df.reset_index(drop=True)
