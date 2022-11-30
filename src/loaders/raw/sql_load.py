"""Utility functions for SQL loading."""

import urllib
import urllib.parse
from collections.abc import Generator
from typing import Optional, Union

import pandas as pd
from sqlalchemy import create_engine


def sql_load(
    query: str,
    server: str = "BI-DPA-PROD",
    database: str = "USR_PS_Forsk",
    chunksize: Optional[int] = None,
    format_timestamp_cols_to_datetime: Optional[bool] = True,
    n_rows: Optional[int] = None,
) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
    """Function to load a SQL query. If chunksize is None, all data will be
    loaded into memory. Otherwise, will stream the data in chunks of chunksize
    as a generator.

    Args:
        query (str): The SQL query
        server (str): The BI server
        database (str): The BI database
        chunksize (int, optional): Defaults to 1000.
        format_timestamp_cols_to_datetime (bool, optional): Whether to format all
            columns with "datotid" in their name as pandas datetime. Defaults to true.
        n_rows (int, optional): Defaults to None. If specified, only returns the first n rows.

    Returns:
        Union[pd.DataFrame, Generator[pd.DataFrame]]: DataFrame or generator of DataFrames

    Example:
        # From USR_PS_Forsk
        >>> view = "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret_2011]"
        >>> sql = "SELECT * FROM [fct]." + view
        >>> df = sql_load(sql, chunksize = None)
    """
    driver = "SQL Server"
    params = urllib.parse.quote(
        f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes",
    )

    if n_rows:
        query = query.replace("SELECT", f"SELECT TOP {n_rows} ")

    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    conn = engine.connect().execution_options(
        stream_results=True,
        fast_executemany=True,
    )

    df = pd.read_sql(query, conn, chunksize=chunksize)

    if format_timestamp_cols_to_datetime:
        datetime_col_names = [
            colname
            for colname in df.columns
            if any(substr in colname.lower() for substr in ["datotid", "timestamp"])
        ]

        df[datetime_col_names] = df[datetime_col_names].apply(pd.to_datetime)

    conn.close()
    engine.dispose()

    return df
