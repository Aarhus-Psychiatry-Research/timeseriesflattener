import urllib
import urllib.parse
from typing import Generator, Optional, Union

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool


def sql_load(
    query: str,
    server: str = "BI-DPA-PROD",
    database: str = "USR_PS_Forsk",
    chunksize: Optional[int] = None,
    format_timestamp_cols_to_datetime: bool = True,
) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
    """Function to load a SQL query. If chunksize is None, all data will be loaded into memory.
    Otherwise, will stream the data in chunks of chunksize as a generator

    Args:
        query (str): The SQL query
        server (str): The BI server
        database (str): The BI database
        chunksize (int, optional): Defaults to 1000.
        format_timestamp_cols_to_datetime (bool, optional): Whether to format all columns with "datotid" in their name as pandas datetime. Defaults to true.

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
        "DRIVER={0};SERVER={1};DATABASE={2};Trusted_Connection=yes".format(
            driver, server, database
        )
    )

    engine = create_engine(
        "mssql+pyodbc:///?odbc_connect=%s" % params, poolclass=NullPool
    )

    conn = engine.connect().execution_options(
        stream_results=True, fast_executemany=True
    )

    df = pd.read_sql(query, conn, chunksize=chunksize)

    if format_timestamp_cols_to_datetime:
        datetime_col_names = [
            colname
            for colname in df.columns
            if any(str for str in ["datotid", "timestamp"] in colname.lower)
        ]

        df[datetime_col_names] = df[datetime_col_names].apply(pd.to_datetime)

    engine.dispose()

    return df
