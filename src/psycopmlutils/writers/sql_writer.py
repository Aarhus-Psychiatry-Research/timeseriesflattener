import pandas as pd


def write_df_to_sql(
    df: pd.DataFrame,
    table_name: str,
    server: str = "BI-DPA-PROD",
    database: str = "USR_PS_Forsk",
    if_exists: str = "fail",
):
    """Writes a pandas dataframe to the SQL server.

    Args:
        df (pd.DataFrame): dataframe to write
        name (str): _description_
        server (str, optional): The SQL server. Defaults to "BI-DPA_PROD".
        database (str, optional): The SQL database. Defaults to "USR_PS_Forsk".
        if_exists (str, optional): What to do if the table already exists. Takes {'fail’, 'replace’, 'append’}. Defaults to "fail".
    """

    df.to_sql(
        name=table_name,
        con=f"mssql+pymssql://{server}/{database}",
        if_exists=if_exists,
        schema="fct",
    )
