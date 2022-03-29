import pandas as pd


def add_outcome_from_df(df_in, df_outcome, new_col_name, id_colname="dw_ek_borger"):
    df_outcome[new_col_name] = 1

    return pd.merge(df_in, df_outcome, how="left", on=id_colname).fillna(0)


def add_outcome_from_csv(df_in, df_outcome_path, new_colname):
    return add_outcome_from_df(
        df_in, pd.read_csv(df_outcome_path), new_colname, id_colname="dw_ek_borger"
    )
