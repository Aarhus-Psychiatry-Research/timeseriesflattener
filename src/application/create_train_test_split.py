import pandas as pd
from sklearn.model_selection import train_test_split
from psycoptts.add_outcomes import add_outcome_from_csv

from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

import urllib
import urllib.parse


def load_all_patients(view="FOR_kohorte_demografi_inkl_2021_feb2022"):
    view = f"{view}"
    query = "SELECT * FROM [fct]." + view

    print(f"Getting data from query: {query}")

    driver = "SQL Server"
    server = "BI-DPA-PROD"
    database = "USR_PS_Forsk"

    params = urllib.parse.quote(
        f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes"
    )
    engine = create_engine(
        "mssql+pyodbc:///?odbc_connect=%s" % params, poolclass=NullPool
    )
    conn = engine.connect().execution_options(stream_results=True)

    df = pd.read_sql(query, conn, chunksize=None)
    return df[["dw_ek_borger"]]


if __name__ == "__main__":
    outcomes = ["lung_cancer", "mamma_cancer"]
    random_state = 42

    combined_df = load_all_patients()

    for outcome in outcomes:
        combined_df = add_outcome_from_csv(
            combined_df, f"outcome_ids/{outcome}_cancer_ids.csv", outcome
        )

    X_train, X_intermediate = train_test_split(
        combined_df,
        test_size=0.3,
        random_state=random_state,
        stratify=combined_df[outcomes],
    )

    X_test, X_val = train_test_split(
        X_intermediate,
        test_size=0.5,
        random_state=random_state,
        stratify=X_intermediate[outcomes],
    )

    X_train["dw_ek_borger"].to_csv("csv/train_ids.csv", index=False)
    X_val["dw_ek_borger"].to_csv("csv/val_ids.csv", index=False)
    X_test["dw_ek_borger"].to_csv("csv/test_ids.csv", index=False)
