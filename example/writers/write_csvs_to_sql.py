from psycopmlutils.writers.sql_writer import write_df_to_sql
from pathlib import Path
import pymssql
import pandas as pd

if __name__ == "__main__":
    midtx_path = Path("\\\\tsclient\\X\\MANBER01\\documentLibrary")

    splits = ["test", "val", "train"]

    for dataset_name in splits:
        base_path = midtx_path / "feature_generation" / "toy_example"

        feature_path = base_path / f"{dataset_name}_features.csv"
        df_features = pd.read_csv(feature_path)
        write_df_to_sql(df=df_features, table_name=f"t2d_{dataset_name}_features", if_exists="replace")

        # event_path = base_path / f"{dataset_name}_events.csv"
        # df_events = pd.read_csv(event_path)
