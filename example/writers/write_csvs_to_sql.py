from psycopmlutils.writers.sql_writer import write_df_to_sql
from pathlib import Path
import pymssql
import pandas as pd
from wasabi import Printer

if __name__ == "__main__":
    msg = Printer(timestamp=True)

    midtx_path = Path("\\\\tsclient\\X\\MANBER01\\documentLibrary")

    splits = ["test", "val", "train"]

    for dataset_name in splits:
        rows_per_chunk = 2_000
        base_path = midtx_path / "feature_generation" / "toy_example"

        msg.info(f"{dataset_name}: Reading features")
        feature_path = base_path / f"{dataset_name}_features.csv"
        df_features = pd.read_csv(feature_path)

        msg.info(f"{dataset_name}: Writing features")
        write_df_to_sql(
            df=df_features,
            table_name=f"psycop_t2d_{dataset_name}_features",
            if_exists="replace",
            rows_per_chunk=rows_per_chunk,
        )

        msg.info(f"{dataset_name}: Reading events")
        event_path = base_path / f"{dataset_name}_events.csv"
        df_events = pd.read_csv(event_path)

        msg.info(f"{dataset_name}: Writing events")
        write_df_to_sql(
            df=df_events,
            table_name=f"psycop_t2d_{dataset_name}_events",
            if_exists="replace",
            rows_per_chunk=rows_per_chunk,
        )

        msg.good(f"{dataset_name}: Succesfully wrote {dataset_name} to sql")
