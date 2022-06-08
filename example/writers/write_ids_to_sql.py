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
        base_path = midtx_path / "train-test-splits"

        msg.info(f"{dataset_name}: Reading ids")
        id_path = base_path / f"{dataset_name}_ids.csv"
        df_ids = pd.read_csv(id_path)

        write_df_to_sql(
            df=df_ids,
            table_name=f"psycop_{dataset_name}_ids",
            if_exists="replace",
            rows_per_chunk=rows_per_chunk,
        )

        msg.good(f"Succesfully wrote {dataset_name} ids to SQL server")
