import pandas as pd
from loaders import sql_load
from wasabi import msg


class LoadData:
    def physical_visits(frac=None):
        msg.info("Loading physical visits")

        view = "[FOR_besoeg_fysiske_fremmoeder_inkl_2021_feb2022]"
        sql = f"SELECT dw_ek_borger, datotid_start FROM [fct].{view} WHERE besoeg=1"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        if frac is not None:
            df = df.sample(frac=frac)

        df.rename(columns={"datotid_start": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

        msg.good("Loaded physical visits")
        return df

    def event_times():
        from pathlib import Path

        msg.info("Loading t2d event times")

        full_csv_path = Path(
            r"C:\Users\adminmanber\Desktop\manber-t2d\csv\first_t2d_diagnosis.csv"
        )

        df = pd.read_csv(str(full_csv_path))
        df = df[["dw_ek_borger", "datotid_first_t2d_diagnosis"]]
        df["val"] = 1

        df.rename(columns={"datotid_first_t2d_diagnosis": "timestamp"}, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%SZ")

        msg.good("Finished loading t2d event times")
        return df

    def hba1c_vals():
        msg.info("Loading hba1c")

        view = "[FOR_LABKA_NPU27300_HbA1c_inkl_2021]"
        sql = f"SELECT dw_ek_borger, datotid_proevemodtagelse, numerisksvar FROM [fct].{view}"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        df.rename(
            columns={"datotid_proevemodtagelse": "timestamp", "numerisksvar": "val"},
            inplace=True,
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

        msg.good("Loaded hba1c")
        return df
