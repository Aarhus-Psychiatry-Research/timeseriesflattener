from wasabi import msg
from psycopmlutils.loaders.sql_load import sql_load
from psycopmlutils.utils import data_loaders
import pandas as pd


class LoadOutcome:
    @data_loaders.register("t2d")
    def t2d():
        # msg.info("Loading t2d event times")

        full_csv_path = Path(
            r"E:\Users\adminmanber\Desktop\T2D\csv\first_t2d_diagnosis.csv"
        )

        df = pd.read_csv(full_csv_path)
        df = df[["dw_ek_borger", "datotid_first_t2d_diagnosis"]]
        df["value"] = 1

        df.rename(columns={"datotid_first_t2d_diagnosis": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

        msg.good("Finished loading t2d event times")
        output = df[["dw_ek_borger", "timestamp", "value"]]
        return output.reset_index(drop=True)

    @data_loaders.register("any_diabetes")
    def any_diabetes():
        df = sql_load(
            "SELECT * FROM [fct].[psycop_t2d_first_diabetes_any]",
            database="USR_PS_FORSK",
            chunksize=None,
        )

        df = df[["dw_ek_borger", "datotid_first_diabetes_any"]]
        df["value"] = 1

        df.rename(columns={"datotid_first_diabetes_any": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

        msg.good("Finished loading any_diabetes event times")
        output = df[["dw_ek_borger", "timestamp", "value"]]
        return output.reset_index(drop=True)
