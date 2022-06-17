from wasabi import msg
from psycopmlutils.loaders.sql_load import sql_load
from psycopmlutils.utils import data_loaders
import pandas as pd


class LoadOutcome:
    @data_loaders.register("t2d")
    def t2d():
        # msg.info("Loading t2d event times")

        df = sql_load(
            "SELECT dw_ek_borger, timestamp FROM [fct].[psycop_t2d_first_diabetes_t2d]",
            database="USR_PS_FORSK",
            chunksize=None,
            format_timestamp_cols_to_datetime=True,
        )
        df["value"] = 1

        msg.good("Finished loading t2d event times")
        return df.reset_index(drop=True)

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
