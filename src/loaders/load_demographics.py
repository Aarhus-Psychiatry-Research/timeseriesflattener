import pandas as pd
from wasabi import msg

from loaders.sql_load import sql_load


class LoadDemographics:
    def birthdays():
        msg.info("Loading birthdays")

        view = "[FOR_kohorte_demografi_inkl_2021_feb2022]"
        sql = f"SELECT dw_ek_borger, foedselsdato FROM [fct].{view}"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        # Typically handled by sql_load, but because foedselsdato doesn't contain "datotid" in its name,
        # We must handle it manually here
        df["foedselsdato"] = pd.to_datetime(df["foedselsdato"], format="%Y-%m-%d")

        df.rename(columns={"foedselsdato": "date_of_birth"}, inplace=True)

        msg.good("Loaded birthdays")
        return df.reset_index(drop=True)

    def sex():
        msg.info("Loading sexes")

        view = "[FOR_kohorte_demografi_inkl_2021_feb2022]"
        sql = f"SELECT dw_ek_borger, koennavn FROM [fct].{view}"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        df.loc[df["koennavn"] == "Mand", "koennavn"] = 1
        df.loc[df["koennavn"] == "Kvinde", "koennavn"] = 0

        df.rename(
            columns={"koennavn": "male"},
            inplace=True,
        )

        msg.good("Loaded sexes")
        return df.reset_index(drop=True)
