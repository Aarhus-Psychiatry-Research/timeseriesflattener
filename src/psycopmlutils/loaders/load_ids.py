import pandas as pd
from psycopmlutils.loaders.sql_load import sql_load
from psycopmlutils.utils import data_loaders
from wasabi import msg


class LoadIDs:
    def load(split: str) -> pd.DataFrame:
        """Loads ids for a given split

        Args:
            split (str): Which split to load IDs from. Takes either "train", "test" or "val".

        Returns:
            pd.DataFrame: Only dw_ek_borger column with ids
        """
        view = f"[psycop_{split}_ids]"
        sql = f"SELECT * FROM [fct].{view}"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        return df.reset_index(drop=True)
