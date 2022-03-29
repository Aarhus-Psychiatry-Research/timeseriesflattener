from pandas import DataFrame
import pandas as pd


def str_to_df(str) -> DataFrame:
    from io import StringIO

    df = pd.read_table(StringIO(str), sep=",", index_col=False)

    # Drop "Unnamed" cols
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]
