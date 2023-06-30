import datetime
from email.headerregistry import Group
from functools import partial

import polars as pl
import statsmodels.api as sm
from polars.dataframe.groupby import GroupBy as plGroupBy
from scipy import stats

from timeseriesflattener.testing.utils_for_testing import str_to_pl_df


def change_per_day(df: pl.DataFrame, y_colname: str, x_colname: str) -> pl.DataFrame:
    y_values = df[y_colname].to_numpy()
    x_values = df[x_colname].to_numpy()

    result = stats.linregress(x_values, y_values)
    return df.with_columns(pl.lit(result.slope).alias("value"))


def test_aggregation_change_per_day():
    df = str_to_pl_df(
        """entity_id,x,value,
                        1,1,1
                        1,2,2
                        2,1,1
                        2,5,2
                        """
    ).lazy()

    input_df = pl.concat([df for _ in range(2_400_000)]).collect()

    start_time = datetime.datetime.now()
    return_df = input_df.groupby("entity_id").apply(
        partial(change_per_day, y_colname="value", x_colname="x")
    )
    end_time = datetime.datetime.now()

    duration_seconds = (end_time - start_time).microseconds / 1_000_000

    assert return_df["value"].to_list() == [0.25, 0.25, 1.0, 1.0]
    assert duration_seconds < 5
