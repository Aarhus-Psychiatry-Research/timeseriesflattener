"""Functions for resolving multiple values in a time-series into a single
value."""


from typing import Callable

import catalogue
from pandas import DataFrame, Series
from pandas.core.groupby.generic import DataFrameGroupBy
from scipy import stats

aggregation_fns = catalogue.create("timeseriesflattener", "resolve_strategies")
import pandas as pd

AggregationFunType = Callable[[DataFrameGroupBy], pd.DataFrame]


def latest(grouped_df: DataFrameGroupBy) -> DataFrame:
    """Get the latest value.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with only the latest value.
    """
    return grouped_df.last()


def earliest(grouped_df: DataFrameGroupBy) -> DataFrame:
    """Get the earliest value.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with only the earliest value in each group.
    """
    return grouped_df.first()


def maximum(grouped_df: DataFrameGroupBy) -> DataFrame:
    return grouped_df.max()


def minimum(grouped_df: DataFrameGroupBy) -> DataFrame:
    return grouped_df.min()


def mean(grouped_df: DataFrameGroupBy) -> DataFrame:
    return grouped_df.mean(numeric_only=True)


def summed(grouped_df: DataFrameGroupBy) -> DataFrame:
    return grouped_df.sum()


def count(grouped_df: DataFrameGroupBy) -> DataFrame:
    return grouped_df.count()


def variance(grouped_df: DataFrameGroupBy) -> DataFrame:
    return grouped_df.var()


def boolean(grouped_df: DataFrameGroupBy) -> DataFrame:
    """Returns a boolean value indicating whether or not event has occurred in
    look ahead/behind window.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with value column containing only 0 or 1s.
    """
    df = grouped_df["timestamp_val"].apply(lambda x: (~x.isna()).sum()).reset_index(name="value")

    df.loc[df["value"] > 0, "value"] = 1

    return df


def change_per_day(grouped_df: DataFrameGroupBy) -> DataFrame:
    """Returns the change in value per day.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with value column containing the change in value per day.
    """

    # Check if some patients have multiple values but only one timestamp
    if any(grouped_df.timestamp_val.apply(lambda x: len(set(x)) == 1 and len(x) > 1).values):
        raise ValueError(
            "One or more patients only have values with identical timestamps. There may be an error in the data."
        )

    return grouped_df.apply(
        lambda x: Series({"value": stats.linregress(x.timestamp_val, x.value)[0]})
    )
