"""Functions for resolving multiple values in a time-series into a single
value."""

# pylint: disable=missing-function-docstring

import catalogue
from pandas import DataFrame, Series
from scipy import stats

resolve_fns = catalogue.create("timeseriesflattener", "resolve_strategies")


@resolve_fns.register("latest")
def get_latest_value_in_group(grouped_df: DataFrame) -> DataFrame:
    """Get the latest value.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with only the latest value.
    """
    return grouped_df.last()


@resolve_fns.register("earliest")
def get_earliest_value_in_group(grouped_df: DataFrame) -> DataFrame:
    """Get the earliest value.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with only the earliest value in each group.
    """
    return grouped_df.first()


@resolve_fns.register("max")
def get_max_in_group(grouped_df: DataFrame) -> DataFrame:
    return grouped_df.max()


@resolve_fns.register("min")
def get_min_in_group(grouped_df: DataFrame) -> DataFrame:
    return grouped_df.min()


@resolve_fns.register("mean")
def get_mean_in_group(grouped_df: DataFrame) -> DataFrame:
    return grouped_df.mean(numeric_only=True)


@resolve_fns.register("sum")
def get_sum_in_group(grouped_df: DataFrame) -> DataFrame:
    return grouped_df.sum()


@resolve_fns.register("count")
def get_count_in_group(grouped_df: DataFrame) -> DataFrame:
    return grouped_df.count()


@resolve_fns.register("bool")
def get_bool_in_group(grouped_df: DataFrame) -> DataFrame:
    """Returns boolean value indicating whether or not event has occured in
    look ahead/behind window.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with value column containing only 0 or 1s.
    """
    grouped_df = grouped_df.count().reset_index()
    grouped_df[grouped_df["value"] > 0] = grouped_df.assign(value=1)
    return grouped_df


@resolve_fns.register("change_per_day")
def get_change_in_value_per_day(grouped_df: DataFrame) -> DataFrame:
    return grouped_df.apply(
        lambda x: Series({"value": stats.linregress(x.val, x.timestamp_val)[0]}),
    )
