import catalogue
from pandas import DataFrame

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


@resolve_fns.register("average")
def get_mean_in_group(grouped_df: DataFrame) -> DataFrame:
    return grouped_df.mean()


@resolve_fns.register("sum")
def get_sum_in_group(grouped_df: DataFrame) -> DataFrame:
    return grouped_df.sum()


@resolve_fns.register("count")
def get_count_in_group(grouped_df: DataFrame) -> DataFrame:
    return grouped_df.count()
