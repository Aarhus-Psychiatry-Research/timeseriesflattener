"""Functions for resolving multiple values in a time-series into a single
value."""


import catalogue
from pandas import DataFrame, Series
from scipy import stats

aggregation_fns = catalogue.create("timeseriesflattener", "resolve_strategies")


def latest(grouped_df: DataFrame) -> Series:
    """Get the latest value.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with only the latest value.
    """
    return grouped_df.last()  # type: ignore


def earliest(grouped_df: DataFrame) -> Series:
    """Get the earliest value.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with only the earliest value in each group.
    """
    return grouped_df.first()  # type: ignore


def maximum(grouped_df: DataFrame) -> Series:
    return grouped_df.max()


def minimum(grouped_df: DataFrame) -> Series:
    return grouped_df.min()


def mean(grouped_df: DataFrame) -> Series:
    return grouped_df.mean(numeric_only=True)


def summed(grouped_df: DataFrame) -> Series:
    return grouped_df.sum()


def count(grouped_df: DataFrame) -> Series:
    return grouped_df.count()


def variance(grouped_df: DataFrame) -> Series:
    return grouped_df.var()


def boolean(grouped_df: DataFrame) -> DataFrame:
    """Returns a boolean value indicating whether or not event has occurred in
    look ahead/behind window.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with value column containing only 0 or 1s.
    """
    grouped_df = (
        grouped_df["timestamp_val"]
        .apply(lambda x: (~x.isna()).sum())
        .reset_index(name="value")
    )

    grouped_df.loc[grouped_df["value"] > 0, "value"] = 1

    return grouped_df


def change_per_day(grouped_df: DataFrame) -> DataFrame:
    """Returns the change in value per day.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with value column containing the change in value per day.
    """

    # Check if some patients have multiple values but only one timestamp
    if any(
        grouped_df.timestamp_val.apply(
            lambda x: len(set(x)) == 1 and len(x) > 1,
        ).values,
    ):
        raise ValueError(
            "One or more patients only have values with identical timestamps. There may be an error in the data.",
        )

    return grouped_df.apply(
        lambda x: Series(
            {"value": stats.linregress(x.timestamp_val, x.value)[0]},
        ),
    )


def concatenate(grouped_df: DataFrame) -> DataFrame:
    """Returns the concatenated values. This is useful for text data.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with value column containing the concatenated values.
    """

    return grouped_df.apply(
        lambda x: Series(
            {"value": " ".join(x.value)},
        ),
    )


def mean_number_of_characters(grouped_df: DataFrame) -> DataFrame:
    """Returns the mean length of values. This is useful for text data.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with value column containing the concatenated values.
    """
    return grouped_df.apply(
        lambda x: Series(
            {"value": x.value.str.len().mean()},
        ),
    )


def type_token_ratio(grouped_df: DataFrame) -> DataFrame:
    """Returns the type-token ratio. This is useful for text data.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        DataFrame: Dataframe with value column containing the concatenated values.
    """

    return grouped_df.apply(
        lambda x: Series(
            {
                "value": len(
                    set(
                        " ".join(
                            x.value.replace(
                                r"[^ÆØÅæøåA-Za-z0-9 ]+",
                                "",
                                regex=True,
                            ).str.lower(),
                        ).split(" "),
                    ),
                )
                / len(
                    " ".join(
                        x.value.replace(
                            r"[^ÆØÅæøåA-Za-z0-9 ]+",
                            "",
                            regex=True,
                        ).str.lower(),
                    ).split(" "),
                ),
            },
        ),
    )