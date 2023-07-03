"""Functions for resolving multiple values in a time-series into a single
value."""


from typing import Callable

import catalogue
from pandas import Series
from scipy import stats

aggregation_fns = catalogue.create("timeseriesflattener", "resolve_strategies")
import polars as pl
from polars.lazyframe.groupby import LazyGroupBy

AggregationFunType = Callable[[LazyGroupBy], pl.LazyFrame]


def latest(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    """Get the latest value.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        pl.LazyFrame: Dataframe with only the latest value.
    """
    return grouped_df.last()


def earliest(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    """Get the earliest value.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        pl.LazyFrame: Dataframe with only the earliest value in each group.
    """
    return grouped_df.first()


def maximum(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    return grouped_df.max()


def minimum(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    return grouped_df.min()


def mean(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    return grouped_df.mean()


def summed(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    return grouped_df.sum()


def count(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    return grouped_df.count()


def variance(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    return grouped_df.var()


def boolean(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    """Returns a boolean value indicating whether or not event has occurred in
    look ahead/behind window.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        pl.LazyFrame: Dataframe with value column containing only 0 or 1s.
    """
    df = (
        grouped_df["timestamp_val"]
        .apply(lambda x: (~x.isna()).sum())
        .reset_index(name="value")
    )

    df.loc[df["value"] > 0, "value"] = 1

    return df


def change_per_day(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    """Returns the change in value per day.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        pl.LazyFrame: Dataframe with value column containing the change in value per day.
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


def concatenate(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    """Returns the concatenated values. This is useful for text data.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        pl.LazyFrame: Dataframe with value column containing the concatenated values.
    """

    return grouped_df.apply(
        lambda x: Series(
            {"value": " ".join(x.value)},
        ),
    )


def mean_number_of_characters(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    """Returns the mean length of values. This is useful for text data.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        pl.LazyFrame: Dataframe with value column containing the concatenated values.
    """
    return grouped_df.apply(
        lambda x: Series(
            {"value": x.value.str.len().mean()},
        ),
    )


def type_token_ratio(grouped_df: LazyGroupBy) -> pl.LazyFrame:
    """Returns the type-token ratio. This is useful for text data.

    Args:
        grouped_df (DataFrame): A dataframe sorted by descending timestamp, grouped by citizen.

    Returns:
        pl.LazyFrame: Dataframe with value column containing the concatenated values.
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
