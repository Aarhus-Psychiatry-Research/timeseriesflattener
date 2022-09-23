"""Exemplifies how to debug an SQL loader function.

Two primary purposes:
1. Check the possible values the dataframe can contain and
2. Check that the dataframe conforms to the required format.
"""

from typing import Any

import pandas as pd

import psycop_feature_generation.loaders.raw as raw_loaders
from psycop_feature_generation.data_checks.raw.check_predictor_lists import check_raw_df


def will_it_float(value: Any) -> bool:
    """Check if a value can be converted to a float.

    Args:
        value (Any): A value.

    Returns:
        bool: True if the value can be converted to a float, False otherwise.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_prop_of_each_unique_value_for_non_floats(series: pd.Series) -> pd.Series:
    """Get the proportion of each unique value in a series, but only for value
    which cannot be converted to floats.

    Args:
        series (pd.Series): A pandas series.

    Returns:
        pd.Series: A series with the proportion of each unique value in the
        original series.
    """
    if series.dtype in ["float64", "int64"]:
        return "All values in series can be converted to floats."

    # Find all strings that start with a number
    starts_with_number_idx = series.str.match(r"^\d+").fillna(False)

    # Convert all strings that start with a number to floats
    # Replace all "," with "."
    series[starts_with_number_idx] = series[starts_with_number_idx].str.replace(
        ",",
        ".",
    )

    # Convert all str in series to float
    series[starts_with_number_idx] = series[starts_with_number_idx].astype(float)

    # Get the unique values
    unique_values = series.unique()

    # Get the proportion of each unique value
    prop_of_each_unique_value = series.value_counts(
        normalize=True,
    )

    # Get the unique values which cannot be converted to floats
    non_float_unique_values = [
        value
        for value in unique_values
        if (not will_it_float(value) and value is not None)
    ]

    # Get the proportion of each unique value which cannot be converted to floats
    prop_of_each_non_float_unique_value = prop_of_each_unique_value[
        non_float_unique_values
    ]

    return prop_of_each_non_float_unique_value


if __name__ == "__main__":
    df = raw_loaders.load_lab_results.ldl(
        n=1_000,
        values_to_load="numerical_and_coerce",
    )

    value_props = get_prop_of_each_unique_value_for_non_floats(df["value"])
    print(value_props)

    errors, duplicates = check_raw_df(
        df=df,
        required_columns=["dw_ek_borger", "timestamp", "value"],
        subset_duplicates_columns=["dw_ek_borger", "timestamp", "value"],
    )
    print(errors)
