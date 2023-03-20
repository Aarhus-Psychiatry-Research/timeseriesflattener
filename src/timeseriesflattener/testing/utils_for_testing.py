"""Utilities for testing."""

from io import StringIO
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_series_equal
from timeseriesflattener import TimeseriesFlattener
from timeseriesflattener.feature_spec_objects import _AnySpec
from timeseriesflattener.testing.load_synth_data import (
    synth_predictor_binary,
)
from timeseriesflattener.utils import data_loaders


def convert_cols_with_matching_colnames_to_datetime(
    df: DataFrame,
    colname_substr: str,
) -> DataFrame:
    """Convert columns that contain colname_substr in their name to datetimes.

    Args:
        df (DataFrame): The df to convert. # noqa: DAR101
        colname_substr (str): Substring to match on. # noqa: DAR101

    Returns:
        DataFrame: The converted df
    """
    df.loc[:, df.columns.str.contains(colname_substr)] = df.loc[
        :,
        df.columns.str.contains(colname_substr),
    ].apply(pd.to_datetime)

    return df


def str_to_df(
    string: str,
    convert_timestamp_to_datetime: bool = True,
    convert_np_nan_to_nan: bool = True,
    convert_str_to_float: bool = False,
) -> DataFrame:
    """Convert a string representation of a dataframe to a dataframe.

    Args:
        string (str): A string representation of a dataframe.
        convert_timestamp_to_datetime (bool): Whether to convert the timestamp column to datetime. Defaults to True.
        convert_np_nan_to_nan (bool): Whether to convert np.nan to np.nan. Defaults to True.
        convert_str_to_float (bool): Whether to convert strings to floats. Defaults to False.

    Returns:
        DataFrame: A dataframe.
    """

    df = pd.read_table(StringIO(string), sep=",", index_col=False)

    if convert_timestamp_to_datetime:
        df = convert_cols_with_matching_colnames_to_datetime(df, "timestamp")

    if convert_np_nan_to_nan:
        # Convert "np.nan" str to the actual np.nan
        df = df.replace("np.nan", np.nan)

    if convert_str_to_float:
        # Convert all str to float
        df = df.apply(pd.to_numeric, axis=0, errors="coerce")

    # Drop "Unnamed" cols
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


def _get_value_cols_based_on_spec(
    df: pd.DataFrame,
    spec: _AnySpec,
) -> Union[str, List[str]]:
    """Get value columns based on spec. Checks if multiple value columns are present."""
    feature_name = spec.feature_name
    value_cols = df.columns[df.columns.str.contains(feature_name)].tolist()
    # to avoid indexing issues
    if len(value_cols) == 1:
        return value_cols[0]

    return value_cols


def assert_flattened_data_as_expected(
    prediction_times_df: Union[pd.DataFrame, str],
    output_spec: _AnySpec,
    expected_df: Optional[pd.DataFrame] = None,
    expected_values: Optional[Sequence[Any]] = None,
):
    """Flatten spec and assert that flattened data is as expected."""
    if isinstance(prediction_times_df, str):
        prediction_times_df = str_to_df(prediction_times_df)

    flattened_ds = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        n_workers=4,
        drop_pred_times_with_insufficient_look_distance=False,
    )

    flattened_ds.add_spec(  # pylint: disable=protected-access
        spec=output_spec,
    )

    if expected_df:
        for col in expected_df.columns:
            assert_series_equal(
                left=flattened_ds.get_df()[col],
                right=expected_df[col],
                check_dtype=False,
            )
    elif expected_values:
        output = flattened_ds.get_df()
        value_cols = _get_value_cols_based_on_spec(output, output_spec)
        output = flattened_ds.get_df()[value_cols].values.tolist()
        expected = list(expected_values)

        for i, expected_val in enumerate(expected):
            # NaN != NaN, hence specific handling
            if not isinstance(expected_val, (str, list)) and np.isnan(expected_val):
                assert np.isnan(output[i])
            else:
                assert expected_val == output[i]
    else:
        raise ValueError("Must provide an expected set of data")


def load_long_df_with_multiple_values() -> DataFrame:
    """Create a long df."""
    df = synth_predictor_binary()
    df = df.rename(columns={"value": "value_name_1"})
    df["value_name_2"] = df["value_name_1"]

    long_df = pd.melt(
        df,
        id_vars=["entity_id", "timestamp"],
        value_vars=["value_name_1", "value_name_2"],
        var_name="value_names",
        value_name="value",
    )

    return long_df


@data_loaders.register("load_event_times")
def load_event_times() -> DataFrame:
    """Load event times."""
    event_times_str = """entity_id,timestamp,value,
                    1,2021-12-30 00:00:01, 1
                    1,2021-12-29 00:00:02, 2
                    """

    return str_to_df(event_times_str)


def check_any_item_in_list_has_str(list_of_str: list, str_: str) -> bool:
    """Check if any item in a list contains a string.

    Args:
        list_of_str (list): A list of strings.
        str_ (str): A string.

    Returns:
        bool: True if any item in the list contains the string.
    """
    return any(str_ in item for item in list_of_str)
