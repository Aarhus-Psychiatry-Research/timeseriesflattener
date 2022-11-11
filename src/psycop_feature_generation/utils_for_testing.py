"""Utilites for testing."""

from collections.abc import Sequence
from io import StringIO
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from pandas.testing import assert_series_equal

from psycop_feature_generation.loaders.synth.raw.load_synth_data import (
    load_synth_outcome,
    load_synth_prediction_times,
)
from psycop_feature_generation.timeseriesflattener import FlattenedDataset
from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    TemporalSpec,
)
from psycop_feature_generation.utils import data_loaders


def convert_cols_with_matching_colnames_to_datetime(
    df: DataFrame,
    colname_substr: str,
) -> DataFrame:
    """Convert columns that contain colname_substr in their name to datetimes
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


def assert_flattened_data_as_expected(
    prediction_times_df: Union[pd.DataFrame, str],
    output_spec: TemporalSpec,
    expected_df: Optional[pd.DataFrame] = None,
    expected_values: Optional[Sequence[Any]] = None,
):
    """Take a prediction times df and output spec and assert that the flattened data is as expected."""
    if isinstance(prediction_times_df, str):
        prediction_times_df = str_to_df(prediction_times_df)

    flattened_ds = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        n_workers=4,
    )

    flattened_ds.add_temporal_col_to_flattened_dataset(output_spec=output_spec)

    if expected_df:
        for col in expected_df.columns:
            assert_series_equal(
                left=flattened_ds.df[col],
                right=expected_df[col],
                check_dtype=False,
            )
    elif expected_values:
        output = flattened_ds.df[output_spec.get_col_str()].values.tolist()
        expected = list(expected_values)

        for i, expected_val in enumerate(expected):
            # NaN != NaN, hence specific handling
            if np.isnan(expected_val):
                assert np.isnan(output[i])
            else:
                assert expected_val == output[i]
    else:
        raise ValueError("Must provide an expected set of data")


@data_loaders.register("load_event_times")
def load_event_times():
    """Load event times."""
    event_times_str = """dw_ek_borger,timestamp,value,
                    1,2021-12-30 00:00:01, 1
                    1,2021-12-29 00:00:02, 2
                    """

    return str_to_df(event_times_str)


def check_any_item_in_list_has_str(list_of_str: list, str_: str):
    """Check if any item in a list contains a string.

    Args:
        list_of_str (list): A list of strings.
        str_ (str): A string.

    Returns:
        bool: True if any item in the list contains the string.
    """
    return any(str_ in item for item in list_of_str)


@pytest.fixture(scope="function")
def synth_prediction_times():
    """Load the prediction times."""
    return load_synth_prediction_times()


@pytest.fixture(scope="function")
def synth_outcome():
    """Load the synth outcome times."""
    return load_synth_outcome()
