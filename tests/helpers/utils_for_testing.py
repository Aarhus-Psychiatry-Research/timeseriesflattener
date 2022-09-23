"""Utilites for testing."""

from collections.abc import Callable
from io import StringIO
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from psycopmlutils.timeseriesflattener.flattened_dataset import FlattenedDataset
from psycopmlutils.utils import data_loaders, generate_feature_colname


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


def assert_flattened_values_as_expected(
    prediction_times_str: str,
    event_times_str: str,
    direction: str,
    interval_days: float,
    resolve_multiple: Union[Callable, str],
    expected_flattened_values: list,
    values_colname: Union[str, list] = "value",
    fallback: Any = np.NaN,
    df_as_str: Optional[bool] = True,
):
    """Run tests from string representations of dataframes.

    Args:
        prediction_times_str (str): A string-representation of prediction-times df.
        event_times_str (str): A string-representation of an event-times df.
        direction (str): Whether to look ahead or behind
        interval_days (float): How far to look in direction
        resolve_multiple (Callable): How to handle multiple values within the lookahead window. Takes a a function that takes a list as an argument and returns a float.
        expected_flattened_values (list): A list of the expected values in the value column of the flattened df
        values_colname (Optional[Union[str, list]]): Column name for the new values. Defaults to "val".
        fallback (Any): What to fill if no outcome within lookahead days. Defaults to 0.
        df_as_str (bool, optional): Whether the input dfs are strings. Defaults to True.

    Raises:
        ValueError: If direction is neither ahead nor behind.
    """

    if df_as_str:
        df_prediction_times = str_to_df(prediction_times_str)
        df_event_times = str_to_df(event_times_str)
    else:
        df_prediction_times = convert_cols_with_matching_colnames_to_datetime(
            prediction_times_str,
            "timestamp",
        )
        df_event_times = convert_cols_with_matching_colnames_to_datetime(
            event_times_str,
            "timestamp",
        )

    dataset = FlattenedDataset(
        prediction_times_df=df_prediction_times,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
    )

    if direction == "behind":
        new_col_name_prefix = "pred"
        dataset.add_temporal_predictor(
            predictor_df=df_event_times,
            lookbehind_days=interval_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            pred_name=values_colname,
        )
    elif direction == "ahead":
        new_col_name_prefix = "outc"
        dataset.add_temporal_outcome(
            outcome_df=df_event_times,
            lookahead_days=interval_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            pred_name=values_colname,
        )
    else:
        raise ValueError("direction only takes look ahead or behind")

    flattened_values_colname = generate_feature_colname(
        prefix=new_col_name_prefix,
        out_col_name=values_colname,
        interval_days=interval_days,
        resolve_multiple=resolve_multiple,
        fallback=fallback,
    )

    if isinstance(flattened_values_colname, str):
        flattened_values_colname = [flattened_values_colname]

    expected_flattened_values = pd.DataFrame(
        expected_flattened_values,
        columns=flattened_values_colname,
    )

    pd.testing.assert_frame_equal(
        left=dataset.df[flattened_values_colname].reset_index(drop=True),
        right=expected_flattened_values[flattened_values_colname].reset_index(
            drop=True,
        ),
        check_dtype=False,
    )


def assert_flattened_outcome_as_expected(
    prediction_times_df_str: str,
    outcome_df_str: str,
    lookahead_days: float,
    expected_flattened_values: list,
    resolve_multiple: Union[Callable, str],
    values_colname: str = "value",
    fallback: Any = np.NaN,
    df_as_str: bool = True,
):
    """Run tests from string representations of dataframes.

    Args:
        prediction_times_df_str (str): A string-representation of prediction-times.
        outcome_df_str (str): A string-representation of an outcome df.
        lookahead_days (float): How far ahead from the prediction time to look for
            outcomes.
        expected_flattened_values (list): A list of the expected values in the value # noqa: DAR101
            column of the flattened df.
        resolve_multiple (Callable): How to handle multiple values within the lookahead window.
            Takes a a function that takes a list as an argument and returns a float.
        values_colname (str): Column name for the new values. Defaults to "val".
        fallback (Any): What to fill if no outcome within lookahead days. Defaults to np.NaN.
        df_as_str (bool, optional): Whether the input dfs are strings. Defaults to True.
    Example:
        >>> prediction_times_df_str = '''dw_ek_borger,timestamp,
        >>>                     1,2021-12-31 00:00:00
        >>>                     '''
        >>> outcome_df_str = '''dw_ek_borger,timestamp,value,
        >>>                     1,2021-12-30 23:59:59, 1
        >>>                     '''
        >>> assert_flattened_outcome_as_expected(
        >>>     prediction_times_df_str=prediction_times_df_str,
        >>>     outcome_df_str=outcome_df_str,
        >>>     lookahead_days=2,
        >>>     resolve_multiple=max,
        >>>     fallback = 0,
        >>>     expected_flattened_vals=[np.NaN],
        >>> )
    """

    assert_flattened_values_as_expected(
        interval_days=lookahead_days,
        direction="ahead",
        prediction_times_str=prediction_times_df_str,
        event_times_str=outcome_df_str,
        resolve_multiple=resolve_multiple,
        expected_flattened_values=expected_flattened_values,
        values_colname=values_colname,
        fallback=fallback,
        df_as_str=df_as_str,
    )


def assert_flattened_predictor_as_expected(
    prediction_times_df_str: str,
    predictor_df_str: str,
    lookbehind_days: float,
    resolve_multiple: Union[Callable, str],
    expected_flattened_values: list,
    values_colname: str = "value",
    fallback: Any = np.NaN,
):
    """Run tests from string representations of dataframes.

    Args:
        prediction_times_df_str (str): A string-representation of prediction-times df
        predictor_df_str (str): A string-representation of the predictor df
        lookbehind_days (float): How far to look behind.
        resolve_multiple (Callable): How to handle multiple values within the lookahead window. Takes a a function that takes a list as an argument and returns a float.
        expected_flattened_values (list): A list of the expected values in the value column of the flattened df.
        values_colname (str): Column name for the new values. Defaults to "val".
        fallback (Any): What to fill if no outcome within lookahead days. Defaults to np.NaN.

    Example:
        >>> prediction_times_df_str =  '''dw_ek_borger,timestamp,
        >>>                            1,2021-12-31 00:00:00
        >>>                            '''
        >>> predictor_df_str =  '''dw_ek_borger,timestamp,value,
        >>>                     1,2022-01-01 00:00:01, 1
        >>>                     '''
        >>> assert_flattened_predictor_as_expected(
        >>>     prediction_times_df_str=prediction_times_df_str,
        >>>     predictor_df_str=predictor_df_str,
        >>>     lookbehind_days=2,
        >>>     resolve_multiple=max,
        >>>     expected_flattened_values=[-1],
        >>>     fallback=-1,
        >>> )
    """

    assert_flattened_values_as_expected(
        interval_days=lookbehind_days,
        direction="behind",
        prediction_times_str=prediction_times_df_str,
        event_times_str=predictor_df_str,
        resolve_multiple=resolve_multiple,
        expected_flattened_values=expected_flattened_values,
        values_colname=values_colname,
        fallback=fallback,
    )


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
