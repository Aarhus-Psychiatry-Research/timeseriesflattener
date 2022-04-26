from typing import Callable, List, Union

import pandas as pd
from pandas import DataFrame
from timeseriesflattener.flattened_dataset import FlattenedDataset


def str_to_df(str) -> DataFrame:
    from io import StringIO

    df = pd.read_table(StringIO(str), sep=",", index_col=False)

    df = convert_cols_with_matching_colnames_to_datetime(df, "timestamp")

    # Drop "Unnamed" cols
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


def convert_cols_with_matching_colnames_to_datetime(
    df: DataFrame, colname_substr: str
) -> DataFrame:
    """Convert columns that contain colname_substr in their name to datetimes
    Args:
        df (DataFrame)
        colname_substr (str): Substring to match on.
    """
    df.loc[:, df.columns.str.contains(colname_substr)] = df.loc[
        :, df.columns.str.contains(colname_substr)
    ].apply(pd.to_datetime)

    return df


def assert_flattened_outcome_as_expected(
    prediction_times_df_str: str,
    outcome_df_str: str,
    lookahead_days: float,
    expected_flattened_vals: List,
    resolve_multiple: Union[Callable, str],
    values_colname: str = "val",
    fallback: List = 0,
):
    """Run tests from string representations of dataframes.
    Args:
        prediction_times_df_str (str): A string-representation of prediction-times
        outcome_df_str (str): A string-representation of an outcome df
        lookahead_days (float): _description_
        expected_flattened_vals (List): A list of the expected values in the value column of the flattened df
        resolve_multiple (Callable): How to handle multiple values within the lookahead window. Takes a a function that takes a list as an argument and returns a float.
        values_colname (str, optional): Column name for the new values. Defaults to "val".
        fallback (List, optional): What to fill if no outcome within lookahead days. Defaults to 0.
    Example:
        >>> prediction_times_df_str = '''dw_ek_borger,timestamp,
        >>>                     1,2021-12-31 00:00:00
        >>>                     '''
        >>> outcome_df_str = '''dw_ek_borger,timestamp,val,
        >>>                     1,2021-12-30 23:59:59, 1
        >>>                     '''
        >>>
        >>> assert_flattened_outcome_as_expected(
        >>>     prediction_times_df_str=prediction_times_df_str,
        >>>     outcome_df_str=outcome_df_str,
        >>>     lookahead_days=2,
        >>>     resolve_multiple=get_max_value_from_list_of_events,
        >>>     fallback = 0,
        >>>     expected_flattened_vals=[0],
        >>> )
    """

    assert_flattened_vals_as_expected(
        interval_days=lookahead_days,
        direction="ahead",
        prediction_times_str=prediction_times_df_str,
        event_times_str=outcome_df_str,
        resolve_multiple=resolve_multiple,
        expected_flattened_vals=expected_flattened_vals,
        values_colname=values_colname,
        fallback=fallback,
    )


def assert_flattened_predictor_as_expected(
    prediction_times_df_str: str,
    predictor_df_str: str,
    lookbehind_days: float,
    resolve_multiple: Union[Callable, str],
    expected_flattened_vals: List,
    values_colname: str = "val",
    fallback: List = 0,
):
    """Run tests from string representations of dataframes.
    Args:
        prediction_times_df_str (str): A string-representation of prediction-times df
        predictor_df_str (str): A string-representation of the predictor df
        lookbehind_days (float): How far to look behind
        resolve_multiple (Callable): How to handle multiple values within the lookahead window. Takes a a function that takes a list as an argument and returns a float.
        expected_flattened_vals (List): A list of the expected values in the value column of the flattened df
        values_colname (str, optional): Column name for the new values. Defaults to "val".
        fallback (List, optional): What to fill if no outcome within lookahead days. Defaults to 0.
    Example:
        >>> prediction_times_df_str =  '''dw_ek_borger,timestamp,
        >>>                            1,2021-12-31 00:00:00
        >>>                            '''
        >>> predictor_df_str =  '''dw_ek_borger,timestamp,val,
        >>>                     1,2022-01-01 00:00:01, 1
        >>>                     '''
        >>>
        >>> assert_flattened_predictor_as_expected(
        >>>     prediction_times_df_str=prediction_times_df_str,
        >>>     predictor_df_str=predictor_df_str,
        >>>     lookbehind_days=2,
        >>>     resolve_multiple=get_max_value_from_list_of_events,
        >>>     expected_flattened_vals=[-1],
        >>>     fallback=-1,
        >>> )
    """

    assert_flattened_vals_as_expected(
        interval_days=lookbehind_days,
        direction="behind",
        prediction_times_str=prediction_times_df_str,
        event_times_str=predictor_df_str,
        resolve_multiple=resolve_multiple,
        expected_flattened_vals=expected_flattened_vals,
        values_colname=values_colname,
        fallback=fallback,
    )


def assert_flattened_vals_as_expected(
    prediction_times_str: str,
    event_times_str: str,
    direction: str,
    interval_days: float,
    resolve_multiple: Union[Callable, str],
    expected_flattened_vals: List,
    values_colname: str = "val",
    fallback: List = 0,
):
    """Run tests from string representations of dataframes.
    Args:
        Args:
        prediction_times_df_str (str): A string-representation of prediction-times df
        predictor_df_str (str): A string-representation of the predictor df
        direction (str): Whether to look ahead or behind
        interval_days (float): How far to look in direction
        resolve_multiple (Callable): How to handle multiple values within the lookahead window. Takes a a function that takes a list as an argument and returns a float.
        expected_flattened_vals (List): A list of the expected values in the value column of the flattened df
        values_colname (str, optional): Column name for the new values. Defaults to "val".
        fallback (List, optional): What to fill if no outcome within lookahead days. Defaults to 0.
    Raises:
        ValueError: _description_
    """

    df_prediction_times = str_to_df(prediction_times_str)
    df_event_times = str_to_df(event_times_str)

    dataset = FlattenedDataset(
        prediction_times_df=df_prediction_times,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
    )

    if direction == "behind":
        dataset.add_predictor(
            predictor_df=df_event_times,
            lookbehind_days=interval_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
        )
    elif direction == "ahead":
        dataset.add_outcome(
            outcome_df=df_event_times,
            lookahead_days=interval_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
        )
    else:
        raise ValueError("direction only takes look ahead or behind")

    flattened_vals_colname = f"{values_colname}_within_{interval_days}_days"

    expected_flattened_vals = pd.DataFrame(
        {flattened_vals_colname: expected_flattened_vals}
    )

    pd.testing.assert_series_equal(
        left=dataset.df[flattened_vals_colname].reset_index(drop=True),
        right=expected_flattened_vals[flattened_vals_colname].reset_index(drop=True),
        check_dtype=False,
    )
