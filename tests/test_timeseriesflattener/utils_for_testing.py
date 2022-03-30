from timeseriesflattener.flattened_dataset import *
from timeseriesflattener.resolve_multiple_functions import *

import pandas as pd


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


def assert_flattened_outcome_vals_as_expected(
    lookahead_days: float,
    prediction_times_str: str,
    resolve_multiple: Callable,
    event_times_str: str,
    expected_flattened_vals: List,
    values_colname: str = "val",
    fallback: List = 0,
):
    """Run tests from string representations of dataframes.

    Args:
        prediction_times (str): A string-representation of prediction-times
        event_times (str): A string-representation of event_times
        expected_values (List): The expected values for each prediction time

    Example:
        >>> prediction_times_str =  '''dw_ek_borger,timestamp,
        >>>                         1,2021-12-31 00:00:00
        >>>                         '''
        >>>
        >>> event_times_str =  '''dw_ek_borger,timestamp,val,
        >>>                    1,2022-01-01 00:00:01, 1
        >>>                    '''
        >>>
        >>> run_tests_from_strings(prediction_times_str, event_times_str, [1])
    """
    df_prediction_times = str_to_df(prediction_times_str)
    df_event_times = str_to_df(event_times_str)

    dataset = FlattenedDataset(
        prediction_times_df=df_prediction_times,
        timestamp_colname="timestamp",
        id_colname="dw_ek_borger",
    )

    dataset.add_outcome(
        outcome_df=df_event_times,
        lookahead_days=lookahead_days,
        resolve_multiple=resolve_multiple,
        fallback=fallback,
        values_colname=values_colname,
    )

    flatenned_vals_colname = f"val_within_{lookahead_days}_days"

    expected_flattened_vals = pd.DataFrame(
        {flatenned_vals_colname: expected_flattened_vals}
    )

    pd.testing.assert_series_equal(
        dataset.df[flatenned_vals_colname],
        expected_flattened_vals[flatenned_vals_colname],
    )
