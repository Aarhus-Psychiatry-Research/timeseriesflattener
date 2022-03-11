from datetime import datetime
from typing import Union
from pandas import DataFrame
from gen_test_dfs import *


def group_df_to_dict_of_list(df: DataFrame, grouping_col: str, date_col: str) -> dict:
    """Creates a dict with dw_ek_borger as keys and a list of dates as values

    Args:
        df (DataFrame): A python dataframe with grouping_var
        grouping_col (str): Variable to group on
        date_col (str): colname

    Returns:
        dict: {dw_ek_borger: [date1, date2, date3 ...]}
    """
    grouped_dict = df.groupby(grouping_col).apply(lambda x: list(x[date_col])).to_dict()

    return grouped_dict


def is_within_n_days_before(
    prediction_date: datetime, event_date: datetime, interval_days: int
):
    """Checks whether date2 is within interval_months of date1"""
    is_before_prediction_time = (prediction_date - event_date).days >= 0
    is_within_threshold_days = (prediction_date - event_date).days <= interval_days

    return is_before_prediction_time & is_within_threshold_days


def event_exists_within_n_days_before(
    id: int, prediction_datetime: datetime, event_dict: dict, interval_days: int
):
    """Checks whether any event in event_dict exists within X months before prediction_time for id

    Args:
        id (int): citizen_id
        prediction_datetime (datetime)
        event_dict (dict)
        interval_months (int)

    Returns:
        _type_: _description_
    """

    for event_date in event_dict[id]:
        if is_within_n_days_before(prediction_datetime, event_date, interval_days):
            return True
    return False


def add_col_event_within_months(
    df_prediction_datetimes: DataFrame,
    df_event_datetimes: DataFrame,
    interval_days: Union[float, int],
    event_str_for_colname: str,
    id_colname: str = "dw_ek_borger",
    prediction_datetime_colname: str = "datotid_start",
    event_datetime_colname: str = "datotid_event",
) -> DataFrame:
    """Adds a boolean column to the df `prediction_datetimes` signifying whether there is an event within
    `interval_days` before the prediction datetime

    Args:
        df_prediction_datetimes (DataFrame): dataframe with an id_col and a datetime col
        df_event_datetimes (DataFrame): dataframe with an id_col and a datetime col
        interval_days (Union[float, int]): how many days to look behind prediction time for events
        event_str_for_colname (str): string to name the new column
        id_colname (str, optional): Defaults to "dw_ek_borger".
        prediction_datetime_colname (str, optional): Defaults to "datotid_start".
        event_datetime_colname (str, optional): Defaults to "datotid_event".

    Returns:
        DataFrame: _description_
    """

    event_datetimes_dict = group_df_to_dict_of_list(
        df_event_datetimes, grouping_col=id_colname, date_col=event_datetime_colname
    )

    new_col = df_prediction_datetimes.apply(
        lambda row: event_exists_within_n_days_before(
            id=row[id_colname],
            prediction_datetime=row[prediction_datetime_colname],
            event_dict=event_datetimes_dict,
            interval_days=interval_days,
        ),
        axis=1,
    )

    out_df = df_prediction_datetimes
    out_df[f"{event_str_for_colname}_within_{interval_days}_days"] = new_col

    return out_df
