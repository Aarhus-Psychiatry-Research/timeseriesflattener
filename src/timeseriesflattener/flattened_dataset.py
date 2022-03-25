from typing import Callable, Dict, List
from pandas import DataFrame
from datetime import datetime


class FlattenedDataset:
    def __init__(
        self,
        prediction_times_df: DataFrame,
        prediction_time_colname: str = "datotid_start",
    ):
        self.prediction_times_df = prediction_times_df
        self.prediction_time_colname = prediction_time_colname

        self.df = self.prediction_times_df

    def add_outcome(
        self,
        outcome_df: DataFrame,
        lookahead_days: float,
        resolve_multiple: str,
        fallback: List[str],
        outcome_colname: str,
        id_colname: str = "dw_ek_borger",
        timestamp_colname: str = "datotid_start",
    ):
        """Adds an outcome-column to the dataset

        Args:
            outcome_df (DataFrame): Cols: dw_ek_borger, datotid, (value if relevant).
            lookahead_days (float): How far ahead to look for an outcome in days. If none found, use fallback.
            resolve_multiple (str): How to handle more than one record within the lookbehind.
                Suggestions: earliest, latest, mean_of_records, max, min.
            fallback (List[str]): How to handle lack of a record within the lookbehind.
                Suggestions: latest, mean_of_patient, mean_of_population, hardcode (qualified guess)
            outcome_colname (str): What to name the column
            id_colname (str): Column name for citizen id
            timestamp_colname (str): Column name for timestamps
        """

        outcome_dict = (
            outcome_df.groupby(id_colname)
            .apply(
                lambda x: [
                    list(x) for x in zip(x[timestamp_colname], x[outcome_colname])
                ]
            )
            .to_dict()
        )

        new_col = self.prediction_times_df.apply(
            lambda row: flatten_events(
                direction="ahead",
                prediction_timestamp=row[self.prediction_time_colname],
                event_dict=outcome_dict,
                interval_days=lookahead_days,
                id=row[id_colname],
                resolve_multiple=resolve_multiple,
                fallback=fallback,
            ),
            axis=1,
        )

        self.df[f"{outcome_colname}_within_{lookahead_days}_days"] = new_col

    def add_predictor(
        self,
        outcome_df: DataFrame,
        lookahead_window: float,
        resolve_multiple: str,
        fallback: List[str],
        name: str,
    ):
        """Adds a predictor-column to the dataset

        Args:
            predictor (DataFrame): Cols: dw_ek_borger, datotid, (value if relevant).
            lookback_window (float): How far back to look for a predictor. If none found, use fallback.
            resolve_multiple (str): How to handle more than one record within the lookbehind.
                Suggestions: earliest, latest, mean_of_records, max, min.
            fallback (List[str]): How to handle lack of a record within the lookbehind.
                Suggestions: latest, mean_of_patient, mean_of_population, hardcode (qualified guess)
            name (str): What to name the column

        Raises:
            NotImplementedError: _description_
        """

        return True


def is_within_n_days(
    direction: str,
    prediction_timestamp: datetime,
    event_timestamp: datetime,
    interval_days: int,
):
    """Checks whether prediction_date is within interval_days of event_date.
    Look ahead from prediction_date: interval_days must be negative

    Args:
        prediction_date (datetime): _description_
        event_date (datetime): _description_
        interval_days (int): _description_
        direction: Whether to look ahead or behind

    Returns:
        _type_: _description_
    """

    difference_in_days = (
        event_timestamp - prediction_timestamp
    ).total_seconds() / 86400
    # Use .seconds instead of .days to get fractions of a day

    if direction == "ahead":
        is_in_interval = difference_in_days <= interval_days and difference_in_days > 0
    elif direction == "behind":
        is_in_interval = difference_in_days >= interval_days and difference_in_days < 0
    else:
        return ValueError("direction can only be 'ahead' or 'behind'")

    return is_in_interval


def get_events_within_n_days(
    direction: str,
    prediction_timestamp: datetime,
    event_dict: dict,
    interval_days: int,
    id: int = "dw_ek_borger",
):
    """Checks whether any event in event_dict is dated within lookahead_days after prediction_time for id
    Args:
        id (int): citizen_id
        prediction_datetime (datetime)
        event_dict (dict)
        interval_months (int)
    Returns:
        _type_: _description_
    """
    events_within_n_days = []

    for event in event_dict[id]:
        event_timestamp = event[0]

        if is_within_n_days(
            direction=direction,
            prediction_timestamp=prediction_timestamp,
            event_timestamp=event_timestamp,
            interval_days=interval_days,
        ):
            events_within_n_days.append(event)

    return events_within_n_days


def flatten_events(
    direction: str,
    prediction_timestamp: str,
    event_dict: Dict[str, List[List]],
    interval_days: int,
    resolve_multiple: Callable,
    fallback: list,
    id: int = "dw_ek_borger",
):
    """Takes a list of events and turns them into a single value for a prediction_time
    given a set of conditions.

    Args:
        direction (str): Whether to look ahead or behind from the prediction time.
        prediction_datetime (str): The datetime to anchor on.
        event_dict (Dict[str, List[Dict[Timestamp, int]]]): A dict containing the timestamps and vals for the events.
            Shaped like {patient_id: [[timestamp1: val1], [timestamp2: val2]]}
        interval_days (int): How many days to look in direction for events.
        resolve_multiple (str): How to handle multiple events within interval_days.
        fallback (list): How to handle no events within interval_days.
        id (int, optional): Column name that identifies unique patients. Defaults to "dw_ek_borger".

    Returns:
        int: Value for each prediction_time.
    """
    events = get_events_within_n_days(
        direction=direction,
        prediction_timestamp=prediction_timestamp,
        event_dict=event_dict,
        interval_days=interval_days,
        id=id,
    )

    if len(events) == 0:
        if type(fallback) == int:
            return fallback
    elif len(events) == 1:
        event_val = events[0][1]
        return event_val
    elif len(events) > 1:
        return resolve_multiple(events)
