from typing import Callable, Dict, List, Union
from pandas import DataFrame
from datetime import date, datetime


class FlattenedDataset:
    def __init__(
        self,
        prediction_times_df: DataFrame,
        prediction_time_colname: str = "timestamp",
    ):
        self.df_prediction_times = prediction_times_df
        self.prediction_time_colname = prediction_time_colname

        self.df = self.df_prediction_times

    def add_outcome(
        self,
        outcome_df: DataFrame,
        lookahead_days: float,
        resolve_multiple: str,
        fallback: List[str],
        timestamp_colname: str = "timestamp",
        values_colname: str = "values",
        id_colname: str = "dw_ek_borger",
        new_col_name: str = None,
    ):
        """Adds an outcome-column to the dataset

        Args:
            outcome_df (DataFrame): Cols: dw_ek_borger, datotid, value if relevant.
            lookahead_days (float): How far ahead to look for an outcome in days. If none found, use fallback.
            resolve_multiple (str): What to do with more than one value within the lookahead.
                Suggestions: earliest, latest, mean, max, min.
            fallback (List[str]): What to do if no value within the lookahead.
                Suggestions: latest, mean_of_patient, mean_of_population, hardcode (qualified guess)
            timestamp_colname (str): Column name for timestamps
            values_colname (str): Colname for outcome values in outcome_df
            id_colname (str): Column name for citizen id
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
                Defaults to using values_colname.
        """

        outcome_dict = (
            outcome_df.groupby(id_colname)
            .apply(
                lambda row: [
                    list(x) for x in zip(row[timestamp_colname], row[values_colname])
                ]
            )
            .to_dict()
        )
        """
            Generate a dict of shape {  
                                        id1: [[timestamp11, val11], [timestamp12, val12]],
                                        id2: [[timestamp21, val21], [timestamp22, val22]]
                                    }
        """

        new_col = self.df_prediction_times.apply(
            lambda row: self._flatten_events(
                direction="ahead",
                prediction_timestamp=row[self.prediction_time_colname],
                val_dict=outcome_dict,
                interval_days=lookahead_days,
                id=row[id_colname],
                resolve_multiple=resolve_multiple,
                fallback=fallback,
            ),
            axis=1,
        )

        if new_col_name is None:
            new_col_name = values_colname

        self.df[f"{new_col_name}_within_{lookahead_days}_days"] = new_col

    def add_predictor(
        self,
        predictor_df: DataFrame,
        lookbehind_days: float,
        resolve_multiple: str,
        fallback: List[str],
        outcome_colname: str,
        id_colname: str = "dw_ek_borger",
        timestamp_colname: str = "timestamp",
    ):
        """Adds a predictor-column to the dataset

        Args:
            predictor_df (DataFrame): Cols: dw_ek_borger, datotid, value if relevant.
            lookahead_days (float): How far ahead to look for an outcome in days. If none found, use fallback.
            resolve_multiple (str): What to do with more than one value within the lookahead.
                Suggestions: earliest, latest, mean, max, min.
            fallback (List[str]): What to do if no value within the lookahead.
                Suggestions: latest, mean_of_patient, mean_of_population, hardcode (qualified guess)
            outcome_colname (str): What to name the column
            id_colname (str): Column name for citizen id
            timestamp_colname (str): Column name for timestamps
        """

        raise NotImplementedError

    def _get_events_within_n_days(
        self,
        direction: str,
        prediction_timestamp: datetime,
        val_dict: Dict[str, List[List]],
        interval_days: float,
        id: int,
    ) -> List:
        """Gets a list of values that are within interval_days in direction from predictin_timestamp for id.

        Args:
            direction (str): Whether to look ahead or behind.
            prediction_timestamp (timestamp):
            val_dict (Dict[str, List[Dict[datetime, int]]]): A dict containing the timestamps and vals for the events.
                Shaped like {patient_id: [[timestamp1: val1], [timestamp2: val2]]}
            interval_days (int): How far to look in direction.
            id (int): Patient id

        Returns:
            list: [datetime, value]
        """

        events_within_n_days = []

        for event in val_dict[id]:
            event_timestamp = event[0]

            if is_within_n_days(
                direction=direction,
                prediction_timestamp=prediction_timestamp,
                event_timestamp=event_timestamp,
                interval_days=interval_days,
            ):
                events_within_n_days.append(event)

        return events_within_n_days

    def _flatten_events(
        self,
        direction: str,
        prediction_timestamp: str,
        val_dict: Dict[str, List[List]],
        interval_days: float,
        resolve_multiple: Callable,
        fallback: list,
        id: int = "dw_ek_borger",
    ) -> float:
        """Takes a list of events and turns them into a single value for a prediction_time
        given a set of conditions.

        Args:
            direction (str): Whether to look ahead or behind from the prediction time.
            prediction_timestamp (str): The timestamp to anchor on.
            val_dict (Dict[str, List[List[datetime, int]]]): A dict containing the timestamps and vals for the events.
                Shaped like {patient_id: [[timestamp1: val1], [timestamp2: val2]]}
            interval_days (float): How many days to look in direction for events.
            resolve_multiple (str): How to handle multiple events within interval_days.
            fallback (list): How to handle no events within interval_days.
            id (int, optional): Column name that identifies unique patients. Defaults to "dw_ek_borger".

        Returns:
            float: Value for each prediction_time.
        """
        events = self._get_events_within_n_days(
            direction=direction,
            prediction_timestamp=prediction_timestamp,
            val_dict=val_dict,
            interval_days=interval_days,
            id=id,
        )

        if len(events) == 0:
            return fallback
        elif len(events) == 1:
            event_val = events[0][1]
            return event_val
        elif len(events) > 1:
            return resolve_multiple(events)


def is_within_n_days(
    direction: str,
    prediction_timestamp: datetime,
    event_timestamp: datetime,
    interval_days: float,
) -> bool:
    """Looks interval_days in direction from prediction_timestamp.
    Returns true if event_timestamp is within interval_days.

    Args:
        direction: Whether to look ahead or behind
        prediction_timestamp (timestamp): timestamp for prediction
        event_timestamp (timestamp): timestamp for event
        interval_days (int): How far to look in direction

    Returns:
        boolean
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
