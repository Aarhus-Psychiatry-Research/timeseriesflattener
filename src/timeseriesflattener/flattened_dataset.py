from typing import Callable, Dict, List, Union, Tuple
from pandas import DataFrame
from datetime import datetime


class FlattenedDataset:
    def __init__(
        self,
        prediction_times_df: DataFrame,
        id_col_name: str = "dw_ek_borger",
        timestamp_col_name: str = "timestamp",
    ):
        """Class containing a time-series, flattened. A 'flattened' version is a tabular representation for each prediction time.
        A prediction time is every timestamp where you want your model to issue a prediction.

        E.g if you have a prediction_times_df:

        id_col_name | timestamp_col_name
        1           | 2022-01-10
        1           | 2022-01-12
        1           | 2022-01-15

        And a time-series of blood-pressure values as an outcome:
        id_col_name | timestamp_col_name | blood_pressure_value
        1           | 2022-01-09         | 120
        1           | 2022-01-14         | 140

        Then you can "flatten" the outcome into a new df, with a row for each of your prediction times:
        id_col_name | timestamp_col_name | latest_blood_pressure_within_24h
        1           | 2022-01-10         | 120
        1           | 2022-01-12         | NA
        1           | 2022-01-15         | 140

        Args:
            prediction_times_df (DataFrame): Dataframe with prediction times, required cols: patient_id, .
            timestamp_col_name (str, optional): Column name name for timestamps. Is used across outcomes and predictors. Defaults to "timestamp".
            id_col_name (str, optional): Column namn name for patients ids. Is used across outcome and predictors. Defaults to "dw_ek_borger".
        """
        self.df_prediction_times = prediction_times_df
        self.timestamp_col_name = timestamp_col_name
        self.id_col_name = id_col_name

        self.df = self.df_prediction_times

    def add_outcome(
        self,
        outcome_df: DataFrame,
        lookahead_days: float,
        resolve_multiple: Callable,
        fallback: float,
        outcome_df_values_col_name: str = "val",
        new_col_name: str = None,
    ):
        """Adds an outcome-column to the dataset

        Args:
            outcome_df (DataFrame): Required columns: patient_id, timestamp, outcome_value.
            lookahead_days (float): How far ahead to look for an outcome in days. If none found, use fallback.
            resolve_multiple (Callable): How to handle multiple values within the lookahead window. Takes a a function that takes a list as an argument and returns a float.
            fallback (float): What to do if no value within the lookahead.
            outcome_df_values_col_name (str): Column name for the outcome values in outcome_df, e.g. whether a patient has t2d or not at the timestamp. Defaults to "val".
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
        """

        self._add_col_to_flattened_dataset(
            values_df=outcome_df,
            direction="ahead",
            interval_days=lookahead_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            new_col_name=new_col_name,
            source_values_col_name=outcome_df_values_col_name,
        )

    def add_predictor(
        self,
        predictor_df: DataFrame,
        lookbehind_days: float,
        resolve_multiple: str,
        fallback: float,
        source_values_col_name: str = "val",
        new_col_name: str = None,
    ):
        """Adds a column with predictor values to the flattened dataset (e.g. "average value of bloodsample within n days")

        Args:
            predictor_df (DataFrame): Required columns: patient_id, timestamp, outcome_value.
            lookbehind_days (float): How far behind to look for a predictor value in days. If none found, use fallback.
            resolve_multiple (Callable): How to handle multiple values within the lookbehind window. Takes a a function that takes a list as an argument and returns a float.
            fallback (List[str]): What to do if no value within the lookahead.
            source_values_col_name (str): Column name for the predictor values in predictor_df, e.g. the patient's most recent blood-sample value. Defaults to "val".
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
        """

        self._add_col_to_flattened_dataset(
            values_df=predictor_df,
            direction="behind",
            interval_days=lookbehind_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            new_col_name=new_col_name,
            source_values_col_name=source_values_col_name,
        )

    def _add_col_to_flattened_dataset(
        self,
        values_df: DataFrame,
        direction: str,
        interval_days: float,
        resolve_multiple: str,
        fallback: float,
        new_col_name: str,
        source_values_col_name: str = "val",
    ):
        """Adds a column to the dataset (either predictor or outcome depending on the value of "direction")

        Args:
            values_df (DataFrame): Required columns: patient_id, timestamp, outcome_value.
            direction (str): Whether to look "ahead" or "behind".
            interval_days (float): How far to look in direction.
            resolve_multiple (Callable): How to handle multiple values within the lookbehind window. Takes a a function that takes a list as an argument and returns a float.
            fallback (List[str]): What to do if no value within the lookahead.
            new_col_name (str): Name to use for new column. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
            source_values_col_name (str, optional): Column name of the values column in values_df. Defaults to "val".
        """

        values_dict = self._events_to_dict_by_patient(
            df=values_df,
            values_col_name=source_values_col_name,
        )

        new_col = self.df_prediction_times.apply(
            lambda row: self._flatten_events_for_prediction_time(
                direction=direction,
                prediction_timestamp=row[self.timestamp_col_name],
                val_dict=values_dict,
                interval_days=interval_days,
                id=row[self.id_col_name],
                resolve_multiple=resolve_multiple,
                fallback=fallback,
            ),
            axis=1,
        )

        if new_col_name is None:
            new_col_name = source_values_col_name

        self.df[f"{new_col_name}_within_{interval_days}_days"] = new_col

    def _events_to_dict_by_patient(
        self,
        df: DataFrame,
        values_col_name: str,
    ) -> Dict[str, List[Tuple[Union[datetime, float]]]]:
        """
        Generate a dict of events grouped by patient_id

        Args:
            df (DataFrame): Dataframe to come from
            values_col_name (str): Column name for event values

        Returns:
            Dict[str, List[Tuple[Union[datetime, float]]]]:
                                    {
                                        patientid1: [(timestamp11, val11), (timestamp12, val12)],
                                        patientid2: [(timestamp21, val21), (timestamp22, val22)]
                                    }
        """

        return (
            df.groupby(self.id_col_name)
            .apply(
                lambda row: tuple(
                    [
                        list(event)
                        for event in zip(
                            row[self.timestamp_col_name], row[values_col_name]
                        )
                    ]
                )
            )
            .to_dict()
        )

    def _get_events_within_n_days(
        self,
        direction: str,
        prediction_timestamp: datetime,
        val_dict: Dict[str, List[Tuple[Union[datetime, float]]]],
        interval_days: float,
        id: int,
    ) -> List:
        """Gets a list of values that are within interval_days in direction from prediction_timestamp for id.

        Args:
            direction (str): Whether to look ahead or behind.
            prediction_timestamp (timestamp):
            val_dict (Dict[str, List[Tuple[Union[datetime, float]]]]): A dict containing the timestamps and vals for the events.
                Shaped like {patient_id: [(timestamp1: val1), (timestamp2: val2)]}
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

    def _flatten_events_for_prediction_time(
        self,
        direction: str,
        prediction_timestamp: str,
        val_dict: Dict[str, List[Tuple[Union[datetime, float]]]],
        interval_days: float,
        resolve_multiple: Callable,
        fallback: list,
        id: int,
    ) -> float:
        """Takes a list of events and turns them into a single value for a prediction_time
        given a set of conditions.

        Args:
            direction (str): Whether to look ahead or behind from the prediction time.
            prediction_timestamp (str): The timestamp to anchor on.
            val_dict (Dict[str, List[Tuple[Union[datetime, float]]]]): A dict containing the timestamps and vals for the events.
                Shaped like {patient_id: [(timestamp1: val1), (timestamp2: val2)]}
            interval_days (float): How many days to look in direction for events.
            resolve_multiple (str): How to handle multiple events within interval_days.
            fallback (list): How to handle no events within interval_days.
            id (int): Which patient ID to flatten events for.

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
        is_in_interval = difference_in_days >= -interval_days and difference_in_days < 0
    else:
        raise ValueError("direction can only be 'ahead' or 'behind'")

    return is_in_interval
