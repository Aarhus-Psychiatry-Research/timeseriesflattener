from typing import List
from pandas import DataFrame


class FlattenedDataset:
    def __init__(self, prediction_times: DataFrame):
        raise NotImplementedError
        self.prediction_times = prediction_times

    def add_outcome(
        outcome_df: DataFrame,
        lookahead_window: float,
        resolve_multiple: str,
        fallback: List[str],
        name: str,
    ):
        """Adds an outcome-column to the dataset

        Args:
            outcome_df (DataFrame): Cols: dw_ek_borger, datotid, (value if relevant).
            lookahead_window (float): How far ahead to look for an outcome. If none found, use fallback.
            resolve_multiple (str): How to handle more than one record within the lookbehind. Suggestions: earliest, latest, mean_of_records, max, min.
            fallback (List[str]): How to handle lack of a record within the lookbehind. Suggestions: latest, mean_of_patient, mean_of_population, hardcode (qualified guess)
            name (str): What to name the column

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def add_predictor(
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
            resolve_multiple (str): How to handle more than one record within the lookbehind. Suggestions: earliest, latest, mean_of_records, max, min.
            fallback (List[str]): How to handle lack of a record within the lookbehind. Suggestions: latest, mean_of_patient, mean_of_population, hardcode (qualified guess)
            name (str): What to name the column

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
