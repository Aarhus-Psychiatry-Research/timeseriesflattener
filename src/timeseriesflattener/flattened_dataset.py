from typing import Callable, Dict, List, Optional, Tuple, Union

import catalogue
import pandas as pd
from pandas import DataFrame

resolve_fns = catalogue.create("timeseriesflattener", "resolve_strategies")

from timeseriesflattener.resolve_multiple_functions import *


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
        self.timestamp_col_name = timestamp_col_name
        self.id_col_name = id_col_name

        self.pred_time_uuid_colname = "prediction_time_uuid"
        self.pred_times_with_uuid = prediction_times_df

        for col_name in [self.timestamp_col_name, self.id_col_name]:
            if col_name not in self.pred_times_with_uuid.columns:
                raise ValueError(
                    f"{col_name} does not exist in df_prediction_times, change the df or set another argument"
                )

        self.pred_times_with_uuid[
            self.pred_time_uuid_colname
        ] = self.pred_times_with_uuid[self.id_col_name].astype(
            str
        ) + self.pred_times_with_uuid[
            self.timestamp_col_name
        ].dt.strftime(
            "-%Y-%m-%d-%H-%M-%S"
        )

        # Having a df_aggregating separate from df allows to only generate the UUID once, while not presenting it
        # in self.df
        self.df_aggregating = self.pred_times_with_uuid
        self.df = self.pred_times_with_uuid

    def add_predictors_from_list_of_argument_dictionaries(
        self,
        predictor_list: List[Dict[str, str]],
        predictor_dfs: Dict[str, DataFrame],
        resolve_multiple_fn_dict: Optional[Dict[str, Callable]] = None,
    ):
        """Add predictors to the flattened dataframe from a list

        Args:
            predictor_list (List[Dict[str, str]]): A list of dictionaries describing the prediction_features you'd like to generate.
            predictor_dfs (Dict[str, DataFrame]): A dictionary mapping the predictor_df in predictor_list to DataFrame objects
            resolve_multiple_fn_dict (Union[str, Callable], optional): If wanting to use manually defined resolve_multiple strategies (i.e. ones that aren't in resolve_fns), requires a dictionary mapping the resolve_multiple string to a Callable object

        Example:
            >>> predictor_list = [
            >>>     {
            >>>         "predictor_df": "df_name",
            >>>         "lookbehind_days": 1,
            >>>         "resolve_multiple": "resolve_multiple_strat_name",
            >>>         "fallback": 0,
            >>>         "source_values_col_name": "val",
            >>>     },
            >>>     {
            >>>         "predictor_df": "df_name",
            >>>         "lookbehind_days": 1,
            >>>         "resolve_multiple": "min",
            >>>         "fallback": 0,
            >>>         "source_values_col_name": "val",
            >>>     }
            >>> ]
            >>> predictor_dfs = {"df_name": df_object}
            >>> resolve_multiple_strategies = {"resolve_multiple_strat_name": resolve_multiple_func}
            >>>
            >>> dataset.add_predictors_from_list(
            >>>     predictor_list=predictor_list,
            >>>     predictor_dfs=predictor_dfs,
            >>>     resolve_multiple_fn_dict=resolve_multiple_strategies,
            >>> )
        """

        for arg_list in predictor_list:
            arg_list["predictor_df"] = predictor_dfs[arg_list["predictor_df"]]
            if (
                resolve_multiple_fn_dict is not None
                and arg_list["resolve_multiple"] in resolve_multiple_fn_dict
            ):
                arg_list["resolve_multiple"] = resolve_multiple_fn_dict[
                    arg_list["resolve_multiple"]
                ]

            self.add_predictor(**arg_list)

    def add_outcome(
        self,
        outcome_df: DataFrame,
        lookahead_days: float,
        resolve_multiple: Union[Callable, str],
        fallback: float,
        outcome_df_values_col_name: str = "val",
        new_col_name: str = None,
    ):
        """Adds an outcome-column to the dataset

        Args:
            outcome_df (DataFrame): A table in wide format. Required columns: patient_id, timestamp, value.
            lookahead_days (float): How far ahead to look for an outcome in days. If none found, use fallback.
            resolve_multiple (Callable, str): How to handle multiple values within the lookahead window. Takes either i) a function that takes a list as an argument and returns a float, or ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (float): What to do if no value within the lookahead.
            outcome_df_values_col_name (str): Column name for the outcome values in outcome_df, e.g. whether a patient has t2d or not at the timestamp. Defaults to "val".
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
        """

        self.add_col_to_flattened_dataset(
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
        resolve_multiple: Union[Callable, str],
        fallback: float,
        source_values_col_name: str = "val",
        new_col_name: str = None,
    ):
        """Adds a column with predictor values to the flattened dataset (e.g. "average value of bloodsample within n days")

        Args:
            predictor_df (DataFrame): A table in wide format. Required columns: patient_id, timestamp, value.
            lookbehind_days (float): How far behind to look for a predictor value in days. If none found, use fallback.
            resolve_multiple (Callable, str): How to handle multiple values within the lookbehind window. Takes either i) a function that takes a list as an argument and returns a float, or ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (List[str]): What to do if no value within the lookahead.
            source_values_col_name (str): Column name for the predictor values in predictor_df, e.g. the patient's most recent blood-sample value. Defaults to "val".
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
        """

        self.add_col_to_flattened_dataset(
            values_df=predictor_df,
            direction="behind",
            interval_days=lookbehind_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            new_col_name=new_col_name,
            source_values_col_name=source_values_col_name,
        )

    def add_col_to_flattened_dataset(
        self,
        values_df: DataFrame,
        direction: str,
        interval_days: float,
        resolve_multiple: Union[Callable, str],
        fallback: float,
        new_col_name: str,
        source_values_col_name: str = "val",
    ):
        """Adds a column to the dataset (either predictor or outcome depending on the value of "direction")

        Args:
            values_df (DataFrame): A table in wide format. Required columns: patient_id, timestamp, value.
            direction (str): Whether to look "ahead" or "behind".
            interval_days (float): How far to look in direction.
            resolve_multiple (Callable, str): How to handle multiple values within interval_days. Takes either i) a function that takes a list as an argument and returns a float, or ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (List[str]): What to do if no value within the lookahead.
            new_col_name (str): Name to use for new column. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
            source_values_col_name (str, optional): Column name of the values column in values_df. Defaults to "val".
        """

        for col_name in [self.timestamp_col_name, self.id_col_name]:
            if col_name not in values_df.columns:
                raise ValueError(
                    f"{col_name} does not exist in df_prediction_times, change the df or set another argument"
                )

        # Generate df with one row for each prediction time x event time combination
        df = pd.merge(
            self.pred_times_with_uuid,
            values_df,
            how="left",
            on=self.id_col_name,
            suffixes=("_pred", "_val"),
        ).drop("dw_ek_borger", axis=1)

        # Drop prediction times without event times within interval days
        df = self.drop_records_outside_interval_days(
            df,
            direction=direction,
            interval_days=interval_days,
            timestamp_pred_colname="timestamp_pred",
            timestamp_val_colname="timestamp_val",
        )

        df = self.add_back_prediction_times_without_value(df).fillna(fallback)
        df = self.resolve_multiple_values_within_interval_days(resolve_multiple, df)

        # Rename column
        if new_col_name is None:
            new_col_name = source_values_col_name

        full_col_str = f"{new_col_name}_within_{interval_days}_days"
        df.rename({"val": full_col_str}, axis=1, inplace=True)

        # Assign to instance
        self.df_aggregating = pd.merge(
            self.df_aggregating,
            df[[self.pred_time_uuid_colname, full_col_str]],
            how="left",
            on=self.pred_time_uuid_colname,
            suffixes=("", ""),
        )

        self.df = self.df_aggregating.drop(self.pred_time_uuid_colname, axis=1)

    def add_back_prediction_times_without_value(self, df: DataFrame) -> DataFrame:
        """Ensures all prediction times are represented in the returned dataframe.

        Args:
            df (DataFrame):

        Returns:
            DataFrame:
        """
        return pd.merge(
            self.pred_times_with_uuid,
            df,
            how="left",
            on=self.pred_time_uuid_colname,
            suffixes=("", ""),
        ).drop(["timestamp_pred", "timestamp_val"], axis=1)

    def resolve_multiple_values_within_interval_days(
        self, resolve_multiple: Callable, df: DataFrame
    ) -> DataFrame:
        """Apply the resolve_multiple function to prediction_times where there are multiple values within the interval_days lookahead

        Args:
            resolve_multiple (Callable): Takes a grouped df and collapses each group to one record (e.g. sum, count etc.).
            df (DataFrame): Source dataframe with all prediction time x val combinations.

        Returns:
            DataFrame: DataFrame with one row pr. prediction time.
        """
        # Sort by timestamp_pred in case resolve_multiple needs dates
        df = df.sort_values(by=self.timestamp_col_name).groupby(
            self.pred_time_uuid_colname
        )

        if isinstance(resolve_multiple, Callable):
            df = resolve_multiple(df).reset_index()
        else:
            resolve_strategy = resolve_fns.get(resolve_multiple)
            df = resolve_strategy(df).reset_index()

        return df

    def drop_records_outside_interval_days(
        self,
        df: DataFrame,
        direction: str,
        interval_days: float,
        timestamp_pred_colname: str,
        timestamp_val_colname: str,
    ) -> DataFrame:
        """Keeps only rows where timestamp_val is within interval_days in direction of timestamp_pred.

        Args:
            direction (str): Whether to look ahead or behind.
            interval_days (float): How far to look
            df (DataFrame): Source dataframe
            timestamp_pred_colname (str):
            timestamp_val_colname (str):

        Raises:
            ValueError: If direction is niether ahead nor behind.

        Returns:
            DataFrame
        """

        df["time_from_pred_to_val_in_days"] = (
            df[timestamp_val_colname] - df[timestamp_pred_colname]
        ).dt.total_seconds() / 86_400
        # Divide by 86.400 seconds/day

        if direction == "ahead":
            df["is_in_interval"] = (
                df["time_from_pred_to_val_in_days"] <= interval_days
            ) & (df["time_from_pred_to_val_in_days"] > 0)
        elif direction == "behind":
            df["is_in_interval"] = (
                df["time_from_pred_to_val_in_days"] >= -interval_days
            ) & (df["time_from_pred_to_val_in_days"] < 0)
        else:
            raise ValueError("direction can only be 'ahead' or 'behind'")

        return df[df["is_in_interval"] == True].drop(
            ["is_in_interval", "time_from_pred_to_val_in_days"], axis=1
        )
