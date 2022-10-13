"""Takes a time-series and flattens it into a set of prediction times with
describing values."""
import datetime as dt
import os
from collections.abc import Callable
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from catalogue import Registry  # noqa # pylint: disable=unused-import
from dask.diagnostics import ProgressBar
from pandas import DataFrame
from tqdm.dask import TqdmCallback
from wasabi import Printer, msg

from psycop_feature_generation.timeseriesflattener.resolve_multiple_functions import (
    resolve_fns,
)
from psycop_feature_generation.utils import (
    data_loaders,
    df_contains_duplicates,
    generate_feature_colname,
    load_dataset_from_file,
    write_df_to_file,
)

ProgressBar().register()


def select_and_assert_keys(dictionary: dict, key_list: list[str]) -> dict:
    """Keep only the keys in the dictionary that are in key_order, and orders
    them as in the lsit.

    Args:
        dictionary (dict): dictionary to process
        key_list (list[str]): list of keys to keep

    Returns:
        dict: dict with only the selected keys
    """
    for key in key_list:
        if key not in dictionary:
            raise KeyError(f"{key} not in dict")

    return {key: dictionary[key] for key in key_list if key in dictionary}


class FlattenedDataset:  # pylint: disable=too-many-instance-attributes
    """Turn a set of time-series into tabular prediction-time data."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        prediction_times_df: DataFrame,
        id_col_name: str = "dw_ek_borger",
        timestamp_col_name: str = "timestamp",
        min_date: Optional[pd.Timestamp] = None,
        n_workers: int = 60,
        predictor_col_name_prefix: str = "pred",
        outcome_col_name_prefix: str = "outc",
        feature_cache_dir: Optional[Path] = None,
    ):
        """Class containing a time-series, flattened.

        A 'flattened' version is a tabular representation for each prediction time.
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
            min_date (Optional[pd.Timestamp], optional): Drop all prediction times before this date. Defaults to None.
            id_col_name (str, optional): Column namn name for patients ids. Is used across outcome and predictors. Defaults to "dw_ek_borger".
            predictor_col_name_prefix (str, optional): Prefix for predictor col names. Defaults to "pred_".
            outcome_col_name_prefix (str, optional): Prefix for outcome col names. Defaults to "outc_".
            n_workers (int): Number of subprocesses to spawn for parallellisation. Defaults to 60.
            feature_cache_dir (Path): Path to cache directory for feature dataframes. Defaults to None.

        Raises:
            ValueError: If timestamp_col_name or id_col_name is not in prediction_times_df
            ValueError: If timestamp_col_name is not and could not be converted to datetime
        """
        self.n_workers = n_workers

        self.timestamp_col_name = timestamp_col_name
        self.id_col_name = id_col_name
        self.pred_time_uuid_col_name = "prediction_time_uuid"
        self.predictor_col_name_prefix = predictor_col_name_prefix
        self.outcome_col_name_prefix = outcome_col_name_prefix
        self.min_date = min_date

        if feature_cache_dir:
            self.feature_cache_dir = feature_cache_dir
            if not self.feature_cache_dir.exists():
                self.feature_cache_dir.mkdir()

        self.n_uuids = len(prediction_times_df)

        self.msg = Printer(timestamp=True)

        if "value" in prediction_times_df.columns:
            prediction_times_df.drop("value", axis=1, inplace=True)

        self.df = prediction_times_df

        self.check_init_df_for_errors()

        # Drop prediction times before min_date
        if min_date is not None:
            self.df = self.df[self.df[self.timestamp_col_name] > self.min_date]

        # Create pred_time_uuid_columne
        self.df[self.pred_time_uuid_col_name] = self.df[self.id_col_name].astype(
            str,
        ) + self.df[self.timestamp_col_name].dt.strftime("-%Y-%m-%d-%H-%M-%S")

        self.loaders_catalogue = data_loaders

    def check_init_df_for_errors(self):
        """Run checks on the initial dataframe."""

        # Check that colnames are present
        self.check_that_timestamp_and_id_columns_exist()
        self.check_for_duplicate_rows()
        self.check_timestamp_col_type()

    def check_timestamp_col_type(self):
        """Check that the timestamp column is of type datetime."""
        timestamp_col_type = type(self.df[self.timestamp_col_name][0]).__name__

        if timestamp_col_type not in ["Timestamp"]:
            try:
                self.df[self.timestamp_col_name] = pd.to_datetime(
                    self.df[self.timestamp_col_name],
                )
            except Exception as exc:
                raise ValueError(
                    f"prediction_times_df: {self.timestamp_col_name} is of type {timestamp_col_type}, and could not be converted to 'Timestamp' from Pandas. Will cause problems. Convert before initialising FlattenedDataset. More info: {exc}",
                ) from exc

    def check_for_duplicate_rows(self):
        """Check that there are no duplicate rows in the initial dataframe."""
        if df_contains_duplicates(
            df=self.df,
            col_subset=[self.id_col_name, self.timestamp_col_name],
        ):
            raise ValueError(
                "Duplicate patient/timestamp combinations in prediction_times_df, aborting",
            )

    def check_that_timestamp_and_id_columns_exist(self):
        """Check that the required columns are present in the initial
        dataframe."""

        for col_name in [self.timestamp_col_name, self.id_col_name]:
            if col_name not in self.df.columns:
                raise ValueError(
                    f"{col_name} does not exist in prediction_times_df, change the df or set another argument",
                )

    def _validate_processed_arg_dicts(self, arg_dicts: list):
        warnings = []

        for d in arg_dicts:
            if not isinstance(d["values_df"], (DataFrame, Callable)):
                warnings.append(
                    f"values_df resolves to neither a Callable nor a DataFrame in {d}",
                )

            if not (d["direction"] == "ahead" or d["direction"] == "behind"):
                warnings.append(f"direction is neither ahead or behind in {d}")

            if not isinstance(d["interval_days"], (int, float)):
                warnings.append(f"interval_days is neither an int nor a float in {d}")

        if len(warnings) != 0:
            raise ValueError(
                f"Didn't generate any features because: {warnings}",
            )

    def _load_most_recent_df_matching_pattern(
        self,
        dir_path: Path,
        file_pattern: str,
        file_suffix: str,
    ) -> DataFrame:
        """Load most recent df matching pattern.

        Args:
            dir (Path): Directory to search
            file_pattern (str): Pattern to match
            file_suffix (str, optional): File suffix to match.

        Returns:
            DataFrame: DataFrame matching pattern

        Raises:
            FileNotFoundError: If no file matching pattern is found
        """
        files_with_suffix = list(dir_path.glob(f"*{file_pattern}*.{file_suffix}"))

        if len(files_with_suffix) == 0:
            raise FileNotFoundError(f"No files matching pattern {file_pattern} found")

        path_of_most_recent_file = max(files_with_suffix, key=os.path.getctime)

        return load_dataset_from_file(
            file_path=path_of_most_recent_file,
        )

    def _load_cached_df_and_expand_fallback(
        self,
        file_pattern: str,
        file_suffix: str,
        fallback: Any,
        full_col_str: str,
    ) -> pd.DataFrame:
        """Load most recent df matching pattern, and expand fallback column.

        Args:
            file_pattern (str): File pattern to search for
            file_suffix (str): File suffix to search for
            fallback (Any): Fallback value
            prediction_times_with_uuid_df (pd.DataFrame): Prediction times with uuids
            full_col_str (str): Full column name for values

        Returns:
            DataFrame: DataFrame with fallback column expanded
        """
        df = self._load_most_recent_df_matching_pattern(
            dir_path=self.feature_cache_dir,
            file_pattern=file_pattern,
            file_suffix=file_suffix,
        )

        # Expand fallback column
        df = pd.merge(
            left=self.df[self.pred_time_uuid_col_name],
            right=df,
            how="left",
            on=self.pred_time_uuid_col_name,
            validate="m:1",
        )

        df[full_col_str] = df[full_col_str].fillna(fallback)

        return df

    def _cache_is_hit(
        self,
        kwargs_dict: dict,
        file_pattern: str,
        full_col_str: str,
        file_suffix: str,
    ) -> bool:
        """Check if cache is hit.

        Args:
            kwargs_dict (dict): dictionary of kwargs
            file_pattern (str): File pattern to match. Looks for *file_pattern* in cache dir.
            e.g. "*feature_name*_uuids*.file_suffix"
            full_col_str (str): Full column string. e.g. "feature_name_ahead_interval_days_resolve_multiple_fallback"
            file_suffix (str): File suffix to match. e.g. "csv"

        Returns:
            bool: True if cache is hit, False otherwise
        """
        # Check that file exists
        file_pattern_hits = list(
            self.feature_cache_dir.glob(f"*{file_pattern}*.{file_suffix}"),
        )

        if len(file_pattern_hits) == 0:
            self.msg.info(f"Cache miss, {file_pattern} didn't exist")
            return False

        # Check that file contents match expected
        cache_df = self._load_most_recent_df_matching_pattern(
            dir_path=self.feature_cache_dir,
            file_pattern=file_pattern,
            file_suffix=file_suffix,
        )

        generated_df = pd.DataFrame({full_col_str: []})

        # Check that some values in generated_df differ from fallback
        # Otherwise, comparison to cache is meaningless
        n_to_generate = 1_000

        while not any(
            generated_df[full_col_str] != kwargs_dict["fallback"],
        ):
            self.msg.info(
                f"{full_col_str}: Generated_df was all fallback values, regenerating",
            )

            generated_df = self.flatten_temporal_values_to_df(
                prediction_times_with_uuid_df=self.df.sample(n_to_generate),
                id_col_name=self.id_col_name,
                timestamp_col_name=self.timestamp_col_name,
                pred_time_uuid_col_name=self.pred_time_uuid_col_name,
                **kwargs_dict,
            )

            n_to_generate = (
                n_to_generate**1.5
            )  # Increase n_to_generate by 1.5x each time to increase chance of non_fallback values

        cached_suffix = "_c"
        generated_suffix = "_g"

        # Merge cache_df onto generated_df
        merged_df = pd.merge(
            left=generated_df,
            right=cache_df,
            how="left",
            on=self.pred_time_uuid_col_name,
            suffixes=(generated_suffix, cached_suffix),
            validate="1:1",
            indicator=True,
        )

        # Check that all rows in generated_df are in cache_df
        if not merged_df[full_col_str + generated_suffix].equals(
            merged_df[full_col_str + cached_suffix],
        ):
            self.msg.info(f"Cache miss, computed values didn't match {file_pattern}")
            return False

        # If all checks passed, return true
        msg.good(f"Cache hit for {full_col_str}")
        return True

    def _get_feature(
        self,
        kwargs_dict: dict,
        file_suffix: str = "parquet",
    ) -> DataFrame:
        """Get features. Either load from cache, or generate if necessary.

        Args:
            kwargs_dict (dict): dictionary of kwargs
            file_suffix (str, optional): File suffix for the cache lookup. Defaults to "parquet".

        Returns:
            DataFrame: DataFrame generates with create_flattened_df
        """
        full_col_str = generate_feature_colname(
            prefix=kwargs_dict["new_col_name_prefix"],
            out_col_name=kwargs_dict["new_col_name"],
            interval_days=kwargs_dict["interval_days"],
            resolve_multiple=kwargs_dict["resolve_multiple"],
            fallback=kwargs_dict["fallback"],
        )

        file_pattern = f"{full_col_str}_{self.n_uuids}_uuids"

        if hasattr(self, "feature_cache_dir"):

            if self._cache_is_hit(
                file_pattern=file_pattern,
                full_col_str=full_col_str,
                kwargs_dict=kwargs_dict,
                file_suffix="parquet",
            ):

                df = self._load_cached_df_and_expand_fallback(
                    file_pattern=file_pattern,
                    full_col_str=full_col_str,
                    fallback=kwargs_dict["fallback"],
                    file_suffix=file_suffix,
                )

                return df
        else:
            msg.info("No cache dir specified, not attempting load")

        df = self.flatten_temporal_values_to_df(
            prediction_times_with_uuid_df=self.df[
                [
                    self.pred_time_uuid_col_name,
                    self.id_col_name,
                    self.timestamp_col_name,
                ]
            ],
            id_col_name=self.id_col_name,
            timestamp_col_name=self.timestamp_col_name,
            pred_time_uuid_col_name=self.pred_time_uuid_col_name,
            **kwargs_dict,
        )

        # Write df to cache if exists
        if hasattr(self, "feature_cache_dir"):
            cache_df = df[[self.pred_time_uuid_col_name, full_col_str]]

            # Drop rows containing fallback, since it's non-informative
            cache_df = cache_df[cache_df[full_col_str] != kwargs_dict["fallback"]]

            # Write df to cache
            timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Write df to cache
            write_df_to_file(
                df=cache_df,
                file_path=self.feature_cache_dir
                / f"{file_pattern}_{timestamp}.parquet",
            )

        return df

    def _check_and_prune_to_required_arg_dict_keys(self, processed_arg_dicts, arg_dict):
        """Check if arg_dict has all required keys, and prune to only required
        keys.

        Args:
            processed_arg_dicts (list[dict[str, str]]): list of processed arg dicts.
            arg_dict (dict[str, str]): Arg dict to check and prune.

        Returns:
            dict[str, str]: Pruned arg dict.
        """

        required_keys = [
            "values_df",
            "direction",
            "interval_days",
            "resolve_multiple",
            "fallback",
            "new_col_name",
            "new_col_name_prefix",
        ]

        if "loader_kwargs" in arg_dict:
            required_keys.append("loader_kwargs")

        processed_arg_dicts.append(
            select_and_assert_keys(dictionary=arg_dict, key_list=required_keys),
        )

    def _resolve_predictor_df_str_to_df(
        self,
        predictor_dfs,
        dicts_found_in_predictor_dfs,
        arg_dict,
    ):
        """Resolve predictor_df to either a dataframe from predictor_dfs_dict
        or a callable from the registry.

        Args:
            predictor_dfs (dict[str, DataFrame]): A dictionary mapping predictor_df strings to dataframes.
            dicts_found_in_predictor_dfs (list[str]): A list of predictor_df strings that have already been resolved to dataframes.
            arg_dict (dict[str, str]): A dictionary describing the prediction_features you'd like to generate.

        Returns:
            dict[str, str]: A dictionary describing the prediction_features you'd like to generate.
        """

        loader_fns = self.loaders_catalogue.get_all()

        try:
            if predictor_dfs is not None:
                if arg_dict["values_df"] in predictor_dfs:
                    if arg_dict["values_df"] not in dicts_found_in_predictor_dfs:
                        dicts_found_in_predictor_dfs.append(arg_dict["values_df"])
                        msg.info(f"Found {arg_dict['values_df']} in predictor_dfs")

                    arg_dict["values_df"] = predictor_dfs[arg_dict["values_df"]].copy()
                else:
                    arg_dict["values_df"] = loader_fns[arg_dict["values_df"]]
            elif predictor_dfs is None:
                arg_dict["values_df"] = loader_fns[arg_dict["values_df"]]
        except LookupError:
            # Error handling in _validate_processed_arg_dicts
            # to handle in bulk
            pass

        return arg_dict

    def _check_if_resolve_multiple_can_convert_to_callable(
        self,
        resolve_multiple_fns,
        arg_dict,
    ):
        """Check if resolve_multiple is a string, if so, see if possible to
        resolve to a Callable.

        Args:
            resolve_multiple_fns (dict[str, Callable]): A dictionary mapping the resolve_multiple string to a Callable object.
            arg_dict (dict[str, str]): A dictionary describing the prediction_features you'd like to generate.
        """
        if isinstance(arg_dict["resolve_multiple"], str):
            # Try from resolve_multiple_fns
            resolved_func = False

            if resolve_multiple_fns is not None:
                resolved_func = resolve_multiple_fns.get(
                    [arg_dict["resolve_multiple"]],
                )

            if not isinstance(resolved_func, Callable):
                resolved_func = resolve_fns.get(arg_dict["resolve_multiple"])

            if not isinstance(resolved_func, Callable):
                raise ValueError(
                    "resolve_function neither is nor resolved to a Callable",
                )

    def _set_kwargs_for_create_flattened_df_for_val(self, arg_dict):
        """Rename arguments for create_flattened_df_for_val."""
        arg_dict["values_df"] = arg_dict["predictor_df"]
        arg_dict["interval_days"] = arg_dict["lookbehind_days"]
        arg_dict["direction"] = "behind"
        arg_dict["id_col_name"] = self.id_col_name
        arg_dict["timestamp_col_name"] = self.timestamp_col_name
        arg_dict["pred_time_uuid_col_name"] = self.pred_time_uuid_col_name
        arg_dict["new_col_name_prefix"] = self.predictor_col_name_prefix

        if "new_col_name" not in arg_dict.keys():
            arg_dict["new_col_name"] = arg_dict["values_df"]

        return arg_dict

    def add_temporal_predictors_from_list_of_argument_dictionaries(  # pylint: disable=too-many-branches
        self,
        predictors: list[dict[str, str]],
        predictor_dfs: dict[str, DataFrame] = None,
        resolve_multiple_fns: Optional[dict[str, Callable]] = None,
    ):
        """Add predictors to the flattened dataframe from a list.

        Args:
            predictors (list[dict[str, str]]): A list of dictionaries describing the prediction_features you'd like to generate.
            predictor_dfs (dict[str, DataFrame], optional): If wanting to pass already resolved dataframes.
                By default, you should add your dataframes to the @data_loaders registry.
                Then the the predictor_df value in the predictor dict will map to a callable which returns the dataframe.
                Optionally, you can map the string to a dataframe in predictor_dfs.
            resolve_multiple_fns (Union[str, Callable], optional): If wanting to use manually defined resolve_multiple strategies
                I.e. ones that aren't in the resolve_fns catalogue require a dictionary mapping the
                resolve_multiple string to a Callable object. Defaults to None.

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
            >>>         "resolve_multiple_fns": "min",
            >>>         "fallback": 0,
            >>>         "source_values_col_name": "val",
            >>>     }
            >>> ]
            >>> predictor_dfs = {"df_name": df_object}
            >>> resolve_multiple_strategies = {"resolve_multiple_strat_name": resolve_multiple_func}

            >>> dataset.add_predictors_from_list(
            >>>     predictor_list=predictor_list,
            >>>     predictor_dfs=predictor_dfs,
            >>>     resolve_multiple_fn_dict=resolve_multiple_strategies,
            >>> )

        Raises:
            ValueError: If predictor_df is not in the data_loaders registry or predictor_dfs.
        """
        processed_arg_dicts = []

        dicts_found_in_predictor_dfs = []

        # Replace strings with objects as relevant
        for arg_dict in predictors:

            # If resolve_multiple is a string, see if possible to resolve to a Callable
            # Actual resolving is handled in resolve_multiple_values_within_interval_days
            # To preserve str for column name generation
            self._check_if_resolve_multiple_can_convert_to_callable(
                resolve_multiple_fns=resolve_multiple_fns,
                arg_dict=arg_dict,
            )

            # Rename arguments for create_flattened_df_for_val
            arg_dict = self._set_kwargs_for_create_flattened_df_for_val(
                arg_dict=arg_dict,
            )

            # Resolve values_df to either a dataframe from predictor_dfs_dict or a callable from the registry
            arg_dict = self._resolve_predictor_df_str_to_df(
                predictor_dfs=predictor_dfs,
                dicts_found_in_predictor_dfs=dicts_found_in_predictor_dfs,
                arg_dict=arg_dict,
            )

            self._check_and_prune_to_required_arg_dict_keys(
                processed_arg_dicts=processed_arg_dicts,
                arg_dict=arg_dict,
            )

        # Validate dicts before starting pool, saves time if errors!
        self._validate_processed_arg_dicts(processed_arg_dicts)

        with Pool(self.n_workers) as p:
            flattened_predictor_dfs = p.map(
                self._get_feature,
                processed_arg_dicts,
            )

        flattened_predictor_dfs = [
            df.set_index(self.pred_time_uuid_col_name) for df in flattened_predictor_dfs
        ]

        msg.info("Feature generation complete, concatenating")

        flattened_predictor_dds = [
            dd.from_pandas(df, npartitions=6) for df in flattened_predictor_dfs
        ]

        # Concatenate with dask, and show progress bar
        with TqdmCallback(desc="compute"):
            concatenated_dfs = (
                dd.concat(flattened_predictor_dds, axis=1, interleave_partitions=True)
                .compute()  # Converts to pandas dataframe
                .reset_index()
            )

        self.df = pd.merge(
            self.df,
            concatenated_dfs,
            how="left",
            on=self.pred_time_uuid_col_name,
            suffixes=("", ""),
            validate="1:1",
        )

        self.df = self.df.copy()

    def add_age(
        self,
        id2date_of_birth: DataFrame,
        date_of_birth_col_name: Optional[str] = "date_of_birth",
    ):
        """Add age at prediction time to each prediction time.

        Args:
            id2date_of_birth (DataFrame): Two columns, id and date_of_birth.
            date_of_birth_col_name (str, optional): Name of the date_of_birth column in id2date_of_birth.
            Defaults to "date_of_birth".

        Raises:
            ValueError: _description_
        """
        if id2date_of_birth[date_of_birth_col_name].dtype != "<M8[ns]":
            try:
                id2date_of_birth[date_of_birth_col_name] = pd.to_datetime(
                    id2date_of_birth[date_of_birth_col_name],
                    format="%Y-%m-%d",
                )
            except Exception as e:
                raise ValueError(
                    f"Conversion of {date_of_birth_col_name} to datetime failed, doesn't match format %Y-%m-%d. Recommend converting to datetime before adding.",
                ) from e

        self.add_static_info(
            info_df=id2date_of_birth,
            input_col_name=date_of_birth_col_name,
        )

        age = (
            (
                self.df[self.timestamp_col_name]
                - self.df[f"{self.predictor_col_name_prefix}_{date_of_birth_col_name}"]
            ).dt.days
            / (365.25)
        ).round(2)

        self.df.drop(
            f"{self.predictor_col_name_prefix}_{date_of_birth_col_name}",
            axis=1,
            inplace=True,
        )

        self.df[f"{self.predictor_col_name_prefix}_age_in_years"] = age

    def add_static_info(
        self,
        info_df: DataFrame,
        prefix: Optional[str] = "self.predictor_col_name_prefix",
        input_col_name: Optional[str] = None,
        output_col_name: Optional[str] = None,
    ):
        """Add static info to each prediction time, e.g. age, sex etc.

        Args:
            info_df (DataFrame): Contains an id_column and a value column.
            prefix (str, optional): Prefix for column. Defaults to self.predictor_col_name_prefix.
            input_col_name (str, optional): Column names for the values you want to add. Defaults to "value".
            output_col_name (str, optional): Name of the output column. Defaults to None.

        Raises:
            ValueError: If input_col_name does not match a column in info_df.
        """

        value_col_name = [col for col in info_df.columns if col not in self.id_col_name]

        # Try to infer value col name if not provided
        if input_col_name is None:
            if len(value_col_name) == 1:
                value_col_name = value_col_name[0]
            elif len(value_col_name) > 1:
                raise ValueError(
                    f"Only one value column can be added to static info, found multiple: {value_col_name}",
                )
            elif len(value_col_name) == 0:
                raise ValueError("No value column found in info_df, please check.")
        else:
            value_col_name = input_col_name

        # Find output_col_name
        if prefix == "self.predictor_col_name_prefix":
            prefix = self.predictor_col_name_prefix

        if output_col_name is None:
            output_col_name = f"{prefix}_{value_col_name}"
        else:
            output_col_name = f"{prefix}_{output_col_name}"

        df = pd.DataFrame(
            {
                self.id_col_name: info_df[self.id_col_name],
                output_col_name: info_df[value_col_name],
            },
        )

        self.df = pd.merge(
            self.df,
            df,
            how="left",
            on=self.id_col_name,
            suffixes=("", ""),
            validate="m:1",
        )

    def add_temporal_outcome(
        self,
        outcome_df: DataFrame,
        lookahead_days: float,
        resolve_multiple: Union[Callable, str],
        fallback: float,
        incident: Optional[bool] = False,
        pred_name: Optional[Union[str, list]] = "value",
        dichotomous: Optional[bool] = False,
    ):
        """Add an outcome-column to the dataset.

        Args:
            outcome_df (DataFrame): A table in wide format. Required columns: patient_id, timestamp, value.
            lookahead_days (float): How far ahead to look for an outcome in days. If none found, use fallback.
            resolve_multiple (Callable, str): How to handle multiple values within the lookahead window. Takes either i) a function that takes a list as an argument and returns a float, or ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (float): What to do if no value within the lookahead.
            incident (Optional[bool], optional): Whether looking for an incident outcome. If true, removes all prediction times after the outcome time. Defaults to false.
            pred_name (Optional[Union[str, list]]): Name to use for new column(s). Automatically generated as '{pred_name}_within_{lookahead_days}_days'. Defaults to "value".
            dichotomous (bool, optional): Whether the outcome is dichotomous. Allows computational shortcuts, making adding an outcome _much_ faster. Defaults to False.
        """
        prediction_timestamp_col_name = f"{self.timestamp_col_name}_prediction"
        outcome_timestamp_col_name = f"{self.timestamp_col_name}_outcome"
        if incident:
            df = pd.merge(
                self.df,
                outcome_df,
                how="left",
                on=self.id_col_name,
                suffixes=("_prediction", "_outcome"),
                validate="m:1",
            )

            df = df.drop(
                df[
                    df[outcome_timestamp_col_name] < df[prediction_timestamp_col_name]
                ].index,
            )

            if dichotomous:
                full_col_str = f"{self.outcome_col_name_prefix}_dichotomous_{pred_name}_within_{lookahead_days}_days_{resolve_multiple}_fallback_{fallback}"

                df[full_col_str] = (
                    df[prediction_timestamp_col_name] + timedelta(days=lookahead_days)
                    > df[outcome_timestamp_col_name]
                ).astype(int)

            df.rename(
                {prediction_timestamp_col_name: "timestamp"},
                axis=1,
                inplace=True,
            )
            df.drop([outcome_timestamp_col_name], axis=1, inplace=True)

            df.drop(["value"], axis=1, inplace=True)

            self.df = df

        if not (dichotomous and incident):
            self.add_temporal_col_to_flattened_dataset(
                values_df=outcome_df,
                direction="ahead",
                interval_days=lookahead_days,
                resolve_multiple=resolve_multiple,
                fallback=fallback,
                new_col_name=pred_name,
            )

    def add_temporal_predictor(
        self,
        predictor_df: DataFrame,
        lookbehind_days: float,
        resolve_multiple: Union[Callable, str],
        fallback: float,
        new_col_name: str = None,
    ):
        """Add a column with predictor values to the flattened dataset (e.g.
        "average value of bloodsample within n days").

        Args:
            predictor_df (DataFrame): A table in wide format. Required columns: patient_id, timestamp, value.
            lookbehind_days (float): How far behind to look for a predictor value in days. If none found, use fallback.
            resolve_multiple (Callable, str): How to handle multiple values within the lookbehind window. Takes either i) a function that takes a list as an argument and returns a float, or ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (float): What to do if no value within the lookahead.
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
        """
        self.add_temporal_col_to_flattened_dataset(
            values_df=predictor_df,
            direction="behind",
            interval_days=lookbehind_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            new_col_name=new_col_name,
        )

    def add_temporal_col_to_flattened_dataset(
        self,
        values_df: Union[DataFrame, str],
        direction: str,
        interval_days: float,
        resolve_multiple: Union[Callable, str],
        fallback: float,
        new_col_name: Optional[Union[str, list]] = None,
    ):
        """Add a column to the dataset (either predictor or outcome depending
        on the value of "direction").

        Args:
            values_df (DataFrame): A table in wide format. Required columns: patient_id, timestamp, value.
            direction (str): Whether to look "ahead" or "behind".
            interval_days (float): How far to look in direction.
            resolve_multiple (Callable, str): How to handle multiple values within interval_days. Takes either i) a function that takes a list as an argument and returns a float, or ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (float): What to do if no value within the lookahead.
            new_col_name (Optional[Union[str, list]]): Name to use for new column(s). Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
        """
        timestamp_col_type = type(values_df[self.timestamp_col_name][0]).__name__

        if timestamp_col_type not in ["Timestamp"]:
            raise ValueError(
                f"{self.timestamp_col_name} is of type {timestamp_col_type}, not 'Timestamp' from Pandas. Will cause problems. Convert before initialising FlattenedDataset.",
            )

        if direction == "behind":
            new_col_name_prefix = self.predictor_col_name_prefix
        elif direction == "ahead":
            new_col_name_prefix = self.outcome_col_name_prefix

        df = FlattenedDataset.flatten_temporal_values_to_df(
            prediction_times_with_uuid_df=self.df[
                [
                    self.id_col_name,
                    self.timestamp_col_name,
                    self.pred_time_uuid_col_name,
                ]
            ],
            values_df=values_df,
            direction=direction,
            interval_days=interval_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            new_col_name=new_col_name,
            id_col_name=self.id_col_name,
            timestamp_col_name=self.timestamp_col_name,
            pred_time_uuid_col_name=self.pred_time_uuid_col_name,
            new_col_name_prefix=new_col_name_prefix,
        )

        self.df = pd.merge(
            self.df,
            df,
            how="left",
            on=self.pred_time_uuid_col_name,
            validate="1:1",
        )

    @staticmethod
    def flatten_temporal_values_to_df(  # noqa pylint: disable=too-many-locals
        prediction_times_with_uuid_df: DataFrame,
        values_df: Union[Callable, DataFrame],
        direction: str,
        interval_days: float,
        resolve_multiple: Union[Callable, str],
        fallback: Union[float, str],
        id_col_name: str,
        timestamp_col_name: str,
        pred_time_uuid_col_name: str,
        new_col_name: Union[str, list],
        new_col_name_prefix: Optional[str] = None,
        loader_kwargs: Optional[dict] = None,
    ) -> DataFrame:

        """Create a dataframe with flattened values (either predictor or
        outcome depending on the value of "direction").

        Args:
            prediction_times_with_uuid_df (DataFrame): Dataframe with id_col and
                timestamps for each prediction time.
            values_df (Union[Callable, DataFrame]): A dataframe or callable resolving to
                a dataframe containing id_col, timestamp and value cols.
            direction (str): Whether to look "ahead" or "behind" the prediction time.
            interval_days (float): How far to look in each direction.
            resolve_multiple (Union[Callable, str]): How to handle multiple values
                within interval_days. Takes either
                i) a function that takes a list as an argument and returns a float, or
                ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (Union[float, str]): Which value to put if no value within the
                lookahead. "NaN" for Pandas NA.
            id_col_name (str): Name of id_column in prediction_times_with_uuid_df and
                values_df. Required because this is a static method.
            timestamp_col_name (str): Name of timestamp column in
                prediction_times_with_uuid_df and values_df. Required because this is a
                static method.
            pred_time_uuid_col_name (str): Name of uuid column in
                prediction_times_with_uuid_df. Required because this is a static method.
            new_col_name (Union[str, list]): Name of new column(s) in returned
                dataframe.
            new_col_name_prefix (str, optional): Prefix to use for new column name.
            loader_kwargs (dict, optional): Keyword arguments to pass to the loader


        Returns:
            DataFrame
        """

        # Rename column
        if new_col_name is None:
            raise ValueError("No name for new colum")

        full_col_str = generate_feature_colname(
            prefix=new_col_name_prefix,
            out_col_name=new_col_name,
            interval_days=interval_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            loader_kwargs=loader_kwargs,
        )

        # Resolve values_df if not already a dataframe.
        if isinstance(values_df, Callable):
            if loader_kwargs:
                values_df = values_df(**loader_kwargs)
            else:
                values_df = values_df()

        if not isinstance(values_df, DataFrame):
            raise ValueError("values_df is not a dataframe")

        for col_name in [timestamp_col_name, id_col_name]:
            if col_name not in values_df.columns:
                raise ValueError(
                    f"{col_name} does not exist in df_prediction_times, change the df or set another argument",
                )

        # Generate df with one row for each prediction time x event time combination
        # Drop dw_ek_borger for faster merge
        df = pd.merge(
            left=prediction_times_with_uuid_df,
            right=values_df,
            how="left",
            on=id_col_name,
            suffixes=("_pred", "_val"),
            validate="m:m",
        ).drop("dw_ek_borger", axis=1)

        # Drop prediction times without event times within interval days
        df = FlattenedDataset.drop_records_outside_interval_days(
            df,
            direction=direction,
            interval_days=interval_days,
            timestamp_pred_colname="timestamp_pred",
            timestamp_value_colname="timestamp_val",
        )

        # Add back prediction times that don't have a value, and fill them with fallback
        df = FlattenedDataset.add_back_prediction_times_without_value(
            df=df,
            pred_times_with_uuid=prediction_times_with_uuid_df,
            pred_time_uuid_colname=pred_time_uuid_col_name,
        ).fillna(fallback)

        df["timestamp_val"].replace({fallback: pd.NaT}, inplace=True)

        df = FlattenedDataset.resolve_multiple_values_within_interval_days(
            resolve_multiple=resolve_multiple,
            df=df,
            timestamp_col_name=timestamp_col_name,
            pred_time_uuid_colname=pred_time_uuid_col_name,
        )

        # If resolve_multiple generates empty values,
        # e.g. when there is only one prediction_time within look_ahead window for slope calculation,
        # replace with NaN

        # if only 1 value, only replace that one
        if isinstance(new_col_name, str):
            df["value"] = df["value"].replace({np.NaN: fallback})
            df = df.rename(columns={"value": full_col_str})
            full_col_str = [full_col_str]  # to concat with other columns

        # if multiple values, replace all with na and handle renaming
        elif isinstance(new_col_name, list):
            metadata_df = df.drop(new_col_name, axis=1)
            df = df[new_col_name]
            df = df.fillna(fallback)
            df.columns = full_col_str
            df = pd.concat([metadata_df, df], axis=1)

        msg.good(
            f"Returning {df.shape[0]} rows of flattened dataframe with {full_col_str}",
        )

        cols_to_return = [pred_time_uuid_col_name] + full_col_str

        return df[cols_to_return]

    @staticmethod
    def add_back_prediction_times_without_value(
        df: DataFrame,
        pred_times_with_uuid: DataFrame,
        pred_time_uuid_colname: str,
    ) -> DataFrame:
        """Ensure all prediction times are represented in the returned
        dataframe.

        Args:
            df (DataFrame): Dataframe with prediction times but without uuid.
            pred_times_with_uuid (DataFrame): Dataframe with prediction times and uuid.
            pred_time_uuid_colname (str): Name of uuid column in both df and pred_times_with_uuid.

        Returns:
            DataFrame: A merged dataframe with all prediction times.
        """
        return pd.merge(
            pred_times_with_uuid,
            df,
            how="left",
            on=pred_time_uuid_colname,
            suffixes=("", "_temp"),
        ).drop(["timestamp_pred"], axis=1)

    @staticmethod
    def resolve_multiple_values_within_interval_days(
        resolve_multiple: Callable,
        df: DataFrame,
        timestamp_col_name: str,
        pred_time_uuid_colname: str,
    ) -> DataFrame:
        """Apply the resolve_multiple function to prediction_times where there
        are multiple values within the interval_days lookahead.

        Args:
            resolve_multiple (Callable): Takes a grouped df and collapses each group to one record (e.g. sum, count etc.).
            df (DataFrame): Source dataframe with all prediction time x val combinations.
            timestamp_col_name (str): Name of timestamp column in df.
            pred_time_uuid_colname (str): Name of uuid column in df.

        Returns:
            DataFrame: DataFrame with one row pr. prediction time.
        """
        # Convert timestamp val to numeric that can be used for resolve_multiple functions
        # Numeric value amounts to days passed since 1/1/1970
        df["timestamp_val"] = (
            df["timestamp_val"] - dt.datetime(1970, 1, 1)
        ).dt.total_seconds() / 86400

        # Sort by timestamp_pred in case resolve_multiple needs dates
        df = df.sort_values(by=timestamp_col_name).groupby(pred_time_uuid_colname)

        if isinstance(resolve_multiple, str):
            resolve_multiple = resolve_fns.get(resolve_multiple)

        if isinstance(resolve_multiple, Callable):
            df = resolve_multiple(df).reset_index()
        else:
            raise ValueError("resolve_multiple must be or resolve to a Callable")

        return df

    @staticmethod
    def drop_records_outside_interval_days(
        df: DataFrame,
        direction: str,
        interval_days: float,
        timestamp_pred_colname: str,
        timestamp_value_colname: str,
    ) -> DataFrame:
        """Keep only rows where timestamp_value is within interval_days in
        direction of timestamp_pred.

        Args:
            direction (str): Whether to look ahead or behind.
            interval_days (float): How far to look
            df (DataFrame): Source dataframe
            timestamp_pred_colname (str): Name of timestamp column for predictions in df.
            timestamp_value_colname (str): Name of timestamp column for values in df.

        Raises:
            ValueError: If direction is niether ahead nor behind.

        Returns:
            DataFrame
        """
        df["time_from_pred_to_val_in_days"] = (
            (df[timestamp_value_colname] - df[timestamp_pred_colname])
            / (np.timedelta64(1, "s"))
            / 86_400
        )
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

        return df[df["is_in_interval"]].drop(
            ["is_in_interval", "time_from_pred_to_val_in_days"],
            axis=1,
        )


__all__ = [
    "FlattenedDataset",
    "select_and_assert_keys",
]
