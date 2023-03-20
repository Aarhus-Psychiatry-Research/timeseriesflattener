"""A set of misc.

utilities. If this file grows, consider splitting it up.
"""

import functools
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional

import catalogue
import pandas as pd

data_loaders = catalogue.create("timeseriesflattener", "data_loaders")
split_dfs: Dict[str, pd.DataFrame] = {}


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def split_df_and_register_to_dict(
    df: pd.DataFrame,
    entity_id_col_name: str = "entity_id",
    timestamp_col_name: str = "timestamp",
    value_col_name: str = "value",
    value_names_col_name: str = "value_names",
):
    """Split a df with multiple different value types into dataframes only containing values for each specific value type.

    Registers the separated dfs in the df_dict.

    Args:
        df (pd.DataFrame): A dataframe in long format containing the values to be grouped into a catalogue.
        entity_id_col_name (str): Name of the column containing the patient id for each value. Defaults to "entity_id".
        timestamp_col_name (str): Name of the column containing the timestamp for each value. Defaults to "timestamp".
        value_col_name (str): Name of the column containing the value for each value. Defaults to "values".
        value_names_col_name (str): Name of the column containing the names of the different types of values for each value. Defaults to "value_names".
    """
    passed_columns = [
        entity_id_col_name,
        timestamp_col_name,
        value_col_name,
        value_names_col_name,
    ]
    missing_columns = [col for col in passed_columns if col not in list(df.columns)]

    # If any of the required columns is missing, raise an error
    if len(missing_columns) > 0:
        raise ValueError(
            f"The following required column(s) is/are missing from the input dataframe: {missing_columns}. Available columns are {df.columns}.",
        )

    # Get the unique value names from the dataframe
    value_names = df[value_names_col_name].unique()

    for value_name in value_names:
        value_df = df[df[value_names_col_name] == value_name][
            [entity_id_col_name, timestamp_col_name, value_col_name]
        ]

        split_dfs[value_name] = value_df


def format_dict_for_printing(d: dict) -> str:
    """Format a dictionary for printing. Removes extra apostrophes, formats

    colon to dashes, separates items with underscores and removes curly
    brackets.

    Args:
        d (dict): dictionary to format.

    Returns:
        str: Formatted dictionary.

    Example:
        >>> d = {"a": 1, "b": 2}
        >>> print(format_dict_for_printing(d))
        >>> "a-1_b-2"
    """
    return (
        str(d)
        .replace("'", "")
        .replace(": ", "-")
        .replace("{", "")
        .replace("}", "")
        .replace(", ", "_")
    )


def load_dataset_from_file(
    file_path: Path,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load dataset from file. Handles csv and parquet files based on suffix.

    Args:
        file_path (str): File name.
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Dataset
    """

    file_suffix = file_path.suffix

    if file_suffix == ".csv":
        return pd.read_csv(file_path, nrows=nrows)

    if file_suffix == ".parquet":
        if nrows:
            raise ValueError("nrows not supported for parquet files")
        return pd.read_parquet(file_path)

    raise ValueError(f"Invalid file suffix {file_suffix}")


def load_most_recent_file_matching_pattern_as_df(
    dir_path: Path,
    file_pattern: str,
    file_suffix: str,
) -> pd.DataFrame:
    """Load most recent df matching pattern.

    Args:
        dir_path (Path): Directory to search
        file_pattern (str): Pattern to match
        file_suffix (str): File suffix. Must be either ".csv" or ".parquet".

    Returns:
        pd.DataFrame: DataFrame matching pattern

    Raises:
        FileNotFoundError: If no file matching pattern is found
    """
    files = list(dir_path.glob(f"*{file_pattern}*.{file_suffix}"))

    if len(files) == 0:
        raise FileNotFoundError(f"No files matching pattern {file_pattern} found")

    most_recent_file = max(files, key=os.path.getctime)

    return load_dataset_from_file(file_path=most_recent_file)


def df_contains_duplicates(df: pd.DataFrame, col_subset: List[str]) -> bool:
    """Check if a dataframe contains duplicates.

    Args:
        df (pd.DataFrame): Dataframe to check.
        col_subset (List[str]): Columns to check for duplicates.

    Returns:
        bool: True if duplicates are found.
    """
    return df.duplicated(subset=col_subset).any()


def write_df_to_file(
    df: pd.DataFrame,
    file_path: Path,
):
    """Write dataset to file. Handles csv and parquet files based on suffix.

    Args:
        df: Dataset
        file_path (str): File name.
    """

    file_suffix = file_path.suffix

    if file_suffix == ".csv":
        df.to_csv(file_path, index=False)
    elif file_suffix == ".parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Invalid file suffix {file_suffix}")


def assert_no_duplicate_dicts_in_list(predictor_spec_list: List[Dict[str, Any]]):
    """Find potential duplicates in list of dicts.

    Args:
        predictor_spec_list (List[Dict[str, Dict[str, Any]]]): List of predictor combinations.
    """
    # Find duplicates in list of dicts
    seen = set()
    duplicates = set()

    for d in predictor_spec_list:
        # Remove any keys with unhashable values
        # Otherwise, we get an error when using "in".
        d = {k: v for k, v in d.items() if isinstance(v, Hashable)}  # noqa

        d_as_tuple = tuple(d.items())
        if d_as_tuple in seen:  # pylint: disable=R6103
            duplicates.add(d_as_tuple)
        else:
            seen.add(d_as_tuple)

    if len(duplicates) > 0:
        raise ValueError(f"Found duplicates in list of dicts: {duplicates}")


def print_df_dimensions_diff(  # noqa
    func: Callable,
    print_when_starting: bool = False,
    print_when_no_diff: bool = False,
):
    """Print the difference in rows between the input and output dataframes."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa
        log = logging.getLogger(__name__)

        if print_when_starting:
            log.info(f"{func.__name__}: Starting")

        arg_dfs = [arg for arg in args if isinstance(arg, pd.DataFrame)]
        kwargs_dfs = [arg for arg in kwargs.values() if isinstance(arg, pd.DataFrame)]
        potential_dfs = arg_dfs + kwargs_dfs

        if len(potential_dfs) > 1:
            raise ValueError("More than one DataFrame found in args or kwargs")

        df = potential_dfs[0]

        for dim in ("rows", "columns"):
            dim_int = 0 if dim == "rows" else 1

            n_in_dim_before_func = df.shape[dim_int]

            if print_when_no_diff:
                log.info(
                    f"{func.__name__}: {n_in_dim_before_func} {dim} before function",
                )

            result = func(*args, **kwargs)

            diff = n_in_dim_before_func - result.shape[dim_int]

            if diff != 0:
                percent_diff = round(
                    (n_in_dim_before_func - result.shape[dim_int])
                    / n_in_dim_before_func
                    * 100,
                    2,
                )

                log.info(f"{func.__name__}: Dropped {diff} ({percent_diff}%) {dim}")
            else:
                if print_when_no_diff:
                    log.info(f"{func.__name__}: No {dim} dropped")

        return result

    return wrapper
