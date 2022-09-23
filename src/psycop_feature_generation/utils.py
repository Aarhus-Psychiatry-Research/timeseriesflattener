"""A set of misc.

utilities. If this file grows, consider splitting it up.
"""

import os
from pathlib import Path
from typing import Optional, Union

import catalogue
import pandas as pd

data_loaders = catalogue.create("timeseriesflattener", "data_loaders")

SHARED_RESOURCES_PATH = Path(r"E:\shared_resources")
FEATURE_SETS_PATH = SHARED_RESOURCES_PATH / "feature_sets"
OUTCOME_DATA_PATH = SHARED_RESOURCES_PATH / "outcome_data"
RAW_DATA_VALIDATION_PATH = SHARED_RESOURCES_PATH / "raw_data_validation"
FEATURIZERS_PATH = SHARED_RESOURCES_PATH / "featurizers"
MODEL_PREDICTIONS_PATH = SHARED_RESOURCES_PATH / "model_predictions"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def generate_feature_colname(
    prefix: str,
    out_col_name: Union[str, list[str]],
    interval_days: int,
    resolve_multiple: str,
    fallback: str,
    loader_kwargs: Optional[dict] = None,
) -> Union[str, list[str]]:
    """Generates standardized column name from feature collapse information. If
    passed a string, generates a single column name. If passed a list of
    strings, generates a list of column names.

    Args:
        prefix (str): Prefix (typically either "pred" or "outc")
        out_col_name (str): Name after the prefix.
        interval_days (int): Fills out "_within_{interval_days}" in the col name.
        resolve_multiple (str): Name of the resolve_multiple strategy.
        fallback (str): Values used for fallback.
        loader_kwargs (dict, optional): Loader kwargs. Defaults to None.

    Returns:
        str: A full column name
    """
    if isinstance(out_col_name, str):
        out_col_name = [out_col_name]

    col_name = [
        f"{prefix}_{col}_within_{interval_days}_days_{resolve_multiple}_fallback_{fallback}"
        for col in out_col_name
    ]

    # Append {loader_kwargs} to colname if it exists
    if loader_kwargs:
        col_name = [
            f"{col}_{format_dict_for_printing(loader_kwargs)}" for col in col_name
        ]

    if len(col_name) == 1:
        col_name = col_name[0]  # type: ignore

    return col_name


def load_most_recent_csv_matching_pattern_as_df(
    dir_path: Path,
    file_pattern: str,
) -> pd.DataFrame:
    """Load most recent df matching pattern.

    Args:
        dir_path (Path): Directory to search
        file_pattern (str): Pattern to match

    Returns:
        pd.DataFrame: DataFrame matching pattern

    Raises:
        FileNotFoundError: If no file matching pattern is found
    """
    files = list(dir_path.glob(f"*{file_pattern}*.csv"))

    if len(files) == 0:
        raise FileNotFoundError(f"No files matching pattern {file_pattern} found")

    most_recent_file = max(files, key=os.path.getctime)

    return pd.read_csv(most_recent_file)


def df_contains_duplicates(df: pd.DataFrame, col_subset: list[str]):
    """Check if a dataframe contains duplicates.

    Args:
        df (pd.DataFrame): Dataframe to check.
        col_subset (list[str]): Columns to check for duplicates.

    Returns:
        bool: True if duplicates are found.
    """
    return df.duplicated(subset=col_subset).any()
