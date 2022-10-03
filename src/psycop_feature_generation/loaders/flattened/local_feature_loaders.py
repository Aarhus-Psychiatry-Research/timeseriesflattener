"""Feature loaders for loading .csv from disk."""

from pathlib import Path
from typing import Optional

import pandas as pd


def get_predictors(df: pd.DataFrame, include_id: bool) -> pd.DataFrame:
    """Returns the predictors from a dataframe.

    Assumes predictors to be prefixed with 'pred'. Timestamp is also
    returned for predictors, and optionally dw_ek_borger.

    Args:
        df: The dataframe to get the predictors from
        include_id (bool): Whether to include 'dw_ek_borger' in the returned df

    Returns:
        pd.DataFrame: Dataframe with only predictor columns
    """
    pred_regex = (
        "^pred|^timestamp" if not include_id else "^pred|^timestamp|dw_ek_borger"
    )
    return df.filter(regex=pred_regex)


def load_split(
    feature_set_dir: Path,
    split: str,
    nrows: Optional[int] = None,
    file_suffix: str = ".parquet",
) -> pd.DataFrame:
    """Loads a given data split as a dataframe from a directory.

    Args:
        feature_set_dir (Path): Path to directory containing data files
        split (str): Which string to look for (e.g. 'train', 'val', 'test')
        nrows (Optional[int]): Whether to only load a subset of the data
        file_suffix (str): The suffix of the files to load. Defaults to '.parquet'.

    Returns:
        pd.DataFrame: The loaded dataframe
    """
    file_path = list(feature_set_dir.glob(f"*{split}*{file_suffix}"))[0]

    if file_suffix == ".parquet":
        return pd.read_parquet(file_path, nrows=nrows)
    else:
        return pd.read_csv(file_path, nrows=nrows)


def load_split_predictors(
    feature_set_dir: Path,
    split: str,
    include_id: bool,
    nrows: Optional[int] = None,
    file_suffix: str = ".parquet",
) -> pd.DataFrame:
    """Loads predictors from a given data split as a dataframe from a
    directory.

    Args:
        feature_set_dir (Path): Path to directory containing data files
        split (str): Which string to look for (e.g. 'train', 'val', 'test')
        include_id (bool): Whether to include 'dw_ek_borger' in the returned df
        nrows (Optional[int]): Whether to only load a subset of the data
        file_suffix (str): The suffix of the files to load. Defaults to '.parquet'.

    Returns:
        pd.DataFrame: The loaded dataframe
    """
    return get_predictors(
        load_split(
            feature_set_dir,
            split,
            nrows=nrows,
            file_suffix=file_suffix,
        ),
        include_id,
    )


def get_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the outcomes from a dataframe.

    Assumes outcomes to be prefixed with 'outc'.

    Args:
        df: The dataframe to get the outcomes from

    Returns:
        pd.DataFrame: Dataframe with only outcome columns
    """
    return df.filter(regex="^outc")


def load_split_outcomes(
    feature_set_dir: Path,
    split: str,
    nrows: Optional[int] = None,
    file_suffix: str = ".parquet",
) -> pd.DataFrame:
    """Loads outcomes from a given data split as a dataframe from a directory.

    Args:
        feature_set_dir (Path): Path to directory containing data files
        split (str): Which string to look for (e.g. 'train', 'val', 'test')
        nrows (Optional[int]): Whether to only load a subset of the data
        file_suffix (str): The suffix of the files to load. Defaults to '.parquet'.

    Returns:
        pd.DataFrame: The loaded dataframe
    """
    return get_outcomes(
        load_split(feature_set_dir, split, nrows=nrows, file_suffix=file_suffix)
    )
