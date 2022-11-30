"""Check that any raw df conforms to the required format."""

from typing import Any, Optional, Union

import pandas as pd


def check_for_duplicates(
    df: pd.DataFrame,
    subset_duplicates_columns: Union[list[str], str],
) -> tuple[pd.Series, list[str]]:
    """Check for duplicates in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe to check.
        subset_duplicates_columns (Union[list[str], str]): list of columns to subset on when checking for duplicates.

    Returns:
        tuple(pd.Series, list[str]): duplicates and list of errors.
    """

    source_failures = []

    duplicates_idx = df.duplicated(subset=subset_duplicates_columns, keep=False)

    if duplicates_idx.any():
        source_failures.append(f"Duplicates found on {subset_duplicates_columns}")
        duplicates = df[duplicates_idx]
    else:
        duplicates = None

    return duplicates, source_failures


def get_column_dtype_failures(
    df: pd.DataFrame,
    expected_val_dtypes: list[str],
    col: str,
) -> Union[str, None]:
    """Check that the column is of the correct dtype.

    Args:
        df (pd.DataFrame): Dataframe to check.
        expected_val_dtypes (list[str]): Expected dtype of value column.
        col (str): Column to check.

    Returns:
        str: errors
    """

    if "timestamp" in col:
        # Check that column has a valid datetime format
        if df[col].dtype != "datetime64[ns]":
            return f"{col}: invalid datetime format"
    elif "value" in col:
        # Check that column has a valid numeric format
        if df[col].dtype not in expected_val_dtypes:
            return f"{col}: dtype {df[col].dtype}, expected {expected_val_dtypes}"

    return None


def get_na_prop_failures(
    df: pd.DataFrame,
    allowed_nan_value_prop: float,
    col: str,
) -> Union[str, None]:
    """Check if column has too many missing values.

    Args:
        df (pd.DataFrame): Dataframe to check.
        allowed_nan_value_prop (float): Allowed proportion of missing values.
        col (str): Column to check.

    Returns:
        str: Error message if too many missing values.
    """

    na_prop = df[col].isna().sum() / df.shape[0]

    if na_prop > 0:
        if col != "value":
            return f"{col}: {na_prop} NaN"
        else:
            if na_prop > allowed_nan_value_prop:
                return f"{col}: {na_prop} NaN (allowed {allowed_nan_value_prop})"

    return None


def check_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    allowed_nan_value_prop: float,
    expected_val_dtypes: list[str],
) -> list[str]:
    """Check that the required columns are present and that the value column.

    Args:
        df (pd.DataFrame): Dataframe to check.
        required_columns (list[str]): list of required columns.
        allowed_nan_value_prop (float): Allowed proportion of missing values.
        expected_val_dtypes (list[str]): Expected dtype of value column.

    Returns:
        list[str]: list of errors.
    """

    source_failures = []

    for col in required_columns:
        if col not in df.columns:
            source_failures.append(f"{col}: not in columns")
            continue

        # Check that columns are correctly formatted
        dtype_failures = get_column_dtype_failures(
            df=df,
            expected_val_dtypes=expected_val_dtypes,
            col=col,
        )

        # Check for NaN in cols
        na_prop_failures = get_na_prop_failures(
            df=df,
            allowed_nan_value_prop=allowed_nan_value_prop,
            col=col,
        )

        for f_type in dtype_failures, na_prop_failures:
            if f_type:
                source_failures.append(f_type)

    return source_failures


def check_raw_df(  # pylint: disable=too-many-branches
    df: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
    allowed_nan_value_prop: float = 0.0,
    expected_val_dtypes: Optional[list[str]] = None,
    subset_duplicates_columns: Union[list, str] = "all",
    raise_error: bool = True,
) -> tuple[list[str], Any]:
    """Check that the raw df conforms to the required format and doesn't
    contain duplicates or missing values.

    Args:
        df (pd.DataFrame): Dataframe to check.
        required_columns (list[str]): list of required columns. Defaults to ["dw_ek_borger", "timestamp", "value"].
        allowed_nan_value_prop (float): Allowed proportion of missing values. Defaults to 0.0.
        expected_val_dtypes (list[str]): Expected dtype of value column. Defaults to "float64".
        subset_duplicates_columns (Union[list, str]): Which columns to check for duplicates across. Defaults to "all".
        raise_error (bool): Whether to raise an error if the df fails the checks. Defaults to True.

    Returns:
        list[str]: list of errors.

    Raises:
        ValueError: If the df fails the checks and raise_error is True.
    """
    source_failures = []

    if required_columns is None:
        if "value" in df.columns:
            required_columns = ["dw_ek_borger", "timestamp", "value"]
        elif "date_of_birth" in df.columns:
            required_columns = ["dw_ek_borger", "date_of_birth"]
        elif "sex_female" in df.columns:
            required_columns = ["dw_ek_borger", "sex_female"]
        else:
            required_columns = ["dw_ek_borger", "timestamp"]

    if subset_duplicates_columns == "all":
        subset_duplicates_columns = list(df.columns)

    # Raise error if string other than 'All' is passed to subset_duplicates_columns
    if (
        isinstance(subset_duplicates_columns, str)
        and subset_duplicates_columns != "all"
    ):
        raise ValueError(
            "subset_duplicates_columns must be a list of column names or 'All'",
        )

    if expected_val_dtypes is None:
        expected_val_dtypes = ["float64", "int64"]

    # Check that the dataframe has a meaningful length
    if df.shape[0] == 0:
        source_failures.append("No rows returned")

    # Check that the dataframe has the required columns
    source_failures += check_required_columns(
        df=df,
        required_columns=required_columns,
        allowed_nan_value_prop=allowed_nan_value_prop,
        expected_val_dtypes=expected_val_dtypes,
    )

    # Check for duplicates
    duplicates, duplicate_sources = check_for_duplicates(
        df=df,
        subset_duplicates_columns=subset_duplicates_columns,
    )

    source_failures += duplicate_sources

    # Raise error if any failures
    if raise_error and len(source_failures) > 0:
        raise ValueError(source_failures)

    return source_failures, duplicates
