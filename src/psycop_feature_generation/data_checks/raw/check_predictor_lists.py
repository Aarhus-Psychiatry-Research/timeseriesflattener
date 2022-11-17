"""Check that all feature_dicts conform to correct formatting.

Also check that they return meaningful dictionaries.
"""
from typing import Optional, Union

import pandas as pd
from wasabi import Printer

from psycop_feature_generation.data_checks.raw.check_raw_df import check_raw_df
from psycop_feature_generation.utils import data_loaders


def check_df_conforms_to_feature_spec(
    df: pd.DataFrame,
    required_columns: list[str],
    subset_duplicates_columns: list[str],
    expected_val_dtypes: list[str],
    msg_prefix: str,
    arg_dict: dict,
    allowed_nan_value_prop: float = 0.01,
):
    """Check that a loaded df conforms to to a given feature specification.
    Useful when creating loaders.

    Args:
        df (pd.DataFrame): Dataframe to check.
        required_columns (list[str]): list of required columns.
        subset_duplicates_columns (list[str]): list of columns to subset on when
            checking for duplicates.
        expected_val_dtypes (list[str]): Expected value dtype.
        msg_prefix (str): Prefix to add to all messages.
        arg_dict (dict): Dictionary with specifications for what df should conform to.
        allowed_nan_value_prop (float): Allowed proportion of missing values. Defaults to 0.01.

    Raises:
        ValueError: If df does not conform to d.
    """

    if required_columns is None:
        required_columns = ["dw_ek_borger", "timestamp", "value"]

    if subset_duplicates_columns is None:
        subset_duplicates_columns = ["dw_ek_borger", "timestamp", "value"]

    if expected_val_dtypes is None:
        expected_val_dtypes = ["float64", "int64"]

    msg = Printer(timestamp=True)

    allowed_nan_value_prop = (
        arg_dict["allowed_nan_value_prop"]
        if arg_dict["allowed_nan_value_prop"]
        else allowed_nan_value_prop
    )

    source_failures, _ = check_raw_df(
        df=df,
        required_columns=required_columns,
        subset_duplicates_columns=subset_duplicates_columns,
        allowed_nan_value_prop=allowed_nan_value_prop,
        expected_val_dtypes=expected_val_dtypes,
    )

    # Return errors
    if len(source_failures) != 0:
        msg.fail(f"{msg_prefix} errors: {source_failures}")
        return {arg_dict["predictor_df"]: source_failures}
    else:
        msg.good(
            f"{msg_prefix} passed data validation criteria.",
        )
        return None


def get_predictor_df_with_loader_fn(d: dict, n_rows: int) -> pd.DataFrame:
    """Get predictor_df from d.

    Args:
        d (dict): Dictionary with key predictor_df
        n_rows (int): Number of rows to load

    Returns:
        pd.DataFrame: predictor_df
    """

    loader_fns_dict = data_loaders.get_all()

    if "loader_kwargs" in d:
        return loader_fns_dict[d["predictor_df"]](n_rows=n_rows, **d["loader_kwargs"])
    else:
        return loader_fns_dict[d["predictor_df"]](n_rows=n_rows)


def get_unique_predictor_dfs(predictor_dict_list: list[dict], required_keys: list[str]):
    """Get unique predictor_dfs from predictor_dict_list.

    Args:
        predictor_dict_list (list[dict]): list of dictionaries where the key predictor_df maps to a catalogue registered data loader
            or is a valid dataframe.
        required_keys (list[str]): list of required keys.

    Returns:
        list[dict]: list of unique predictor_dfs.
    """

    dicts_with_subset_keys = []

    for d in predictor_dict_list:
        new_d = {k: d[k] for k in required_keys}

        if "loader_kwargs" in d:
            new_d["loader_kwargs"] = d["loader_kwargs"]

        dicts_with_subset_keys.append(new_d)

    unique_subset_dicts = []

    for predictor_dict in dicts_with_subset_keys:
        if predictor_dict not in unique_subset_dicts:
            unique_subset_dicts.append(predictor_dict)

    return unique_subset_dicts


def check_feature_combinations_return_correct_dfs(  # noqa pylint: disable=too-many-branches
    predictor_dict_list: list[dict[str, Union[str, float, int]]],
    n_rows: int = 1_000,
    required_columns: Optional[list[str]] = None,
    subset_duplicates_columns: Optional[list[str]] = None,
    allowed_nan_value_prop: float = 0.01,
    expected_val_dtypes: Optional[list[str]] = None,
):
    """Test that all predictor_dfs in predictor_list return a valid df.

    Args:
        predictor_dict_list (list[dict[str, Union[str, float, int]]]): list of dictionaries
            where the key predictor_df maps to a catalogue registered data loader
            or is a valid dataframe.
        n_rows (int): Number of rows to test. Defaults to 1_000.
        required_columns (list[str]): list of required columns. Defaults to ["dw_ek_borger", "timestamp", "value"].
        subset_duplicates_columns (list[str]): list of columns to subset on when
            checking for duplicates. Defaults to ["dw_ek_borger", "timestamp"].
        allowed_nan_value_prop (float): Allowed proportion of missing values. Defaults to 0.0.
        expected_val_dtypes (list[str]): Expected value dtype. Defaults to ["float64", "int64"].
    """

    if required_columns is None:
        required_columns = ["dw_ek_borger", "timestamp", "value"]

    if subset_duplicates_columns is None:
        subset_duplicates_columns = ["dw_ek_borger", "timestamp", "value"]

    if expected_val_dtypes is None:
        expected_val_dtypes = ["float64", "int64"]

    msg = Printer(timestamp=True)

    msg.info("Checking that feature combinations conform to correct formatting")

    # Find all dicts that are unique on keys predictor_df and allowed_nan_value_prop
    required_keys = ["predictor_df", "allowed_nan_value_prop"]

    unique_subset_dicts = get_unique_predictor_dfs(
        predictor_dict_list=predictor_dict_list,
        required_keys=required_keys,
    )

    msg.info(f"Loading {n_rows} rows from each predictor_df")

    failure_dicts = []

    # Check each predictor df
    for i, d in enumerate(unique_subset_dicts):  # pylint: disable=invalid-name
        # Check that it returns a dataframe
        try:
            df = get_predictor_df_with_loader_fn(d=d, n_rows=n_rows)
        except KeyError:
            msg.warn(
                f"{d['predictor_df']} does not appear to be a loader function in catalogue, assuming a well-formatted dataframe. Continuing.",
            )
            continue

        msg_prefix = f"{i+1}/{len(unique_subset_dicts)} {d['predictor_df']}:"

        df_failures = check_df_conforms_to_feature_spec(
            df=df,
            required_columns=required_columns,
            subset_duplicates_columns=subset_duplicates_columns,
            expected_val_dtypes=expected_val_dtypes,
            msg_prefix=msg_prefix,
            allowed_nan_value_prop=allowed_nan_value_prop,
            arg_dict=d,
        )

        if df_failures:
            failure_dicts.append(df_failures)

    if not failure_dicts:
        msg.good(
            f"Checked {len(unique_subset_dicts)} predictor_dfs, all returned appropriate dfs",
        )
    else:
        raise ValueError(f"{failure_dicts}")
