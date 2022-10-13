"""Pre-load dataframes to avoid duplicate loading."""

from multiprocessing import Pool
from typing import Any, Union

import pandas as pd
from wasabi import Printer

from psycop_feature_generation.data_checks.raw.check_raw_df import check_raw_df
from psycop_feature_generation.utils import data_loaders


def load_df(predictor_df: str, values_to_load: Union[str, None] = None) -> pd.DataFrame:
    """Load a dataframe from a SQL database.

    Args:
        predictor_df (str): The name of the loader function which calls the SQL database.
        values_to_load (dict): Which values to load for medications. Takes "all", "numerical" or "numerical_and_coerce". Defaults to None.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    msg = Printer(timestamp=True)

    df = pd.DataFrame()

    msg.info(f"Loading {predictor_df}")

    loader_fns = data_loaders.get_all()

    if predictor_df not in loader_fns:
        msg.fail(f"Could not find loader for {predictor_df}.")
    else:
        # We need this control_flow since some loader_fns don't take values_to_load
        if values_to_load is not None:
            df = loader_fns[predictor_df](values_to_load=values_to_load)
        else:
            df = loader_fns[predictor_df]()

    # Check that df is a dataframe
    if df.shape[0] == 0:
        raise ValueError(f"Loaded dataframe {predictor_df} is empty.")

    msg.info(f"Loaded {predictor_df} with {len(df)} rows")
    return df


def load_df_wrapper(predictor_dict: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Wrapper to load a dataframe from a dictionary.

    Args:
        predictor_dict (dict[str, Any]): dictionary where the key predictor_df maps to an SQL database.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    return {
        predictor_dict["predictor_df"]: load_df(
            predictor_df=predictor_dict["predictor_df"],
            values_to_load=predictor_dict.get("values_to_load"),
        ),
    }


def error_check_dfs(
    pre_loaded_dfs: list[dict[str, pd.DataFrame]],
    subset_duplicates_columns: Union[list, str] = "all",
) -> None:
    """Error check the pre-loaded dataframes.

    Args:
        pre_loaded_dfs (list): list of pre-loaded dataframes.
        subset_duplicates_columns (Union[list, str]): Which columns to check for duplicates across. Defaults to 'All'.
    """
    # Error check the laoded dfs
    failures = []

    msg = Printer(timestamp=True)

    for d in pre_loaded_dfs:
        for k in d.keys():
            source_failures, _ = check_raw_df(
                df=d[k],
                subset_duplicates_columns=subset_duplicates_columns,
                raise_error=False,
            )

            if len(source_failures) > 0:
                failures.append({k: source_failures})

    if len(failures) > 0:
        raise ValueError(
            f"Pre-loaded dataframes failed source checks. {failures}",
        )

    msg.info(f"Pre-loaded {len(pre_loaded_dfs)} dataframes, all conformed to criteria")


def pre_load_unique_dfs(
    predictor_dict_list: list[dict[str, Any]],
    subset_duplicates_columns: Union[list, str] = "all",
) -> dict[str, pd.DataFrame]:
    """Pre-load unique dataframes to avoid duplicate loading.

    Args:
        predictor_dict_list (list[dict[str, Union[str, float, int]]]): list of dictionaries where the key predictor_df maps to an SQL database.
        subset_duplicates_columns (Union[list, str]): Which columns to check for duplicates across. Defaults to "all".

    Returns:
        dict[str, pd.DataFrame]: A dictionary with keys predictor_df and values the loaded dataframe.
    """

    # Get unique predictor_df values from predictor_dict_list
    unique_dfs: set[str] = set()

    selected_predictor_df_specs: list[dict[str, Any]] = []

    for feature_dict in predictor_dict_list:
        if feature_dict["predictor_df"] not in unique_dfs:
            unique_dfs.add(feature_dict["predictor_df"])
            selected_predictor_df_specs.append(feature_dict)

    msg = Printer(timestamp=True)

    msg.info(f"Pre-loading {len(unique_dfs)} dataframes")
    n_workers = min(
        len(unique_dfs),
        16,
    )  # 16 subprocesses should be enough to not be IO bound

    with Pool(n_workers) as p:
        pre_loaded_dfs = p.map(load_df_wrapper, selected_predictor_df_specs)

        error_check_dfs(
            pre_loaded_dfs=pre_loaded_dfs,
            subset_duplicates_columns=subset_duplicates_columns,
        )

        # Combined pre_loaded dfs into one dictionary
        pre_loaded_dfs = {k: v for d in pre_loaded_dfs for k, v in d.items()}  # type: ignore

    return pre_loaded_dfs  # type: ignore
