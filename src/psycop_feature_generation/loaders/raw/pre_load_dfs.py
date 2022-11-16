"""Pre-load dataframes to avoid duplicate loading."""
from multiprocessing import Pool
from typing import Union

import pandas as pd
import tqdm
from wasabi import Printer

from psycop_feature_generation.data_checks.raw.check_raw_df import check_raw_df
from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    AnySpec,
    TemporalSpec,
)
from psycop_feature_generation.utils import data_loaders


def load_df(
    loader_fn_name: str,
    values_to_load: Union[str, None] = None,
) -> pd.DataFrame:
    """Load a dataframe from a SQL database.

    Args:
        loader_fn_name (str): The name of the loader function which calls the SQL database.
        values_to_load (dict): Which values to load for medications. Takes "all", "numerical" or "numerical_and_coerce". Defaults to None.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    msg = Printer(timestamp=True)

    df = pd.DataFrame()

    msg.info(f"{loader_fn_name}: Loading")

    loader_fns = data_loaders.get_all()

    if loader_fn_name not in loader_fns:
        msg.fail(f"Could not find loader for {loader_fn_name}.")
    else:
        # We need this control_flow since some loader_fns don't take values_to_load
        if values_to_load is not None:
            df = loader_fns[loader_fn_name](values_to_load=values_to_load)
        else:
            df = loader_fns[loader_fn_name]()

    # Check that df is a dataframe
    if df.shape[0] == 0:
        raise ValueError(f"Loaded dataframe {loader_fn_name} is empty.")

    msg.info(f"{loader_fn_name}: Loaded with {len(df)} rows")
    return df


def load_df_wrapper(spec: TemporalSpec) -> dict[str, pd.DataFrame]:
    """Wrapper to load a dataframe from a dictionary.

    Args:
        spec (MinSpec): A MinSpec object.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    return {
        spec.values_lookup_name: load_df(
            loader_fn_name=spec.values_lookup_name,
            values_to_load=spec.lab_values_to_load
            if hasattr(spec, "lab_values_to_load")
            else None,
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
    specs: list[AnySpec],
    subset_duplicates_columns: Union[list, str] = "all",
    max_workers: int = 16,
) -> dict[str, pd.DataFrame]:
    """Pre-load unique dataframes to avoid duplicate loading.

    Args:
        specs (list): A list of MinSpec objects.
        subset_duplicates_columns (Union[list, str]): Which columns to check for duplicates across. Defaults to "all".

    Returns:
        dict[str, pd.DataFrame]: A dictionary with keys predictor_df and values the loaded dataframe.
    """

    # Get unique predictor_df values from predictor_dict_list
    selected_df_names = []
    selected_specs = []

    for spec in specs:
        if not spec.values_lookup_name in selected_df_names:
            selected_df_names += [spec.values_lookup_name]
            selected_specs += [spec]

    msg = Printer(timestamp=True)

    msg.info(f"Pre-loading {len(selected_specs)} dataframes")

    n_workers = min(
        len(selected_specs),
        max_workers,
    )  # 16 subprocesses should be enough to not be IO bound

    with Pool(n_workers) as p:
        pre_loaded_dfs = list(
            tqdm.tqdm(
                p.imap(func=load_df_wrapper, iterable=selected_specs),
                total=len(selected_specs),
            ),
        )

        error_check_dfs(
            pre_loaded_dfs=pre_loaded_dfs,
            subset_duplicates_columns=subset_duplicates_columns,
        )

        # Combined pre_loaded dfs into one dictionary
        pre_loaded_dfs = {k: v for d in pre_loaded_dfs for k, v in d.items()}  # type: ignore

    return pre_loaded_dfs  # type: ignore
