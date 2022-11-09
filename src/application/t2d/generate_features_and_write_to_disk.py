"""Main example on how to generate features.

Uses T2D-features. WIP, will be migrated to psycop-t2d when reaching
maturity.
"""

import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import psutil
import wandb
from wasabi import Printer

import psycop_feature_generation.loaders.raw  # noqa
from psycop_feature_generation.data_checks.flattened.data_integrity import (
    save_feature_set_integrity_from_dir,
)
from psycop_feature_generation.data_checks.flattened.feature_describer import (
    save_feature_description_from_dir,
)
from psycop_feature_generation.loaders.raw.pre_load_dfs import pre_load_unique_dfs
from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    OutcomeGroupSpec,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
    TemporalSpec,
)
from psycop_feature_generation.timeseriesflattener.flattened_dataset import (
    FlattenedDataset,
)
from psycop_feature_generation.utils import (
    FEATURE_SETS_PATH,
    PROJECT_ROOT,
    write_df_to_file,
)


def finish_wandb(run: wandb.wandb_sdk.wandb_run.Run):
    """Log artifacts and finish the run."""

    run.log_artifact("poetry.lock", name="poetry_lock_file", type="poetry_lock")

    run.finish()


def init_wandb(
    wandb_project_name: str,
    predictor_specs: Sequence[PredictorSpec],
    save_dir: Union[Path, str],
) -> wandb.wandb_sdk.wandb_run.Run:
    """Initialise wandb logging. Allows to use wandb to track progress, send
    Slack notifications if failing, and track logs.

    Args:
        wandb_project_name (str): Name of wandb project.
        predictor_specs (Iterable[dict[str, Any]]): List of predictor specs.
        save_dir (Union[Path, str]): Path to save dir.

    Return:
        wandb.wandb_sdk.wandb_run.Run: WandB run.
    """

    feature_settings = {
        "save_path": save_dir,
        "predictor_list": [spec.__dict__ for spec in predictor_specs],
    }

    # on Overtaci, the wandb tmp directory is not automatically created
    # so we create it here
    # create dewbug-cli.one folders in /tmp and project dir
    if sys.platform == "win32":
        (Path(tempfile.gettempdir()) / "debug-cli.onerm").mkdir(
            exist_ok=True,
            parents=True,
        )
        (PROJECT_ROOT / "wandb" / "debug-cli.onerm").mkdir(exist_ok=True, parents=True)

    run = wandb.init(project=wandb_project_name, config=feature_settings)

    return run  # type: ignore


def save_feature_set_description_to_disk(
    predictor_specs: list,
    flattened_dataset_file_dir: Path,
    out_dir: Path,
    file_suffix: str,
    describe_splits: bool = True,
    compare_splits: bool = True,
):
    """Describe output.

    Args:
        predictor_specs (list): List of predictor specs.
        flattened_dataset_file_dir (Path): Path to dir containing flattened time series files.
        out_dir (Path): Path to output dir.
        file_suffix (str): File suffix.
        describe_splits (bool, optional): Whether to describe each split. Defaults to True.
        compare_splits (bool, optional): Whether to compare splits, e.g. do all categories exist in both train and val. Defaults to True.
    """

    # Create data integrity report
    if describe_splits:
        save_feature_description_from_dir(
            feature_set_dir=flattened_dataset_file_dir,
            predictor_combinations=predictor_specs,
            splits=["train"],
            out_dir=out_dir,
            file_suffix=file_suffix,
        )

    # Describe/compare splits control flow happens within this function
    if compare_splits:
        save_feature_set_integrity_from_dir(
            feature_set_dir=flattened_dataset_file_dir,
            split_names=["train", "val", "test"],
            out_dir=out_dir,
            file_suffix=file_suffix,
            describe_splits=describe_splits,
            compare_splits=compare_splits,
        )


def create_save_dir_path(
    proj_path: Path,
    feature_set_id: str,
) -> Path:
    """Create save directory.

    Args:
        proj_path (Path): Path to project.
        feature_set_id (str): Feature set id.

    Returns:
        Path: Path to sub directory.
    """

    # Split and save to disk
    # Create directory to store all files related to this run
    save_dir = proj_path / "feature_sets" / feature_set_id

    if not save_dir.exists():
        save_dir.mkdir()

    return save_dir


def split_and_save_to_disk(
    flattened_df: pd.DataFrame,
    out_dir: Path,
    file_prefix: str,
    file_suffix: str,
    split_ids_dict: Optional[dict[str, pd.Series]] = None,
    splits: Optional[list[str]] = None,
):
    """Split and save to disk.

    Args:
        flattened_df (pd.DataFrame): Flattened dataframe.
        out_dir (Path): Path to output directory.
        file_prefix (str): File prefix.
        file_suffix (str, optional): Format to save to. Takes any of ["parquet", "csv"].
        split_ids_dict (Optional[dict[str, list[str]]]): Dictionary of split ids, like {"train": pd.Series with ids}.
        splits (list, optional): Which splits to create. Defaults to ["train", "val", "test"].
    """

    if splits is None:
        splits = ["train", "val", "test"]

    msg = Printer(timestamp=True)

    flattened_df_ids = flattened_df["dw_ek_borger"].unique()

    # Version table with current date and time
    # prefix with user name to avoid potential clashes

    # Create splits
    for dataset_name in splits:
        if split_ids_dict is None:
            df_split_ids = psycop_feature_generation.loaders.raw.load_ids(
                split=dataset_name,
            )
        else:
            df_split_ids = split_ids_dict[dataset_name]

        # Find IDs which are in split_ids, but not in flattened_df
        split_ids = df_split_ids["dw_ek_borger"].unique()
        flattened_df_ids = flattened_df["dw_ek_borger"].unique()

        ids_in_split_but_not_in_flattened_df = split_ids[
            ~np.isin(split_ids, flattened_df_ids)
        ]

        msg.warn(
            f"{dataset_name}: There are {len(ids_in_split_but_not_in_flattened_df)} ({round(len(ids_in_split_but_not_in_flattened_df)/len(split_ids)*100, 2)}%) ids which are in {dataset_name}_ids but not in flattened_df_ids, will get dropped during merge. If examining patients based on physical visits, see 'OBS: Patients without physical visits' on the wiki for more info.",
        )

        split_df = pd.merge(flattened_df, df_split_ids, how="inner", validate="m:1")

        # Version table with current date and time
        filename = f"{file_prefix}_{dataset_name}.{file_suffix}"
        msg.info(f"Saving {filename} to disk")

        file_path = out_dir / filename

        write_df_to_file(df=split_df, file_path=file_path)

        msg.good(f"{dataset_name}: Succesfully saved to {file_path}")


def add_metadata(
    pre_loaded_dfs: dict[str, pd.DataFrame],
    flattened_dataset: FlattenedDataset,
) -> FlattenedDataset:
    """Add metadata.

    Args:
        outcome_loader_str (str): String to lookup in catalogue to load outcome.
        pre_loaded_dfs (dict[str, pd.DataFrame]): Dictionary of pre-loaded dataframes.
        flattened_dataset (FlattenedDataset): Flattened dataset.

    Returns:
        FlattenedDataset: Flattened dataset.
    """

    # Add timestamp from outcomes
    flattened_dataset.add_static_info(
        info_df=pre_loaded_dfs["t2d"],
        prefix_override="",
        input_col_name_override="timestamp",
        output_col_name_override="timestamp_first_t2d_hba1c",
    )

    flattened_dataset.add_static_info(
        info_df=pre_loaded_dfs["timestamp_exclusion"],
        prefix_override="",
        input_col_name_override="timestamp",
        output_col_name_override="timestamp_exclusion",
    )

    return flattened_dataset


def add_outcomes(
    pre_loaded_dfs: dict[str, pd.DataFrame],
    flattened_dataset: FlattenedDataset,
    outcome_specs: list[OutcomeSpec],
) -> FlattenedDataset:
    """Add outcomes.

    Args:
        outcome_loader_str (str): String to lookup in catalogue to load outcome.
        pre_loaded_dfs (dict[str, pd.DataFrame]): Dictionary of pre-loaded dataframes.
        flattened_dataset (FlattenedDataset): Flattened dataset.
        lookahead_years (list[Union[int, float]]): List of lookahead years.

    Returns:
        FlattenedDataset: Flattened dataset.
    """

    msg = Printer(timestamp=True)
    msg.info("Adding outcomes")

    for spec in outcome_specs:
        msg.info(f"Adding outcome with {spec.interval_days} days of lookahead")

        spec.values_lookup_name = pre_loaded_dfs[spec.outcome_name]

        flattened_dataset.add_temporal_outcome(
            output_spec=spec,
        )

    msg.good("Finished adding outcomes")

    return flattened_dataset


def add_predictors(
    pre_loaded_dfs: dict[str, pd.DataFrame],
    predictor_specs: list[PredictorSpec],
    flattened_dataset: FlattenedDataset,
):
    """Add predictors.

    Args:
        pre_loaded_dfs (dict[str, pd.DataFrame]): Dictionary of pre-loaded dataframes.
        predictor_specs (list[PredictorSpec]): List of predictor specs.
        flattened_dataset (FlattenedDataset): Flattened dataset.
    """

    msg = Printer(timestamp=True)

    msg.info("Adding static predictors")

    flattened_dataset.add_static_info(
        info_df=pre_loaded_dfs["sex_female"],
        input_col_name_override="sex_female",
    )

    flattened_dataset.add_age(pre_loaded_dfs["birthdays"])

    start_time = time.time()

    msg.info("Adding temporal predictors")
    flattened_dataset.add_temporal_predictors_from_pred_specs(
        predictor_specs=predictor_specs,
        preloaded_predictor_dfs=pre_loaded_dfs,
    )

    end_time = time.time()

    # Finish
    msg.good(
        f"Finished adding {len(predictor_specs)} predictors, took {round((end_time - start_time)/60, 1)} minutes",
    )

    return flattened_dataset


def create_full_flattened_dataset(
    outcome_specs: list[OutcomeSpec],
    prediction_time_loader_str: str,
    pre_loaded_dfs: dict[str, pd.DataFrame],
    predictor_specs: list[dict[str, dict[str, Any]]],
    proj_path: Path,
) -> pd.DataFrame:
    """Create flattened dataset.

    Args:
        outcome_specs (list[OutcomeSpec]): List of outcome specs.
        prediction_time_loader_str (str): String to lookup in catalogue to load prediction time.
        pre_loaded_dfs (dict[str, pd.DataFrame]): Dictionary of pre-loaded dataframes.
        predictor_combinations (list[dict[str, dict[str, Any]]]): List of predictor combinations.
        proj_path (Path): Path to project directory.
        lookahead_years (list[Union[int,float]]): List of lookahead years.

    Returns:
        FlattenedDataset: Flattened dataset.
    """
    msg = Printer(timestamp=True)

    msg.info(f"Generating {len(predictor_specs)} features")

    msg.info("Initialising flattened dataset")

    flattened_dataset = FlattenedDataset(
        prediction_times_df=pre_loaded_dfs[prediction_time_loader_str],
        n_workers=min(
            len(predictor_specs),
            psutil.cpu_count(logical=False),
        ),
        feature_cache_dir=proj_path / "feature_cache",
    )

    # Outcome
    flattened_dataset = add_metadata(
        pre_loaded_dfs=pre_loaded_dfs,
        flattened_dataset=flattened_dataset,
    )

    flattened_dataset = add_outcomes(
        pre_loaded_dfs=pre_loaded_dfs,
        outcome_specs=outcome_specs,
        flattened_dataset=flattened_dataset,
    )

    flattened_dataset = add_predictors(
        pre_loaded_dfs=pre_loaded_dfs,
        predictor_specs=predictor_specs,
        flattened_dataset=flattened_dataset,
    )

    return flattened_dataset.df


def setup_for_main(
    predictor_specs: list[PredictorSpec],
    feature_sets_path: Path,
    proj_name: str,
) -> tuple[Path, str]:
    """Setup for main.

    Args:
        predictor_group_spec (list[dict[str, dict[str, Any]]]): List of predictor specifications.
        feature_sets_path (Path): Path to feature sets.
        proj_name (str): Name of project.

    Returns:
        tuple[list[dict[str, dict[str, Any]]], dict[str, pd.DataFrame], Path]: Tuple of predictor combinations, pre-loaded dataframes, and project path.
    """
    # Some predictors take way longer to complete. Shuffling ensures that e.g. the ones that take the longest aren't all
    # at the end of the list.
    random.shuffle(predictor_specs)

    proj_path = feature_sets_path / proj_name

    if not proj_path.exists():
        proj_path.mkdir()

    current_user = Path().home().name
    feature_set_id = f"psycop_{proj_name}_{current_user}_{len(predictor_specs)}_features_{time.strftime('%Y_%m_%d_%H_%M')}"

    return proj_path, feature_set_id


def pre_load_project_dfs(
    specs: list[Union[TemporalSpec, StaticSpec]],
    outcome_loader_str: str,
    prediction_time_loader_str: str,
) -> dict[str, pd.DataFrame]:
    """Pre-load dataframes for project.

    Args:
        predictor_group_spec (list[dict[str, dict[str, Any]]]): List of predictor specs.
        outcome_loader_str (str): Outcome loader string.
        prediction_time_loader_str (str): Prediction time loader string.

    Returns:
        dict[str, pd.DataFrame]: Dictionary of pre-loaded dataframes.
    """

    specs_to_load = (
        specs
        + [StaticSpec(values_df=outcome_loader_str)]
        + [StaticSpec(values_df=prediction_time_loader_str)]
        + [StaticSpec(values_df="sex_female")]
        + [StaticSpec(values_df="birthdays")]
    )

    # Many features will use the same dataframes, so we can load them once and reuse them.
    pre_loaded_dfs = pre_load_unique_dfs(
        specs=specs_to_load,
    )

    return pre_loaded_dfs


def main(
    proj_name: str,
    feature_sets_path: Path,
    prediction_time_loader_str: str,
    predictor_specs: list[PredictorSpec],
    outcome_specs: list[OutcomeSpec],
):
    """Main function for loading, generating and evaluating a flattened
    dataset.

    Args:
        proj_name (str): Name of project.
        feature_sets_path (Path): Path to where feature sets should be stored.
        prediction_time_loader_str (str): Key to lookup in data_loaders registry for prediction time dataframe.
        outcome_loader_str (str): Key to lookup in data_loaders registry for outcome dataframe.
        predictor_specs (list[dict[str, dict[str, Any]]]): List of predictor specs.
        outcome_specs (list[dict[str, dict[str, Any]]]): List of outcome specs.
    """
    proj_path, feature_set_id = setup_for_main(
        predictor_specs=predictor_specs,
        feature_sets_path=feature_sets_path,
        proj_name=proj_name,
    )

    out_dir = create_save_dir_path(
        feature_set_id=feature_set_id,
        proj_path=proj_path,
    )

    run = init_wandb(
        wandb_project_name=proj_name,
        predictor_specs=predictor_specs,
        save_dir=out_dir,  # Save-dir as argument because we want to log the path
    )

    pre_loaded_dfs = pre_load_unique_dfs(
        specs=predictor_specs + outcome_specs,
    )

    flattened_df = create_full_flattened_dataset(
        prediction_time_loader_str=prediction_time_loader_str,
        pre_loaded_dfs=pre_loaded_dfs,
        predictor_specs=predictor_specs,
        outcome_specs=outcome_specs,
        proj_path=proj_path,
    )

    split_and_save_to_disk(
        flattened_df=flattened_df,
        out_dir=out_dir,
        file_prefix=feature_set_id,
        file_suffix="parquet",
    )

    save_feature_set_description_to_disk(
        predictor_specs=predictor_specs,
        flattened_dataset_file_dir=out_dir,
        out_dir=out_dir,
        file_suffix="parquet",
    )

    finish_wandb(
        run=run,
    )


def gen_predictor_spec_list():
    """Generate predictor spec list."""
    resolve_multiple = ["max", "min", "mean", "latest", "count"]
    interval_days = [30, 90, 180, 365, 730]

    predictor_group_specs: list[PredictorSpec] = []

    predictor_group_specs += PredictorSpec(
        values_lookup_name=("hba1c",),
        fallback=np.nan,
        lab_values_to_load="numerical_and_coerce",
        interval_days=[9999],
        resolve_multiple_fn="count",
    )

    predictor_group_specs += PredictorSpec(
        values_lookup_name=(
            "hba1c",
            "alat",
            "hdl",
            "ldl",
            "scheduled_glc",
            "unscheduled_p_glc",
            "triglycerides",
            "fasting_ldl",
            "crp",
            "egfr",
            "albumine_creatinine_ratio",
        ),
        resolve_multiple_fn=resolve_multiple,
        interval_days=interval_days,
        fallback=np.nan,
        lab_values_to_load="numerical_and_coerce",
    )

    predictor_group_specs += PredictorSpec(
        values_lookup_name=(
            "essential_hypertension",
            "hyperlipidemia",
            "polycystic_ovarian_syndrome",
            "sleep_apnea",
        ),
        resolve_multiple_fn=resolve_multiple,
        interval_days=interval_days,
        fallback=0,
    )

    predictor_group_specs += PredictorSpec(
        values_lookup_name=("antipsychotics",),
        interval_days=interval_days,
        resolve_multiple_fn=resolve_multiple,
        fallback=0,
    )

    predictor_group_specs += PredictorSpec(
        values_lookup_name=("weight_in_kg", "height_in_cm", "bmi"),
        interval_days=interval_days,
        resolve_multiple_fn=["latest"],
        fallback=np.nan,
    )

    predictor_specs = []

    for group_spec in predictor_group_specs:
        predictor_specs += group_spec.create_combinations()

    return predictor_specs


if __name__ == "__main__":
    PREDICTOR_SPECS = gen_predictor_spec_list()
    OUTCOME_SPECS = OutcomeGroupSpec(
        values_lookup_name=["t2d"],
        interval_days=[year * 365 for year in [1, 2, 3, 4, 5]],
        resolve_multiple=["max"],
        fallback=0,
        incident=[True],
    )

    main(
        feature_sets_path=FEATURE_SETS_PATH,
        predictor_specs=PREDICTOR_SPECS,
        proj_name="t2d",
        prediction_time_loader_str="physical_visits_to_psychiatry",
        outcome_specs=OUTCOME_SPECS,
    )
