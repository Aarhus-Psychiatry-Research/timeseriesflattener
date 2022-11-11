"""Main example on how to generate features.

Uses T2D-features. WIP, will be migrated to psycop-t2d when reaching
maturity.
"""

import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import psutil
import wandb
from wasabi import Printer

import psycop_feature_generation.loaders.raw  # noqa
from application.t2d.unresolved_feature_spec_objects import (
    UnresolvedAnySpec,
    UnresolvedLabPredictorGroupSpec,
    UnresolvedLabPredictorSpec,
    UnresolvedOutcomeGroupSpec,
    UnresolvedOutcomeSpec,
    UnresolvedPredictorSpec,
    UnresolvedStaticSpec,
)
from psycop_feature_generation.data_checks.flattened.data_integrity import (
    save_feature_set_integrity_from_dir,
)
from psycop_feature_generation.data_checks.flattened.feature_describer import (
    save_feature_description_from_dir,
)
from psycop_feature_generation.loaders.raw.load_demographic import birthdays
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from psycop_feature_generation.loaders.raw.pre_load_dfs import pre_load_unique_dfs
from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    AnySpec,
    BaseModel,
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

    # on Overtaci, the wandb tmp directory is not automatically created,
    # so we create it here.
    # create debug-cli.one folders in /tmp and project dir
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
            feature_specs=predictor_specs,
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
            f"{dataset_name}: There are {len(ids_in_split_but_not_in_flattened_df)} ({round(len(ids_in_split_but_not_in_flattened_df) / len(split_ids) * 100, 2)}%) ids which are in {dataset_name}_ids but not in flattened_df_ids, will get dropped during merge. If examining patients based on physical visits, see 'OBS: Patients without physical visits' on the wiki for more info.",
        )

        split_df = pd.merge(flattened_df, df_split_ids, how="inner", validate="m:1")

        # Version table with current date and time
        filename = f"{file_prefix}_{dataset_name}.{file_suffix}"
        msg.info(f"Saving {filename} to disk")

        file_path = out_dir / filename

        write_df_to_file(df=split_df, file_path=file_path)

        msg.good(f"{dataset_name}: Succesfully saved to {file_path}")


def add_metadata_to_ds(
    specs: list[AnySpec],
    flattened_dataset: FlattenedDataset,
) -> FlattenedDataset:
    """Add metadata.

    Args:
        static_specs (list[AnySpec]): List of specs.
        flattened_dataset (FlattenedDataset): Flattened dataset.

    Returns:
        FlattenedDataset: Flattened dataset.
    """

    for spec in specs:
        if isinstance(spec, StaticSpec):
            flattened_dataset.add_static_info(
                static_spec=spec,
            )
        elif isinstance(spec, TemporalSpec):
            flattened_dataset.add_temporal_predictor(output_spec=spec)

    return flattened_dataset


def add_outcomes_to_ds(
    flattened_dataset: FlattenedDataset,
    outcome_specs: list[OutcomeSpec],
) -> FlattenedDataset:
    """Add outcomes.

    Args:
        flattened_dataset (FlattenedDataset): Flattened dataset.
        outcome_specs (list[OutcomeSpec]): List of outcome specs.

    Returns:
        FlattenedDataset: Flattened dataset.
    """

    msg = Printer(timestamp=True)
    msg.info("Adding outcomes")

    for spec in outcome_specs:
        msg.info(f"Adding outcome with {spec.interval_days} days of lookahead")

    msg.good("Finished adding outcomes")

    return flattened_dataset


def add_predictors_to_ds(
    temporal_predictor_specs: list[PredictorSpec],
    static_predictor_specs: list[AnySpec],
    birthdays: pd.DataFrame,
    flattened_dataset: FlattenedDataset,
):
    """Add predictors.

    Args:
        temporal_predictor_specs (list[PredictorSpec]): List of predictor specs.
        static_predictor_specs (list[StaticSpec]): List of static specs.
        birthdays (pd.DataFrame): Birthdays. Used for inferring age at each prediction time.
        flattened_dataset (FlattenedDataset): Flattened dataset.
    """

    msg = Printer(timestamp=True)

    msg.info("Adding static predictors")

    for static_spec in static_predictor_specs:
        flattened_dataset.add_static_info(
            static_spec=static_spec,
        )

    flattened_dataset.add_age_and_date_of_birth(birthdays)

    start_time = time.time()

    msg.info("Adding temporal predictors")
    flattened_dataset.add_temporal_predictors_from_pred_specs(
        predictor_specs=temporal_predictor_specs,
    )

    end_time = time.time()

    # Finish
    msg.good(
        f"Finished adding {len(temporal_predictor_specs)} predictors, took {round((end_time - start_time) / 60, 1)} minutes",
    )

    return flattened_dataset


def create_full_flattened_dataset(
    prediction_times: pd.DataFrame,
    birthdays: pd.DataFrame,
    metadata_specs: list[AnySpec],
    temporal_predictor_specs: list[PredictorSpec],
    static_predictor_specs: list[AnySpec],
    outcome_specs: list[OutcomeSpec],
    proj_path: Path,
) -> pd.DataFrame:
    """Create flattened dataset.

    Args:
        outcome_specs (list[OutcomeSpec]): List of outcome specs.
        prediction_times (pd.DataFrame): Dataframe with prediction times.
        pre_loaded_dfs (dict[str, pd.DataFrame]): Dictionary of pre-loaded dataframes.
        temporal_predictor_specs (list[dict[str, dict[str, Any]]]): List of temporal predictor specs.
        static_predictor_specs (list[dict[str, dict[str, Any]]]): List of static predictor specs.
        proj_path (Path): Path to project directory.

    Returns:
        FlattenedDataset: Flattened dataset.
    """
    msg = Printer(timestamp=True)

    msg.info(f"Generating {len(temporal_predictor_specs)} features")

    msg.info("Initialising flattened dataset")

    flattened_dataset = FlattenedDataset(
        prediction_times_df=prediction_times,
        n_workers=min(
            len(temporal_predictor_specs),
            psutil.cpu_count(logical=False),
        ),
        feature_cache_dir=proj_path / "feature_cache",
    )

    # Outcome
    flattened_dataset = add_metadata_to_ds(
        flattened_dataset=flattened_dataset,
        specs=metadata_specs,
    )

    flattened_dataset = add_outcomes_to_ds(
        outcome_specs=outcome_specs,
        flattened_dataset=flattened_dataset,
    )

    flattened_dataset = add_predictors_to_ds(
        temporal_predictor_specs=temporal_predictor_specs,
        static_predictor_specs=static_predictor_specs,
        flattened_dataset=flattened_dataset,
        birthdays=birthdays,
    )

    return flattened_dataset.df


def setup_for_main(
    n_predictors: int,
    feature_sets_path: Path,
    proj_name: str,
) -> tuple[Path, str]:
    """Setup for main.

    Args:
        n_predictors (int): Number of predictors.
        feature_sets_path (Path): Path to feature sets.
        proj_name (str): Name of project.
    Returns:
        tuple[Path, str]: Tuple of project path, and feature_set_id
    """
    # Some predictors take way longer to complete. Shuffling ensures that e.g. the ones that take the longest aren't all
    # at the end of the list.
    proj_path = feature_sets_path / proj_name

    if not proj_path.exists():
        proj_path.mkdir()

    current_user = Path().home().name
    feature_set_id = f"psycop_{proj_name}_{current_user}_{n_predictors}_features_{time.strftime('%Y_%m_%d_%H_%M')}"

    return proj_path, feature_set_id


class ResolvedSpecSet(BaseModel):
    """A set of resolved specs, ready for flattening."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]
    outcomes: list[OutcomeSpec]
    metadata: list[Union[StaticSpec, TemporalSpec]]


class UnresolvedSpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[UnresolvedPredictorSpec]
    static_predictors: list[UnresolvedStaticSpec]
    outcomes: list[UnresolvedOutcomeSpec]
    metadata: list[UnresolvedAnySpec]


def resolve_specifications(
    pre_loaded_dfs: dict[str, pd.DataFrame],
    unresolved_specs: UnresolvedSpecSet,
) -> ResolvedSpecSet:
    resolved_spec_set: dict[str, list[AnySpec]] = defaultdict(list)

    for spec_type, specs in unresolved_specs.__dict__.items():
        for spec in specs:
            resolved_spec_set[spec_type] += [spec.resolve_spec(str2df=pre_loaded_dfs)]

    return ResolvedSpecSet(**resolved_spec_set)


def get_static_predictor_specs():
    """Get static predictor specs."""
    return [
        UnresolvedStaticSpec(
            values_lookup_name="sex_female",
            input_col_name_override="sex_female",
            prefix="pred",
        ),
    ]


def get_unresolved_metadata_specs():
    """Get metadata specs."""
    return [
        UnresolvedStaticSpec(
            values_lookup_name="t2d",
            input_col_name_override="timestamp",
            output_col_name_override="timestamp_first_t2d_hba1c",
        ),
        UnresolvedStaticSpec(
            values_lookup_name="timestamp_exclusion",
            input_col_name_override="timestamp",
            output_col_name_override="timestamp_exclusion",
        ),
        UnresolvedLabPredictorSpec(
            values_lookup_name="hba1c",
            fallback=np.nan,
            lab_values_to_load="numerical_and_coerce",
            interval_days=9999,
            resolve_multiple_fn_name="count",
            allowed_nan_value_prop=0.0,
            prefix="eval",
        ),
    ]


def get_unresolved_outcome_specs():
    """Get outcome specs."""
    return UnresolvedOutcomeGroupSpec(
        values_lookup_name=["t2d"],
        interval_days=[year * 365 for year in (1, 2, 3, 4, 5)],
        resolve_multiple_fn_name=["max"],
        fallback=[0],
        incident=[True],
        allowed_nan_value_prop=[0],
    ).create_combinations()


def get_unresolved_temporal_predictor_specs() -> list[UnresolvedPredictorSpec]:
    """Generate predictor spec list."""
    resolve_multiple = ["max"]  # , "min", "mean", "latest", "count"]
    interval_days = [30]  # , 90, 180, 365, 730]
    allowed_nan_value_prop = [0]

    unresolved_temporal_predictor_specs: list[UnresolvedPredictorSpec] = []

    unresolved_temporal_predictor_specs += UnresolvedLabPredictorGroupSpec(
        values_lookup_name=(
            "hba1c",
            # "alat",
            # "hdl",
            # "ldl",
            # "scheduled_glc",
            # "unscheduled_p_glc",
            # "triglycerides",
            # "fasting_ldl",
            # "crp",
            # "egfr",
            # "albumine_creatinine_ratio",
        ),
        resolve_multiple_fn_name=resolve_multiple,
        interval_days=interval_days,
        fallback=[np.nan],
        lab_values_to_load=["numerical_and_coerce"],
        allowed_nan_value_prop=allowed_nan_value_prop,
    ).create_combinations()

    # unresolved_temporal_predictor_specs += UnresolvedPredictorGroupSpec(
    #     values_lookup_name=(
    #         "essential_hypertension",
    #         "hyperlipidemia",
    #         "polycystic_ovarian_syndrome",
    #         "sleep_apnea",
    #     ),
    #     resolve_multiple_fn_name=resolve_multiple,
    #     interval_days=interval_days,
    #     fallback=[0],
    #     allowed_nan_value_prop=allowed_nan_value_prop,
    # ).create_combinations()

    # unresolved_temporal_predictor_specs += UnresolvedPredictorGroupSpec(
    #     values_lookup_name=("antipsychotics",),
    #     interval_days=interval_days,
    #     resolve_multiple_fn_name=resolve_multiple,
    #     fallback=[0],
    #     allowed_nan_value_prop=allowed_nan_value_prop,
    # ).create_combinations()

    # unresolved_temporal_predictor_specs += UnresolvedPredictorGroupSpec(
    #     values_lookup_name=["weight_in_kg", "height_in_cm", "bmi"],
    #     interval_days=interval_days,
    #     resolve_multiple_fn_name=["latest"],
    #     fallback=[np.nan],
    #     allowed_nan_value_prop=allowed_nan_value_prop,
    # ).create_combinations()

    return unresolved_temporal_predictor_specs


def create_unresolved_specs() -> UnresolvedSpecSet:
    """Create column specifications.

    Resolve when preloading is finished, so don't do it here.
    """
    return UnresolvedSpecSet(
        temporal_predictors=get_unresolved_temporal_predictor_specs(),
        static_predictors=get_static_predictor_specs(),
        outcomes=get_unresolved_outcome_specs(),
        metadata=get_unresolved_metadata_specs(),
    )


def main(
    proj_name: str,
    feature_sets_path: Path,
):
    """Main function for loading, generating and evaluating a flattened
    dataset.

    Args:
        proj_name (str): Name of project.
        feature_sets_path (Path): Path to where feature sets should be stored.
        prediction_time_loader_str (str): Key to lookup in data_loaders registry for prediction time dataframe.
    """
    unresolved_specs = create_unresolved_specs()

    proj_path, feature_set_id = setup_for_main(
        n_predictors=len(unresolved_specs.temporal_predictors),
        feature_sets_path=feature_sets_path,
        proj_name=proj_name,
    )

    out_dir = create_save_dir_path(
        feature_set_id=feature_set_id,
        proj_path=proj_path,
    )

    run = init_wandb(
        wandb_project_name=proj_name,
        predictor_specs=unresolved_specs.temporal_predictors,
        save_dir=out_dir,  # Save-dir as argument because we want to log the path
    )

    pre_loaded_dfs = pre_load_unique_dfs(
        specs=unresolved_specs.static_predictors
        + unresolved_specs.temporal_predictors
        + unresolved_specs.metadata
        + unresolved_specs.outcomes,
    )

    resolved_specs = resolve_specifications(
        pre_loaded_dfs=pre_loaded_dfs,
        unresolved_specs=unresolved_specs,
    )

    flattened_df = create_full_flattened_dataset(
        prediction_times=physical_visits_to_psychiatry(),
        temporal_predictor_specs=resolved_specs.temporal_predictors,
        static_predictor_specs=resolved_specs.static_predictors,
        metadata_specs=resolved_specs.metadata,
        outcome_specs=resolved_specs.outcomes,
        proj_path=proj_path,
        birthdays=birthdays(),
    )

    split_and_save_to_disk(
        flattened_df=flattened_df,
        out_dir=out_dir,
        file_prefix=feature_set_id,
        file_suffix="parquet",
    )

    save_feature_set_description_to_disk(
        predictor_specs=resolved_specs.temporal_predictors
        + resolved_specs.static_predictors,
        flattened_dataset_file_dir=out_dir,
        out_dir=out_dir,
        file_suffix="parquet",
    )

    finish_wandb(
        run=run,
    )


if __name__ == "__main__":
    main(
        feature_sets_path=FEATURE_SETS_PATH,
        proj_name="t2d",
    )
