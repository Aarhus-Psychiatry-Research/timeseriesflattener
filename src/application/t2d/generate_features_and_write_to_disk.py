"""Main example on how to generate features.

Uses T2D-features. WIP, will be migrated to psycop-t2d when reaching
maturity.
"""

import sys
import tempfile
import time
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Optional, Union

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
from psycop_feature_generation.loaders.raw.load_demographic import birthdays
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    AnySpec,
    BaseModel,
    OutcomeGroupSpec,
    OutcomeSpec,
    PredictorGroupSpec,
    PredictorSpec,
    StaticSpec,
    TemporalSpec,
)
from psycop_feature_generation.timeseriesflattener.flattened_dataset import (
    FlattenedDataset,
)
from psycop_feature_generation.utils import (
    FEATURE_SETS_PATH,
    N_WORKERS,
    PROJECT_ROOT,
    write_df_to_file,
)

LOOKAHEAD_YEARS = (1, 2, 3, 4, 5)
msg = Printer(timestamp=True)


def init_wandb(
    wandb_project_name: str,
    predictor_specs: Sequence[PredictorSpec],
    save_dir: Union[Path, str],
) -> None:
    """Initialise wandb logging. Allows to use wandb to track progress, send
    Slack notifications if failing, and track logs.

    Args:
        wandb_project_name (str): Name of wandb project.
        predictor_specs (Iterable[dict[str, Any]]): List of predictor specs.
        save_dir (Union[Path, str]): Path to save dir.
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

    wandb.init(project=wandb_project_name, config=feature_settings)


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


def split_and_save_dataset_to_disk(
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
        specs (list[AnySpec]): List of specifications.
        flattened_dataset (FlattenedDataset): Flattened dataset.

    Returns:
        FlattenedDataset: Flattened dataset.
    """
    msg.info("Adding metadata to dataset")

    for spec in specs:
        msg.info(f"Adding metadata from {spec.feature_name}")
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
        outcome_specs (list[OutcomeSpec]): List of outcome specifications.

    Returns:
        FlattenedDataset: Flattened dataset.
    """
    msg.info("Adding outcomes to dataset")

    for spec in outcome_specs:
        msg.info(f"Adding outcome with {spec.interval_days} days of lookahead")
        flattened_dataset.add_temporal_outcome(
            output_spec=spec,
        )

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

    flattened_dataset.add_age_and_birth_year(
        id2date_of_birth=birthdays, birth_year_as_predictor=True
    )

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


def resolve_spec_set_component(spec_set_component: dict[str, Callable]):
    for k, v in spec_set_component.items():
        spec_set_component[k] = v()

    return spec_set_component


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]
    outcomes: list[OutcomeSpec]
    metadata: list[AnySpec]


def create_flattened_dataset(
    prediction_times: pd.DataFrame,
    birthdays: pd.DataFrame,
    spec_set: SpecSet,
    proj_path: Path,
) -> pd.DataFrame:
    """Create flattened dataset.

    Args:
        prediction_times (pd.DataFrame): Dataframe with prediction times.
        birthdays (pd.DataFrame): Birthdays. Used for inferring age at each prediction time.
        spec_set (SpecSet): Set of specifications.
        proj_path (Path): Path to project directory.

    Returns:
        FlattenedDataset: Flattened dataset.
    """

    msg.info(f"Generating {len(spec_set.temporal_predictors)} features")

    msg.info("Initialising flattened dataset")

    flattened_dataset = FlattenedDataset(
        prediction_times_df=prediction_times,
        n_workers=min(
            len(spec_set.temporal_predictors),
            30,
        ),
        feature_cache_dir=proj_path / "feature_cache",
    )

    flattened_dataset = add_predictors_to_ds(
        temporal_predictor_specs=spec_set.temporal_predictors,
        static_predictor_specs=spec_set.static_predictors,
        flattened_dataset=flattened_dataset,
        birthdays=birthdays,
    )

    flattened_dataset = add_metadata_to_ds(
        flattened_dataset=flattened_dataset,
        specs=spec_set.metadata,
    )

    flattened_dataset = add_outcomes_to_ds(
        outcome_specs=spec_set.outcomes,
        flattened_dataset=flattened_dataset,
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
    proj_path = feature_sets_path / proj_name

    if not proj_path.exists():
        proj_path.mkdir()

    current_user = Path().home().name
    feature_set_id = f"psycop_{proj_name}_{current_user}_{n_predictors}_features_{time.strftime('%Y_%m_%d_%H_%M')}"

    return proj_path, feature_set_id


def get_static_predictor_specs():
    """Get static predictor specs."""
    return [
        StaticSpec(
            values_loader="sex_female",
            input_col_name_override="sex_female",
            prefix="pred",
        ),
    ]


def get_metadata_specs() -> list[AnySpec]:
    """Get metadata specs."""
    metadata_specs = [
        StaticSpec(
            values_loader="t2d",
            input_col_name_override="timestamp",
            output_col_name_override="timestamp_first_t2d_hba1c",
        ),
        StaticSpec(
            values_loader="timestamp_exclusion",
            input_col_name_override="timestamp",
            output_col_name_override="timestamp_exclusion",
        ),
        PredictorSpec(
            values_loader="hba1c",
            fallback=np.nan,
            interval_days=9999,
            resolve_multiple_fn="count",
            allowed_nan_value_prop=0.0,
            prefix="eval",
        ),
    ]

    metadata_specs += OutcomeGroupSpec(
        values_loader=["hba1c"],
        interval_days=[year * 365 for year in LOOKAHEAD_YEARS],
        resolve_multiple_fn=["count"],
        fallback=[0],
        incident=[False],
        allowed_nan_value_prop=[0.0],
        prefix="eval",
    ).create_combinations()

    return metadata_specs


def get_outcome_specs():
    """Get outcome specs."""
    return OutcomeGroupSpec(
        values_loader=["t2d"],
        interval_days=[year * 365 for year in LOOKAHEAD_YEARS],
        resolve_multiple_fn=["max"],
        fallback=[0],
        incident=[True],
        allowed_nan_value_prop=[0],
    ).create_combinations()


def resolve_group_spec(group_spec: Union[PredictorGroupSpec, OutcomeGroupSpec]):
    return group_spec.create_combinations()


def get_temporal_predictor_specs() -> list[PredictorSpec]:
    """Generate predictor spec list."""
    base_resolve_multiple = ["max", "min", "mean", "latest", "count"]
    base_interval_days = [30, 90, 180, 365, 730]
    base_allowed_nan_value_prop = [0]

    temporal_predictor_groups = [
        PredictorGroupSpec(
            values_loader=(
                "alat",
                "hdl",
                "ldl",
                "triglycerides",
                "fasting_ldl",
                "crp",
            ),
            resolve_multiple_fn=base_resolve_multiple,
            interval_days=base_interval_days,
            fallback=[np.nan],
            allowed_nan_value_prop=base_allowed_nan_value_prop,
        ),
        PredictorGroupSpec(
            values_loader=(
                "hba1c",
                "scheduled_glc",
                "unscheduled_p_glc",
                "egfr",
                "albumine_creatinine_ratio",
            ),
            resolve_multiple_fn=base_resolve_multiple,
            interval_days=base_interval_days,
            fallback=[np.nan],
            allowed_nan_value_prop=base_allowed_nan_value_prop,
        ),
        PredictorGroupSpec(
            values_loader=(
                "essential_hypertension",
                "hyperlipidemia",
                "polycystic_ovarian_syndrome",
                "sleep_apnea",
                "gerd",
            ),
            resolve_multiple_fn=base_resolve_multiple,
            interval_days=base_interval_days,
            fallback=[0],
            allowed_nan_value_prop=base_allowed_nan_value_prop,
        ),
        PredictorGroupSpec(
            values_loader=(
                "f0_disorders",
                "f1_disorders",
                "f2_disorders",
                "f3_disorders",
                "f4_disorders",
                "f5_disorders",
                "f6_disorders",
                "f7_disorders",
                "f8_disorders",
                "hyperkinetic_disorders",
            ),
            resolve_multiple_fn=base_resolve_multiple,
            interval_days=base_interval_days,
            fallback=[0],
            allowed_nan_value_prop=base_allowed_nan_value_prop,
        ),
        PredictorGroupSpec(
            values_loader=(
                "antipsychotics",
                "clozapine",
                "top_10_weight_gaining_antipsychotics",
                "lithium",
                "valproate",
                "lamotrigine",
                "benzodiazepines",
                "pregabaline",
                "ssri",
                "snri",
                "tca",
                "selected_nassa",
                "benzodiazepine_related_sleeping_agents",
            ),
            interval_days=base_interval_days,
            resolve_multiple_fn=base_resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=base_allowed_nan_value_prop,
        ),
        PredictorGroupSpec(
            values_loader=(
                "gerd_drugs",
                "statins",
                "antihypertensives",
                "diuretics",
            ),
            interval_days=base_interval_days,
            resolve_multiple_fn=base_resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=base_allowed_nan_value_prop,
        ),
        PredictorGroupSpec(
            values_loader=["weight_in_kg", "height_in_cm", "bmi"],
            interval_days=base_interval_days,
            resolve_multiple_fn=["latest"],
            fallback=[np.nan],
            allowed_nan_value_prop=base_allowed_nan_value_prop,
        ),
    ]

    with Pool(min(N_WORKERS, len(temporal_predictor_groups))) as p:
        temporal_predictor_specs: list[PredictorSpec] = p.map(
            func=resolve_group_spec, iterable=temporal_predictor_groups
        )

        # Unpack list of lists
        temporal_predictor_specs = [
            item for sublist in temporal_predictor_specs for item in sublist
        ]

    return temporal_predictor_specs


def main(
    proj_name: str,
    feature_sets_path: Path,
):
    """Main function for loading, generating and evaluating a flattened
    dataset.

    Args:
        proj_name (str): Name of project.
        feature_sets_path (Path): Path to where feature sets should be stored.
    """
    spec_set = SpecSet(
        temporal_predictors=get_temporal_predictor_specs(),
        static_predictors=get_static_predictor_specs(),
        outcomes=get_outcome_specs(),
        metadata=get_metadata_specs(),
    )

    proj_path, feature_set_id = setup_for_main(
        n_predictors=len(spec_set.temporal_predictors),
        feature_sets_path=feature_sets_path,
        proj_name=proj_name,
    )

    out_dir = create_save_dir_path(
        feature_set_id=feature_set_id,
        proj_path=proj_path,
    )

    init_wandb(
        wandb_project_name=proj_name,
        predictor_specs=spec_set.temporal_predictors,
        save_dir=out_dir,  # Save-dir as argument because we want to log the path
    )

    flattened_df = create_flattened_dataset(
        prediction_times=physical_visits_to_psychiatry(),
        spec_set=spec_set,
        proj_path=proj_path,
        birthdays=birthdays(),
    )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df,
        out_dir=out_dir,
        file_prefix=feature_set_id,
        file_suffix="parquet",
    )

    save_feature_set_description_to_disk(
        predictor_specs=spec_set.temporal_predictors + spec_set.static_predictors,
        flattened_dataset_file_dir=out_dir,
        out_dir=out_dir,
        file_suffix="parquet",
    )

    wandb.log_artifact("poetry.lock", name="poetry_lock_file", type="poetry_lock")


if __name__ == "__main__":
    main(
        feature_sets_path=FEATURE_SETS_PATH,
        proj_name="t2d",
    )
