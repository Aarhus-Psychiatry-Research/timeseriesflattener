import time

import pandas as pd
import numpy as np
import psycopmlutils.loaders  # noqa
from psycopmlutils.timeseriesflattener import (
    FlattenedDataset,
    create_feature_combinations,
)
from pathlib import Path
from wasabi import msg

if __name__ == "__main__":
    RESOLVE_MULTIPLE = ["mean", "latest", "earliest", "max", "min"]
    LOOKBEHIND_DAYS = [365, 9999]

    PREDICTOR_LIST = create_feature_combinations(
        [
            {
                "predictor_df": "hba1c",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
            },
            {
                "predictor_df": "alat",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
            },
            {
                "predictor_df": "hdl",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
            },
            {
                "predictor_df": "ldl",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": np.nan,
            },
        ]
    )

    event_times = psycopmlutils.loaders.LoadDiagnoses.t2d_times()

    msg.info(f"Generating {len(PREDICTOR_LIST)} features")

    msg.info("Loading prediction times")
    prediction_times = psycopmlutils.loaders.LoadVisits.physical_visits_to_psychiatry()

    msg.info("Initialising flattened dataset")
    flattened_df = FlattenedDataset(prediction_times_df=prediction_times, n_workers=20)

    # Outcome
    msg.info("Adding outcome")
    flattened_df.add_temporal_outcome(
        outcome_df=event_times,
        lookahead_days=365.25 * 5,
        resolve_multiple="max",
        fallback=0,
        outcome_df_values_col_name="value",
        new_col_name="t2d",
    )
    msg.good("Finished adding outcome")

    # Predictors
    msg.info("Adding static predictors")
    flattened_df.add_static_predictor(psycopmlutils.loaders.LoadDemographics.male())
    flattened_df.add_age(psycopmlutils.loaders.LoadDemographics.birthdays())

    start_time = time.time()

    msg.info("Adding temporal predictors")
    flattened_df.add_temporal_predictors_from_list_of_argument_dictionaries(
        predictors=PREDICTOR_LIST,
    )

    end_time = time.time()
    msg.good(
        f"Finished adding {len(PREDICTOR_LIST)} predictors, took {round((end_time - start_time)/60, 1)} minutes"
    )

    msg.info(
        f"Dataframe size is {flattened_df.df.memory_usage(index=True, deep=True).sum() / 1024 / 1024} MiB"
    )

    msg.good("Done!")

    midtx_path = Path("\\\\tsclient\\X\\MANBER01\\documentLibrary")

    splits = ["train", "test", "val"]

    outcome_col_name = "t2d_within_1826.25_days_max_fallback_0"

    flattened_df_ids = flattened_df.df["dw_ek_borger"].unique()

    for dataset_name in splits:
        split_id_path = midtx_path / "train-test-splits" / f"{dataset_name}_ids.csv"
        df_split_ids = pd.read_csv(split_id_path)
        split_ids = df_split_ids["dw_ek_borger"].unique()

        ids_in_split_but_not_in_flattened_df = split_ids[
            ~np.isin(split_ids, flattened_df_ids)
        ]

        msg.warn(
            f"There are {len(ids_in_split_but_not_in_flattened_df)} ({round(len(ids_in_split_but_not_in_flattened_df)/len(split_ids)*100, 2)}%) ids which are in {dataset_name}_ids but not in flattened_df_ids, will get dropped during merge"
        )

        split_df = pd.merge(flattened_df.df, df_split_ids, how="inner")

        split_features = split_df.loc[:, ~split_df.columns.str.startswith("t2d")]
        split_events = split_df[["dw_ek_borger", "timestamp", outcome_col_name]]

        base_path = midtx_path / "feature_generation" / "toy_example"

        msg.info("Writing {dataset_name}_features csv")
        feature_path = base_path / f"{dataset_name}_features.csv"
        split_features.to_csv(feature_path)

        msg.info("Writing {dataset_name}_events csv")
        event_path = base_path / f"{dataset_name}_events.csv"
        split_events.to_csv(event_path)
