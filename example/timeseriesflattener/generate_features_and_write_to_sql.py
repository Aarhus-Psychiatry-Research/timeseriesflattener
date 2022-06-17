import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopmlutils.loaders  # noqa
from psycopmlutils.timeseriesflattener import (
    FlattenedDataset,
    create_feature_combinations,
)
from psycopmlutils.writers.sql_writer import write_df_to_sql
from wasabi import msg

if __name__ == "__main__":
    RESOLVE_MULTIPLE = ["mean", "max", "min"]
    LOOKBEHIND_DAYS = [365, 730, 1825, 9999]

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

    event_times = psycopmlutils.loaders.LoadOutcome.t2d()

    msg.info(f"Generating {len(PREDICTOR_LIST)} features")

    msg.info("Loading prediction times")
    prediction_times = psycopmlutils.loaders.LoadVisits.physical_visits_to_psychiatry()

    msg.info("Initialising flattened dataset")
    flattened_df = FlattenedDataset(prediction_times_df=prediction_times, n_workers=20)

    # Outcome
    msg.info("Adding outcome")
    for i in [0.5, 1, 2, 3, 4, 5]:
        lookahead_days = i * 365.25
        msg.info(f"Adding outcome with {lookahead_days} days of lookahead")
        flattened_df.add_temporal_outcome(
            outcome_df=event_times,
            lookahead_days=lookahead_days,
            resolve_multiple="max",
            fallback=0,
            outcome_df_values_col_name="value",
            new_col_name="t2d",
            incident=True,
            dichotomous=True,
        )
        msg.good("Finished adding outcome")

    # Predictors
    msg.info("Adding static predictors")
    flattened_df.add_static_predictor(psycopmlutils.loaders.LoadDemographic.male())
    flattened_df.add_age(psycopmlutils.loaders.LoadDemographic.birthdays())

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

    # Split and upload to SQL_server
    midtx_path = Path("\\\\tsclient\\X\\MANBER01\\documentLibrary")

    splits = ["test", "val", "train"]

    outcome_col_name = "t2d_within_1826.25_days_max_fallback_0"

    flattened_df_ids = flattened_df.df["dw_ek_borger"].unique()

    for dataset_name in splits:
        ROWS_PER_CHUNK = 5_000

        df_split_ids = psycopmlutils.loaders.LoadIDs.load(split=dataset_name)

        # Find IDs which are in split_ids, but not in flattened_df.
        split_ids = df_split_ids["dw_ek_borger"].unique()
        flattened_df_ids = flattened_df.df["dw_ek_borger"].unique()

        ids_in_split_but_not_in_flattened_df = split_ids[
            ~np.isin(split_ids, flattened_df_ids)
        ]

        msg.warn(
            f"{dataset_name}: There are {len(ids_in_split_but_not_in_flattened_df)} ({round(len(ids_in_split_but_not_in_flattened_df)/len(split_ids)*100, 2)}%) ids which are in {dataset_name}_ids but not in flattened_df_ids, will get dropped during merge"
        )

        split_df = pd.merge(flattened_df.df, df_split_ids, how="inner")

        msg.info(f"{dataset_name}: Writing to SQL")
        write_df_to_sql(
            df=split_df,
            table_name=f"psycop_t2d_{dataset_name}",
            if_exists="replace",
            rows_per_chunk=ROWS_PER_CHUNK,
        )
        msg.good(f"{dataset_name}: Succesfully wrote {dataset_name} to SQL server")
