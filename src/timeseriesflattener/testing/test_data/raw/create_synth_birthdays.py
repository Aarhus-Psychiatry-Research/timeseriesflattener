from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
from timeseriesflattener.testing.load_synth_data import TEST_DATA_PATH, load_synth_prediction_times

if __name__ == "__main__":
    prediction_times = load_synth_prediction_times()

    min_lifespan = [
        dt.timedelta(days=np.random.randint(365 * 19, 365 * 70))
        for _ in range(prediction_times["entity_id"].n_unique())
    ]

    birthdays = (
        prediction_times.group_by("entity_id")
        .agg(pl.col("timestamp").min().alias("min_date"))
        .with_columns((pl.col("min_date") - pl.Series(min_lifespan)).alias("birthday"))
        .drop("min_date")
    )

    birthdays.write_csv(TEST_DATA_PATH / "raw" / "synth_birthdays.csv")
