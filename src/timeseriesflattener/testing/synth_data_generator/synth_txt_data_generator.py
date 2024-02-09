"""Script for creating synthetic text data for testing purposes.

Produces a .csv file with the following columns: citizen_id, timestamp,
text.
"""
from pathlib import Path
from typing import Optional

import pandas as pd
from timeseriesflattener.testing.synth_data_generator.synth_col_generators import (
    generate_data_columns,
)
from timeseriesflattener.testing.synth_data_generator.utils import replace_vals_with_na


def generate_synth_txt_data(
    predictors: dict,
    n_samples: int,
    na_prob: Optional[float] = 0.1,
    na_ignore_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Takes a dict and generates synth data from it.

    Args:
        predictors (dict): A dict representing each column. Key is col_name (str), values are column_type (str), output_type (float|int), min (int), max(int).
        n_samples (int): Number of samples (rows) to generate.
        na_prob (float): Probability of changing a value in a predictor column to NA.
        na_ignore_cols (list[str]): Columns to ignore when creating NAs

    Returns:
        pd.DataFrame: The synthetic dataset
    """

    # Initialise dataframe
    df = pd.DataFrame(columns=list(predictors.keys()))

    # Generate data
    df = generate_data_columns(predictors=predictors, n_samples=n_samples, df=df)

    # randomly replace predictors with NAs
    if na_prob:
        df = replace_vals_with_na(df=df, na_prob=na_prob, na_ignore_cols=na_ignore_cols)

    return df


if __name__ == "__main__":
    column_specifications = {
        "citizen_ids": {"column_type": "uniform_int", "min": 0, "max": 1_200_000},
        "timestamp": {"column_type": "datetime_uniform", "min": 0, "max": 5 * 365},
        "text": {"column_type": "text"},
    }

    out_df = generate_synth_txt_data(predictors=column_specifications, n_samples=100)

    save_path = Path(__file__).parent.parent.parent.parent
    out_df.to_csv(save_path / "tests" / "test_data" / "synth_txt_data.csv")
