"""Generator for synth prediction data."""
from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd
from timeseriesflattener.testing.synth_data_generator.synth_col_generators import (
    create_outcome_values,
    generate_data_columns,
)
from timeseriesflattener.testing.synth_data_generator.utils import replace_vals_with_na


def generate_synth_data(
    predictors: Iterable[dict],
    outcome_column_name: str,
    n_samples: int,
    logistic_outcome_model: str,
    intercept: float = 0,
    na_prob: Optional[float] = 0.1,
    na_ignore_cols: Optional[list[str]] = None,
    prob_outcome: Optional[float] = 0.08,
    noise_mean_sd: tuple[float, float] = (0, 1),
) -> pd.DataFrame:
    """Takes a dict and generates synth data from it.

    Args:
        predictors (Iterable[dict]): A dict representing each column. Key is col_name (str), values are column_type (str), output_type (float|int), min (int), max(int).
        outcome_column_name (str): Name of the outcome column.
        n_samples (int): Number of samples (rows) to generate.
        logistic_outcome_model (str): The statistical model used to generate outcome values, e.g. specified as'1*col_name+1*col_name2'.
        intercept (float, optional): The intercept of the logistic outcome model. Defaults to 0.
        na_prob (float, optional): Probability of changing a value in a predictor column
            to NA.
        na_ignore_cols (list[str], optional): Columns to ignore when creating NAs
        prob_outcome (float): Probability of a given row receiving "1" for the outcome.
        noise_mean_sd (tuple[float, float], optional): mean and sd of the noise.
            Increase SD to obtain more uncertain models.

    Returns:
        pd.DataFrame: The synthetic dataset
    """

    cols_to_init = []

    for pred_spec in predictors:
        for col_name, _ in pred_spec.items():
            cols_to_init.append(col_name)

    # Initialise dataframe
    df = pd.DataFrame(columns=list(cols_to_init))

    # Generate data
    df = generate_data_columns(predictors=predictors, n_samples=n_samples, df=df)

    # Sigmoid it to get probabilities with mean = 0.5
    df[outcome_column_name] = create_outcome_values(
        n_samples=n_samples,
        logistic_outcome_model=logistic_outcome_model,
        intercept=intercept,
        noise_mean_sd=noise_mean_sd,
        df=df,
    )

    df[outcome_column_name] = np.where(df[outcome_column_name] < prob_outcome, 1, 0)

    # randomly replace predictors with NAs
    if na_prob:
        df = replace_vals_with_na(na_prob=na_prob, na_ignore_cols=na_ignore_cols, df=df)

    return df.reset_index(drop=True)
