"""Tests of the feature describer module."""
import numpy as np
import pandas as pd
import pytest

from psycop_feature_generation.data_checks.flattened.feature_describer import (
    generate_feature_description_df,
    generate_feature_description_row,
)
from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    PredictorSpec,
)
from psycop_feature_generation.utils import PROJECT_ROOT, generate_feature_colname

# pylint: disable=redefined-outer-name, missing-function-docstring


@pytest.fixture()
def predictor_specs(df):
    return [
        PredictorSpec(
            values_df=pd.DataFrame({"hba1c": [0]}),
            interval_days=100,
            resolve_multiple_fn_name="max",
            fallback=np.nan,
            feature_name="hba1c",
        ),
    ]


@pytest.fixture()
def df():
    """Load the synthetic flattened data set."""
    return pd.read_csv(
        PROJECT_ROOT
        / "tests/test_data/flattened/generated_with_outcome/synth_flattened_with_outcome.csv",
    )


def test_load_dataset(df):
    """Check loading of synthetic dataset."""
    assert df.shape[0] == 10_000


def test_generate_feature_description_row(
    df: pd.DataFrame,
    predictor_specs: list[PredictorSpec],
):
    spec = predictor_specs[0]

    column_name = spec.get_col_str()

    generate_feature_description_row(series=df[column_name], predictor_spec=spec)

    generate_feature_description_df(df=df, predictor_specs=predictor_specs)
