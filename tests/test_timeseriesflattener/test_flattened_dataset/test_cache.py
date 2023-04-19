"""Test that cache hits."""


from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
from timeseriesflattener.feature_cache.cache_to_disk import DiskCache
from timeseriesflattener.feature_spec_objects import PredictorGroupSpec, PredictorSpec
from timeseriesflattener.testing.load_synth_data import (
    load_synth_prediction_times,
)

from tests.test_timeseriesflattener.test_flattened_dataset.utils import (
    check_dfs_have_same_contents_by_column,
    create_flattened_df,
)

base_float_predictor_combinations = PredictorGroupSpec(
    values_loader=["synth_predictor_float"],
    lookbehind_days=[365, 730],
    resolve_multiple_fn=["mean"],
    fallback=[np.NaN],
    allowed_nan_value_prop=[0.0],
).create_combinations()

base_binary_predictor_combinations = PredictorGroupSpec(
    values_loader=["synth_predictor_binary"],
    lookbehind_days=[365, 730],
    resolve_multiple_fn=["max"],
    fallback=[np.NaN],
    allowed_nan_value_prop=[0.0],
).create_combinations()


@pytest.mark.parametrize(
    "predictor_specs",
    [base_float_predictor_combinations, base_binary_predictor_combinations],
)
def test_cache_hitting(
    tmp_path: Path,
    predictor_specs: List[PredictorSpec],
    synth_prediction_times: pd.DataFrame,
):
    """Test that cache hits."""

    cache = DiskCache(
        feature_cache_dir=tmp_path,
        entity_id_col_name="entity_id",
        prediction_times_df=synth_prediction_times,
    )

    # Create the cache
    first_df = create_flattened_df(
        predictor_specs=predictor_specs.copy(),
        prediction_times_df=synth_prediction_times,
        cache=cache,
    )

    # Load the cache
    cache_df = create_flattened_df(
        predictor_specs=predictor_specs.copy(),
        prediction_times_df=synth_prediction_times,
        cache=cache,
    )

    # Assert that each column has the same contents
    check_dfs_have_same_contents_by_column(df1=first_df, df2=cache_df)

    # If cache_df doesn't hit the cache, it creates its own files
    # Thus, number of files is an indicator of whether the cache was hit
    assert len(list(tmp_path.glob("*"))) == len(predictor_specs)


if __name__ == "__main__":
    test_cache_hitting(
        tmp_path=Path("tmp"),
        synth_prediction_times=load_synth_prediction_times(),
        predictor_specs=base_float_predictor_combinations,
    )
