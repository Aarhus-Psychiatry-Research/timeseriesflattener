"""Test that cache hits."""


from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from timeseriesflattener.aggregation_fns import maximum, mean
from timeseriesflattener.feature_cache.cache_to_disk import DiskCache
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import PredictorSpec
from timeseriesflattener.testing.load_synth_data import (
    load_synth_prediction_times,
    load_synth_predictor_float,
    synth_predictor_binary,
)
from timeseriesflattener.tests.test_timeseriesflattener.test_flattened_dataset.utils import (
    check_dfs_have_same_contents_by_column,
    create_flattened_df,
)

base_float_predictor_combinations = PredictorGroupSpec(
    named_dataframes=[
        NamedDataframe(df=load_synth_predictor_float(), name="synth_predictor_float"),
    ],
    lookbehind_days=[365, 730],
    aggregation_fns=[mean],
    fallback=[np.NaN],
).create_combinations()

base_binary_predictor_combinations = PredictorGroupSpec(
    named_dataframes=[
        NamedDataframe(df=synth_predictor_binary(), name="synth_predictor_binary"),
    ],
    lookbehind_days=[365, 730],
    aggregation_fns=[maximum],
    fallback=[np.NaN],
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
