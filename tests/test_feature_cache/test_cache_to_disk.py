"""Testing of the DiskCache class."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from timeseriesflattener.feature_cache.cache_to_disk import DiskCache
from timeseriesflattener.feature_spec_objects import PredictorSpec
from timeseriesflattener.resolve_multiple_functions import latest


def test_write_and_check_feature(
    tmp_path: Path,
):
    """Test that write_feature writes a feature to disk."""

    cache = DiskCache(
        feature_cache_dir=tmp_path,
        pred_time_uuid_col_name="pred_time_uuid",
        entity_id_col_name="entity_id",
        cache_file_suffix="csv",
        prediction_times_df=pd.DataFrame(
            {"uuid": [1, 2, 3], "pred_time_uuid": [1, 2, 3]},
        ),
    )

    values_df = pd.DataFrame(
        {
            "entity_id": [1, 2, 3],
            "pred_time_uuid": [1, 2, 3],
            "timestamp": [1, 2, 3],
            "value": [1, 2, 3],
        },
    )

    test_spec = PredictorSpec(
        values_df=values_df,
        lookbehind_days=5,
        resolve_multiple_fn=latest,
        key_for_resolve_multiple="latest",
        fallback=np.nan,
        feature_name="test_feature",
    )

    generated_df = pd.DataFrame(
        {
            "entity_id": [1, 2, 3],
            "pred_time_uuid": [1, 2, 3],
            "timestamp": [1, 2, 3],
            f"{test_spec.get_col_str()}": [1, 2, 3],
        },
    )

    assert cache.feature_exists(feature_spec=test_spec) is False

    cache.write_feature(feature_spec=test_spec, df=generated_df)

    assert cache.feature_exists(feature_spec=test_spec) is True


def test_read_feature(tmp_path: Path):
    """Test that read_feature reads a feature from disk.

    Important that one row contains the fallback because we then test
    removing fallback vals when saving and expanding them again when
    reading.
    """

    # Note that initialisation is much simpler i flattened dataset, since
    # many of the col names are specified in the instantiation of the
    # flattened dataset, and passed along to the cache.
    cache = DiskCache(
        feature_cache_dir=tmp_path,
        pred_time_uuid_col_name="pred_time_uuid",
        entity_id_col_name="entity_id",
        timestamp_col_name="timestamp",
        cache_file_suffix="csv",
        prediction_times_df=pd.DataFrame(
            {"pred_time_uuid": [1, 2, 3], "entity_id": [1, 2, 3]},
        ),
    )

    values_df = pd.DataFrame(
        {
            "entity_id": [1, 2, 3, 4, 5],
            "timestamp": [1, 2, 3, 4, 5],
            "value": [1, 2, 3, 4, 5],
        },
    )

    test_spec = PredictorSpec(
        values_df=values_df,
        interval_days=5,
        resolve_multiple_fn=latest,
        key_for_resolve_multiple="latest",
        fallback=np.nan,
        feature_name="test_feature",
    )

    generated_df = pd.DataFrame(
        {
            "entity_id": [
                1,
                2,
                3,
            ],
            "pred_time_uuid": [1, 2, 3],
            "timestamp": [1, 2, 3],
            f"{test_spec.get_col_str()}": [1, 2, np.nan],
        },
    )

    cache.write_feature(feature_spec=test_spec, df=generated_df)

    df = cache.read_feature(feature_spec=test_spec)

    # For each column in df, check that the values are equal to generated_df
    for col in df.columns:
        assert_frame_equal(
            df[col].to_frame(),
            generated_df[col].to_frame(),
        )
