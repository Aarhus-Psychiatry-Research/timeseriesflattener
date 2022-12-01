"""Testing of the DiskCache class."""
import numpy as np
import pandas as pd

from timeseriesflattener.feature_cache.cache_to_disk import DiskCache
from timeseriesflattener.feature_spec_objects import PredictorSpec
from timeseriesflattener.resolve_multiple_functions import latest


def test_write_and_check_feature(tmp_path):
    """Test that write_feature writes a feature to disk."""

    cache = DiskCache(
        feature_cache_dir=tmp_path,
        pred_time_uuid_col_name="pred_time_uuid",
        cache_file_suffix="csv",
        prediction_times_df=pd.DataFrame(
            {"uuid": [1, 2, 3], "pred_time_uuid": [1, 2, 3]}
        ),
    )

    values_df = pd.DataFrame(
        {
            "dw_ek_borger": [1, 2, 3],
            "pred_time_uuid": [1, 2, 3],
            "timestamp": [1, 2, 3],
        }
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
            "dw_ek_borger": [1, 2, 3],
            "pred_time_uuid": [1, 2, 3],
            "timestamp": [1, 2, 3],
            f"{test_spec.get_col_str()}": [1, 2, 3],
        }
    )

    assert cache.feature_exists(feature_spec=test_spec) is False

    cache.write_feature(feature_spec=test_spec, df=generated_df)

    assert cache.feature_exists(feature_spec=test_spec) is True


def test_read_feature(tmp_path):
    """Test that read_feature writes a feature to disk.

    Important that one row contains the fallback because we then test removing fallback vals when saving and expanding them again when reading."""

    cache = DiskCache(
        feature_cache_dir=tmp_path,
        pred_time_uuid_col_name="pred_time_uuid",
        cache_file_suffix="csv",
        prediction_times_df=pd.DataFrame(
            {"uuid": [1, 2, 3], "pred_time_uuid": [1, 2, 3]}
        ),
    )

    values_df = pd.DataFrame(
        {
            "dw_ek_borger": [1, 2, 3],
            "pred_time_uuid": [1, 2, 3],
            "timestamp": [1, 2, 3],
        }
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
            "dw_ek_borger": [1, 2, 3],
            "pred_time_uuid": [1, 2, 3],
            "timestamp": [1, 2, 3],
            f"{test_spec.get_col_str()}": [1, 2, np.nan],
        }
    )

    cache.write_feature(feature_spec=test_spec, df=generated_df)

    df = cache.read_feature(feature_spec=test_spec)

    assert df.shape[0] == 3
    assert df[df["timestamp"].isna()].shape[0] == 1
