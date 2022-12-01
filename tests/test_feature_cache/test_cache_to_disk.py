import pandas as pd

from timeseriesflattener.feature_cache.cache_to_disk import DiskCache


def test_write_feature(tmp_path):
    """Test that write_feature writes a feature to disk."""

    cache = DiskCache(
        feature_cache_dir=tmp_path,
        pred_time_uuid_col_name="pred_time_uuid",
        cache_file_suffix="csv",
        prediction_times_df=pd.DataFrame(
            {"uuid": [1, 2, 3], "pred_time_uuid": [1, 2, 3]}
        ),
    )
