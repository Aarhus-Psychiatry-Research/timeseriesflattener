import datetime as dt
import os
from pathlib import Path
from typing import Any

import pandas as pd

from timeseriesflattener.feature_cache.abstract_feature_cache import FeatureCache
from timeseriesflattener.feature_spec_objects import TemporalSpec
from timeseriesflattener.utils import load_dataset_from_file, write_df_to_file


class DiskCache(FeatureCache):
    def __init__(
        self,
        feature_cache_dir: Path,
        pred_time_uuid_col_name: str,
        cache_file_suffix: str,
        prediction_times_df: pd.DataFrame,
        validate: bool = True,
    ):
        """Initialize DiskCache."""

        self.feature_cache_dir = feature_cache_dir
        self.cache_file_suffix = cache_file_suffix
        self.prediction_times_df = prediction_times_df
        self.pred_time_uuid_col_name = pred_time_uuid_col_name

        if not self.feature_cache_dir.exists():
            self.feature_cache_dir.mkdir()

        self.validate = validate

    def _load_most_recent_df_matching_pattern(
        self,
        file_pattern: str,
    ) -> pd.DataFrame:
        """Load most recent df matching pattern.

        Args:
            file_pattern (str): Pattern to match
            file_suffix (str, optional): File suffix to match.

        Returns:
            DataFrame: DataFrame matching pattern

        Raises:
            FileNotFoundError: If no file matching pattern is found
        """
        files_with_suffix = list(self.feature_cache_dir.glob(file_pattern))

        if len(files_with_suffix) == 0:
            raise FileNotFoundError(f"No files matching pattern {file_pattern} found")

        path_of_most_recent_file = max(files_with_suffix, key=os.path.getctime)

        return load_dataset_from_file(
            file_path=path_of_most_recent_file,
        )

    def read_feature(self, feature_spec: TemporalSpec) -> pd.DataFrame:
        """Load most recent df matching pattern, and expand fallback column.

        Args:
            feature_spec (AnySpec): Feature spec

        Returns:
            DataFrame: DataFrame with fallback column expanded
        """
        df = self._load_most_recent_df_matching_pattern(
            file_pattern=self._get_file_pattern(feature_spec=feature_spec),
        )

        # Expand fallback column
        df = pd.merge(
            left=self.prediction_times_df[self.pred_time_uuid_col_name],
            right=df,
            how="left",
            on=self.pred_time_uuid_col_name,
            validate="m:1",
        )

        df[feature_spec.get_col_str()] = df[feature_spec.get_col_str()].fillna(
            feature_spec.fallback
        )

        return df

    def write_feature(
        self,
        feature_spec: TemporalSpec,
        df: pd.DataFrame,
    ):
        """Write feature to cache."""
        n_uuids = df[self.pred_time_uuid_col_name].nunique()
        file_name = f"{feature_spec.get_col_str()}_{n_uuids}_uuids"

        # Drop rows containing fallback, since it's non-informative
        df = df[df[feature_spec.get_col_str()] != feature_spec.fallback].dropna()

        # Write df to cache
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Write df to cache
        write_df_to_file(
            df=df,
            file_path=self.feature_cache_dir
            / f"{file_name}_{timestamp}.{self.cache_file_suffix}",
        )

    def feature_exists(
        self,
        feature_spec: TemporalSpec,
    ) -> bool:
        """Check if cache is hit.

        Args:
            feature_spec (AnySpec): Feature spec

        Returns:
            bool: True if cache is hit, False otherwise
        """
        file_pattern = self._get_file_pattern(feature_spec=feature_spec)

        # Check that file exists
        file_pattern_hits = list(
            self.feature_cache_dir.glob(file_pattern),
        )

        if len(file_pattern_hits) == 0:
            return False

        return True

    def _get_file_name(
        self,
        feature_spec: TemporalSpec,
    ) -> str:
        """Get file name for feature spec.

        Args:
            feature_spec (AnySpec): Feature spec

        Returns:
            str: File name
        """
        n_uuids = feature_spec.values_df[self.pred_time_uuid_col_name].nunique()

        return f"{feature_spec.get_col_str()}_{n_uuids}_uuids"

    def _get_file_pattern(
        self,
        feature_spec: TemporalSpec,
    ) -> str:
        """Get file pattern for feature spec.

        Args:
            feature_spec (AnySpec): Feature spec

        Returns:
            str: File pattern
        """
        file_name = self._get_file_name(feature_spec=feature_spec)

        return f"*{file_name}*.{self.cache_file_suffix}*"
