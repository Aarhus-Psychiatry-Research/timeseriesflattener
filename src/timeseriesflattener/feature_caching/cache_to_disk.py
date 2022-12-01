import os
from pathlib import Path

import pandas as pd

from timeseriesflattener import TimeseriesFlattener
from timeseriesflattener.feature_caching.abstract_feature_cache import FeatureCache
from timeseriesflattener.feature_spec_objects import AnySpec
from timeseriesflattener.utils import load_dataset_from_file, write_df_to_file


class DiskCache(FeatureCache):
    def __init__(
        self, feature_cache_dir: Path, cache_file_suffix: str, validate: bool = True, uuid_col_name: str
    ):
        """Initialize DiskCache."""

        self.feature_cache_dir = feature_cache_dir
        self.cache_file_suffix = cache_file_suffix
        self.uuid_col_name: str

        if not self.feature_cache_dir.exists():
            self.feature_cache_dir.mkdir()

        self.validate = validate

    def _generate_values_for_cache_checking(
        self,
        msg,
        prediction_times_df,
        id_col_name,
        timestamp_col_name,
        pred_time_uuid_col_name,
        output_spec: AnySpec,
        value_col_str: str,
        n_to_generate: int = 100_000,
    ):
        generated_df = pd.DataFrame({value_col_str: []})

        # Check that some values in generated_df differ from fallback
        # Otherwise, comparison to cache is meaningless
        n_trials = 0

        while not any(
            generated_df[value_col_str] != output_spec.fallback,
        ):
            if n_trials != 0:
                msg.info(
                    f"{value_col_str[20]}, {n_trials}: Generated_df was all fallback values, regenerating",
                )

            n_to_generate = int(min(n_to_generate, len(prediction_times_df)))

            generated_df = TimeseriesFlattener.flatten_temporal_values_to_df(
                prediction_times_with_uuid_df=prediction_times_df.sample(
                    n=n_to_generate,
                    replace=False,
                ),
                id_col_name=id_col_name,
                timestamp_col_name=timestamp_col_name,
                pred_time_uuid_col_name=pred_time_uuid_col_name,
                output_spec=output_spec,
            ).dropna()

            # Fallback values are not interesting for cache hit. If they exist in generated_df, they should be dropped
            # in the cache. Saves on storage. Don't use them to check if cache is hit.
            if not np.isnan(output_spec.fallback):  # type: ignore
                generated_df = generated_df[
                    generated_df[value_col_str] != output_spec.fallback
                ]

            n_to_generate = (
                n_to_generate**1.5
            )  # Increase n_to_generate by 1.5x each time to increase chance of non_fallback values

            n_trials += 1

        return generated_df

    def load_most_recent_df_matching_pattern(
        self,
        file_pattern: str,
        file_suffix: str,
    ) -> DataFrame:
        """Load most recent df matching pattern.

        Args:
            file_pattern (str): Pattern to match
            file_suffix (str, optional): File suffix to match.

        Returns:
            DataFrame: DataFrame matching pattern

        Raises:
            FileNotFoundError: If no file matching pattern is found
        """
        files_with_suffix = list(
            self.feature_cache_dir.glob(f"*{file_pattern}*.{file_suffix}")
        )

        if len(files_with_suffix) == 0:
            raise FileNotFoundError(f"No files matching pattern {file_pattern} found")

        path_of_most_recent_file = max(files_with_suffix, key=os.path.getctime)

        return load_dataset_from_file(
            file_path=path_of_most_recent_file,
        )

    def load_cached_df_and_expand_fallback(
        self,
        feature_cache_dir,
        _df,
        pred_time_uuid_col_name,
        file_pattern: str,
        file_suffix: str,
        fallback: Any,
        full_col_str: str,
    ) -> pd.DataFrame:
        """Load most recent df matching pattern, and expand fallback column.

        Args:
            file_pattern (str): File pattern to search for
            file_suffix (str): File suffix to search for
            fallback (Any): Fallback value
            full_col_str (str): Full column name for values

        Returns:
            DataFrame: DataFrame with fallback column expanded
        """
        df = self.load_most_recent_df_matching_pattern(
            dir_path=feature_cache_dir,
            file_pattern=file_pattern,
            file_suffix=file_suffix,
        )

        # Expand fallback column
        df = pd.merge(
            left=_df[pred_time_uuid_col_name],
            right=df,
            how="left",
            on=pred_time_uuid_col_name,
            validate="m:1",
        )

        df[full_col_str] = df[full_col_str].fillna(fallback)

        return df

    def _validate_feature_cache_matches_source(
        self,
        cache_df: pd.DataFrame,
        prediction_times_df: pd.DataFrame,
        pred_time_uuid_col_name: str,
        value_col_str: str,
    ) -> bool:

        generated_df = self._generate_values_for_cache_checking(
            prediction_times_df=raw_df,
            id_col_name=id_col_name,
            timestamp_col_name=timestamp_col_name,
            pred_time_uuid_col_name=pred_time_uuid_col_name,
            value_col_str=value_col_str,
        )

        cached_suffix = "_c"
        generated_suffix = "_g"

        # We frequently hit rounding errors with cache hits, so we round to 3 decimal places
        generated_df[value_col_str] = generated_df[value_col_str].round(3)
        cache_df[value_col_str] = cache_df[value_col_str].round(3)

        # Merge cache_df onto generated_df
        merged_df = pd.merge(
            left=generated_df,
            right=cache_df,
            how="left",
            on=pred_time_uuid_col_name,
            suffixes=(generated_suffix, cached_suffix),
            validate="1:1",
            indicator=True,
        )

        # Check that all rows in generated_df are in cache_df
        if not merged_df[value_col_str + generated_suffix].equals(
            merged_df[value_col_str + cached_suffix],
        ):
            msg1.info(f"Cache miss, computed values didn't match {file_pattern}")

            # Keep this variable for easier inspection
            unequal_rows = merged_df[  # pylint: disable=unused-variable
                merged_df[value_col_str + generated_suffix]
                != merged_df[value_col_str + cached_suffix]
            ]

            return False

        # If all checks passed, return true
        msg.good(f"Cache hit for {value_col_str}")
        return True

    def write_feature(
        self,
        values_df: pd.DataFrame,
        feature_spec: AnySpec,
    ):
        """Write feature to cache."""
        n_uuids = len(unique(values_df[self.uuid_col_name]))
        file_name = f"{feature_spec.get_col_str()}_{n_uuids}_uuids"

        # Drop rows containing fallback, since it's non-informative
        values_df = values_df[
            values_df[feature_spec.get_col_str()] != feature_spec.fallback
        ].dropna()

        # Write df to cache
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Write df to cache
        write_df_to_file(
            df=values_df,
            file_path=self.feature_cache_dir
            / f"{file_name}_{timestamp}.{self.cache_file_suffix}",
        )

    def feature_exists(
        self,
        prediction_times_df: pd.DataFrame,
        pred_time_uuid_col_name: str,
        feature_spec: AnySpec,
        file_pattern: str,
        file_suffix: str,
    ) -> bool:
        """Check if cache is hit.

        Args:
            kwargs_dict (dict): dictionary of kwargs
            file_pattern (str): File pattern to match. Looks for *file_pattern* in cache dir.
            e.g. "*feature_name*_uuids*.file_suffix"
            full_col_str (str): Full column string. e.g. "feature_name_ahead_interval_days_resolve_multiple_fallback"
            file_suffix (str): File suffix to match. e.g. "csv"

        Returns:
            bool: True if cache is hit, False otherwise
        """

        file_name = f"{feature_spec.get_col_str()}_{self.n_uuids}_uuids"
        # Check that file exists
        file_pattern_hits = list(
            self.feature_cache_dir.glob(f"*{file_pattern}*.{file_suffix}"),
        )

        if len(file_pattern_hits) == 0:
            return False

        value_col_str = feature_spec.get_col_str()

        # Check that file contents match expected
        # NAs are not interesting when comparing if computed values are identical
        # TODO: Handle if the file doesn't exist. Write a test that covers it.
        cache_df = self.load_most_recent_df_matching_pattern(
            file_pattern=file_pattern,
            file_suffix=file_suffix,
        )

        if self.validate:
            return self._validate_feature_cache_matches_source(
                cache_df=cache_df,
                pred_time_uuid_col_name=pred_time_uuid_col_name,
                value_col_str=value_col_str,
                output_spec=feature_spec,
                prediction_times_df=prediction_times_df,
            )

        return True