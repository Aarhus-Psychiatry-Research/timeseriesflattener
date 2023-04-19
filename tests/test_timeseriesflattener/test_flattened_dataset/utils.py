"""Integration test for the flattened dataset generation."""


from typing import List, Optional

import numpy as np
import pandas as pd
from timeseriesflattener.feature_cache.abstract_feature_cache import FeatureCache
from timeseriesflattener.feature_spec_objects import PredictorSpec
from timeseriesflattener.flattened_dataset import TimeseriesFlattener


def check_dfs_have_same_contents_by_column(df1: pd.DataFrame, df2: pd.DataFrame):
    """Check that two dataframes have the same contents by column.

    Makes debugging much easier, as it generates a diff df which is easy to read.

    Args:
        df1 (pd.DataFrame): First dataframe.
        df2 (pd.DataFrame): Second dataframe.

    Raises:
        AssertionError: If the dataframes don't have the same contents by column.
    """

    cols_to_test = [c for c in df1.columns if "prediction_time_uuid" not in c]

    for col in cols_to_test:
        # Saving to csv rounds floats, so we need to round here too
        # to avoid false negatives. Otherwise, it thinks the .csv
        # file has different values from the generated_df, simply because
        # generated_df has more decimal places.
        for df in (df1, df2):
            if df[col].dtype not in (np.dtype("O"), np.dtype("<M8[ns]")):
                df[col] = df[col].round(4)

        merged_df = df1.merge(
            df2,
            indicator=True,
            how="outer",
            on=[col, "prediction_time_uuid"],
            suffixes=("_first", "_cache"),
        )

        # Get diff rows
        diff_rows = merged_df[merged_df["_merge"] != "both"]

        # Sort rows and columns for easier comparison
        diff_rows = diff_rows.sort_index(axis=1)
        diff_rows = diff_rows.sort_values(by=[col, "prediction_time_uuid"])

        # Set display options
        pd.options.display.width = 0

        assert len(diff_rows) == 0


def create_flattened_df(
    predictor_specs: List[PredictorSpec],
    prediction_times_df: pd.DataFrame,
    cache: Optional[FeatureCache] = None,
) -> pd.DataFrame:
    """Create a dataset df for testing."""
    flat_ds = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        n_workers=1,
        cache=cache,
        drop_pred_times_with_insufficient_look_distance=False,
    )

    flat_ds.add_spec(
        spec=predictor_specs,  # type: ignore
    )

    return flat_ds.get_df()
