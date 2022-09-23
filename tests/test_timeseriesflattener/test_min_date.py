"""Tests of the min_date argument in the FlattenedDataset class.

May want to refactor into a test_flattened_dataset module.
"""
import pandas as pd
from utils_for_testing import str_to_df  # pylint: disable=import-error

from psycopmlutils.timeseriesflattener import FlattenedDataset


def test_min_date():
    """Test that the min_date argument works as expected."""
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:01
                            1,2023-12-31 00:00:01
                            """

    expected_df_str = """dw_ek_borger,timestamp,
                            1,2023-12-31 00:00:01
                            """

    prediction_times_df = str_to_df(prediction_times_str)
    expected_df = str_to_df(expected_df_str)

    flattened_dataset = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
        min_date=pd.Timestamp(2022, 12, 31),
        n_workers=4,
    )

    outcome_df = flattened_dataset.df

    for col in [
        "timestamp",
    ]:
        pd.testing.assert_series_equal(
            outcome_df[col].reset_index(drop=True),
            expected_df[col].reset_index(drop=True),
        )
