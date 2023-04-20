"""Tests for errors raised from flattened dataset class."""

import pytest
from timeseriesflattener.feature_spec_objects import PredictorSpec
from timeseriesflattener.flattened_dataset import TimeseriesFlattener
from timeseriesflattener.testing.utils_for_testing import (
    str_to_df,
)


def test_col_does_not_exist_in_prediction_times():
    prediction_times_str = """entity_id,
                            1,
                            """

    prediction_times_df = str_to_df(prediction_times_str)

    with pytest.raises(KeyError, match=r".*does not exist.*"):
        TimeseriesFlattener(
            prediction_times_df=prediction_times_df,
            timestamp_col_name="timestamp",
            entity_id_col_name="entity_id",
            drop_pred_times_with_insufficient_look_distance=False,
        )


def test_col_does_not_exist():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """

    event_times_str = """entity_id,val,
                        1, 1
                        1, 2
                        """

    prediction_times_df = str_to_df(prediction_times_str)
    event_times_df = str_to_df(event_times_str)

    flattened_df = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        entity_id_col_name="entity_id",
        drop_pred_times_with_insufficient_look_distance=False,
    )

    with pytest.raises(KeyError):
        flattened_df.add_spec(
            spec=PredictorSpec(
                values_df=event_times_df,
                interval_days=2,
                resolve_multiple_fn="max",
                fallback=2,
                feature_name="value",
            ),
        )


def test_duplicate_prediction_times():
    with pytest.raises(ValueError, match=r".*Duplicate.*"):  # noqa
        prediction_times_df_str = """entity_id,timestamp,
                                1,2021-12-30 00:00:00
                                1,2021-12-30 00:00:00
                                """

        TimeseriesFlattener(
            prediction_times_df=str_to_df(prediction_times_df_str),
            drop_pred_times_with_insufficient_look_distance=False,
        )
