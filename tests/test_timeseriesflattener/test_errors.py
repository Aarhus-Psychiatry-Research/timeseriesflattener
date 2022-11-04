"""Tests for errors raised from flattened dataset class."""

import pytest

from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    PredictorSpec,
)
from psycop_feature_generation.timeseriesflattener.flattened_dataset import (
    FlattenedDataset,
)
from psycop_feature_generation.utils_for_testing import (
    str_to_df,  # pylint: disable=import-error
)

# pylint: disable=missing-function-docstring


def test_col_does_not_exist_in_prediction_times():
    prediction_times_str = """dw_ek_borger,
                            1,
                            """

    prediction_times_df = str_to_df(prediction_times_str)

    with pytest.raises(ValueError):
        FlattenedDataset(  # noqa
            prediction_times_df=prediction_times_df,
            timestamp_col_name="timestamp",
            id_col_name="dw_ek_borger",
        )


def test_col_does_not_exist():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """

    event_times_str = """dw_ek_borger,val,
                        1, 1
                        1, 2
                        """

    prediction_times_df = str_to_df(prediction_times_str)
    event_times_df = str_to_df(event_times_str)

    flattened_df = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
    )

    with pytest.raises(KeyError):
        flattened_df.add_temporal_predictor(
            output_spec=PredictorSpec(
                values_df=event_times_df,
                interval_days=2,
                resolve_multiple="max",
                fallback=2,
            )
        )


def test_duplicate_prediction_times():
    with pytest.raises(ValueError):
        prediction_times_df_str = """dw_ek_borger,timestamp,
                                1,2021-12-31 00:00:00
                                1,2021-11-31 00:00:00
                                """

        FlattenedDataset(
            prediction_times_df=str_to_df(prediction_times_df_str),
        )
