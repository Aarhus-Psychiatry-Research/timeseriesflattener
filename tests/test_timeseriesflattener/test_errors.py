import pytest
from psycopmlutils.timeseriesflattener.flattened_dataset import FlattenedDataset

from utils_for_testing import str_to_df


def test_col_does_not_exist_in_prediction_times():
    prediction_times_str = """dw_ek_borger,
                            1,
                            """

    prediction_times_df = str_to_df(prediction_times_str)

    with pytest.raises(ValueError):
        flattened_df = FlattenedDataset(
            prediction_times_df=prediction_times_df,
            timestamp_col_name="timestamp",
            id_col_name="dw_ek_borger",
        )


def test_col_does_not_exist():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """

    event_times_str = """dw_ek_borger,value,
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

    with pytest.raises(ValueError):
        flattened_df.add_temporal_predictor(
            predictor_df=event_times_df,
            lookbehind_days=2,
            resolve_multiple="max",
            fallback=2,
            source_values_col_name="val",
        )
