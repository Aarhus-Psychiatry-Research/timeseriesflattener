from timeseriesflattener.flattened_dataset import *
from timeseriesflattener.resolve_multiple_functions import *
from utils_for_testing import *
import pandas as pd

# Currently only "integration tests" (correct term?)
# Will expand with proper unit tests if we agree on the API


def test_event_after_prediction_time():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-01 00:00:01, 1
                        """

    assert_flattened_outcome_vals_as_expected(
        prediction_times_str=prediction_times_str,
        event_times_str=event_times_str,
        lookahead_days=2,
        resolve_multiple=get_max_value_from_list_of_events,
        expected_flattened_vals=[1],
    )


def test_event_before_prediction():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2021-12-30 23:59:59, 1
                        """

    assert_flattened_outcome_vals_as_expected(
        prediction_times_str=prediction_times_str,
        event_times_str=event_times_str,
        lookahead_days=2,
        resolve_multiple=get_max_value_from_list_of_events,
        expected_flattened_vals=[0],
    )


def test_multiple_citizens():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            1,2022-01-02 00:00:00
                            5,2025-01-02 00:00:00
                            5,2025-08-05 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2021-12-31 00:00:01, 1
                        1,2023-01-02 00:00:00, 1
                        5,2025-01-03 00:00:00, 1
                        5,2022-01-05 00:00:01, 1
                        """

    assert_flattened_outcome_vals_as_expected(
        prediction_times_str=prediction_times_str,
        event_times_str=event_times_str,
        lookahead_days=2,
        resolve_multiple=get_max_value_from_list_of_events,
        expected_flattened_vals=[1, 0, 1, 0],
    )
