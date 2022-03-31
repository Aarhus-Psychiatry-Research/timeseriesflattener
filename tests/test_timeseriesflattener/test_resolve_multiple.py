from pydoc import resolve
from utils_for_testing import *
from timeseriesflattener.resolve_multiple_functions import (
    get_max_value_from_list_of_events,
    get_avg_value_from_list_of_events,
)


def test_resolve_multiple_max():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple=get_max_value_from_list_of_events,
        lookahead_days=2,
        expected_flattened_vals=[2],
    )


def test_resolve_multiple_min():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple=get_min_value_from_list_of_events,
        lookahead_days=2,
        expected_flattened_vals=[1],
    )


def test_resolve_multiple_avg():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple=get_avg_value_from_list_of_events,
        lookahead_days=2,
        expected_flattened_vals=[1.5],
    )


def test_resolve_multiple_latest():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple=get_latest_value_from_list_of_events,
        lookahead_days=2,
        expected_flattened_vals=[2],
    )


def test_resolve_multiple_earliest():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple=get_earliest_value_from_list_of_events,
        lookahead_days=2,
        expected_flattened_vals=[1],
    )
