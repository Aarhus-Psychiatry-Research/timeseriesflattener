from timeseriesflattener.resolve_multiple_functions import get_max_in_group

from utils_for_testing import (
    assert_flattened_outcome_as_expected,
    assert_flattened_predictor_as_expected,
)


# Predictors
def test_predictor_after_prediction_time():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    predictor_df_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-01 00:00:01, 1.0
                        """

    assert_flattened_predictor_as_expected(
        prediction_times_df_str=prediction_times_df_str,
        predictor_df_str=predictor_df_str,
        lookbehind_days=2,
        resolve_multiple=get_max_in_group,
        expected_flattened_values=[-1],
        fallback=-1,
    )


def test_predictor_before_prediction():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    predictor_df_str = """dw_ek_borger,timestamp,val,
                        1,2021-12-30 22:59:59, 1
                        """

    assert_flattened_predictor_as_expected(
        prediction_times_df_str=prediction_times_df_str,
        predictor_df_str=predictor_df_str,
        lookbehind_days=2,
        resolve_multiple=get_max_in_group,
        expected_flattened_values=[1],
        fallback=-1,
    )


def test_multiple_citizens_predictor():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            1,2022-01-02 00:00:00
                            5,2022-01-02 00:00:00
                            5,2022-01-05 00:00:00
                            6,2022-01-05 00:00:00
                            """
    predictor_df_str = """dw_ek_borger,timestamp,val,
                        1,2021-12-30 00:00:01, 0
                        1,2022-01-01 00:00:00, 1
                        5,2022-01-01 00:00:00, 0
                        5,2022-01-04 00:00:01, 2
                        7,2022-01-05 00:00:00, 5
                        """

    assert_flattened_predictor_as_expected(
        prediction_times_df_str=prediction_times_df_str,
        predictor_df_str=predictor_df_str,
        lookbehind_days=2,
        resolve_multiple=get_max_in_group,
        expected_flattened_values=[0, 1, 0, 2, -1.0],
        fallback=-1,
    )


# Outcomes
def test_event_after_prediction_time():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    outcome_df_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-01 00:00:01, 1
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_df_str,
        outcome_df_str=outcome_df_str,
        lookahead_days=2,
        resolve_multiple=get_max_in_group,
        expected_flattened_values=[1],
    )


def test_event_before_prediction():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    outcome_df_str = """dw_ek_borger,timestamp,val,
                        1,2021-12-30 23:59:59, 1.0
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_df_str,
        outcome_df_str=outcome_df_str,
        lookahead_days=2,
        resolve_multiple=get_max_in_group,
        expected_flattened_values=[0],
    )


def test_multiple_citizens_outcome():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            1,2022-01-02 00:00:00
                            5,2025-01-02 00:00:00
                            5,2025-08-05 00:00:00
                            """
    outcome_df_str = """dw_ek_borger,timestamp,val,
                        1,2021-12-31 00:00:01, 1.0
                        1,2023-01-02 00:00:00, 1.0
                        5,2025-01-03 00:00:00, 1.0
                        5,2022-01-05 00:00:01, 1.0
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_df_str,
        outcome_df_str=outcome_df_str,
        lookahead_days=2,
        resolve_multiple=get_max_in_group,
        expected_flattened_values=[1, 0, 1, 0],
    )


def test_citizen_without_outcome():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    outcome_df_str = """dw_ek_borger,timestamp,val,
                        0,2021-12-31 00:00:01, 1.0
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_df_str,
        outcome_df_str=outcome_df_str,
        lookahead_days=2,
        resolve_multiple=get_max_in_group,
        fallback=0,
        expected_flattened_values=[0],
    )
