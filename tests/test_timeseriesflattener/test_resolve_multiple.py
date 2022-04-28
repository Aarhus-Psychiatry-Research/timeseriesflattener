from timeseriesflattener.resolve_multiple_functions import (
    get_earliest_value_in_group,
    get_latest_value_in_group,
    get_max_in_group,
    get_mean_in_group,
    get_min_in_group,
)

from utils_for_testing import assert_flattened_outcome_as_expected


def test_resolve_multiple_catalogue():
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
        resolve_multiple="min",
        lookahead_days=2,
        expected_flattened_values=[1],
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
        resolve_multiple=get_max_in_group,
        lookahead_days=2,
        expected_flattened_values=[2],
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
        resolve_multiple=get_min_in_group,
        lookahead_days=2,
        expected_flattened_values=[1],
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
        resolve_multiple=get_mean_in_group,
        lookahead_days=2,
        expected_flattened_values=[1.5],
    )


def test_resolve_multiple_latest():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        2,2022-01-01 00:00:01, 3
                        2,2022-01-01 00:00:02, 6
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple=get_latest_value_in_group,
        lookahead_days=2,
        expected_flattened_values=[2, 6],
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
        resolve_multiple=get_earliest_value_in_group,
        lookahead_days=2,
        expected_flattened_values=[1],
    )


def test_resolve_multiple_sum():
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
        resolve_multiple="sum",
        lookahead_days=2,
        expected_flattened_values=[3],
    )


def test_resolve_multiple_count():
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
        resolve_multiple="count",
        lookahead_days=2,
        expected_flattened_values=[2],
    )
