"""Tests of resolve_multiple strategies."""
# pylint: disable=missing-function-docstring

import numpy as np

from psycopmlutils.timeseriesflattener.resolve_multiple_functions import (  # noqa pylint: disable=unused-import
    get_earliest_value_in_group,
    get_latest_value_in_group,
    get_max_in_group,
    get_mean_in_group,
    get_min_in_group,
)
from tests.helpers.utils_for_testing import (  # pylint: disable=import-error
    assert_flattened_outcome_as_expected,
    assert_flattened_predictor_as_expected,
)


def test_resolve_multiple_catalogue():
    """Test that resolve_multiple functions can be retrieved from catalogue."""
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,value,
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
    event_times_str = """dw_ek_borger,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple="max",
        lookahead_days=2,
        expected_flattened_values=[2],
    )


def test_resolve_multiple_min():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,value,
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


def test_resolve_multiple_avg():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 08:00:00
                            """
    predictor_df_str = """dw_ek_borger,timestamp,value,
                        1,2021-12-30 00:00:01, 1
                        1,2021-12-30 00:00:02, 2
                        """

    assert_flattened_predictor_as_expected(
        prediction_times_df_str=prediction_times_str,
        predictor_df_str=predictor_df_str,
        resolve_multiple="mean",
        lookbehind_days=2,
        expected_flattened_values=[1.5],
    )


def test_resolve_multiple_latest():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        2,2022-01-01 00:00:01, 3
                        2,2022-01-01 00:00:02, 6
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple="latest",
        lookahead_days=2,
        expected_flattened_values=[2, 6],
    )


def test_resolve_multiple_latest_no_values():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple="latest",
        lookahead_days=2,
        expected_flattened_values=[2, np.nan],
    )


def test_resolve_multiple_latest_one_vlaue():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple="latest",
        lookahead_days=2,
        expected_flattened_values=[1],
    )


def test_resolve_multiple_earliest():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple="earliest",
        lookahead_days=2,
        expected_flattened_values=[1],
    )


def test_resolve_multiple_sum():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    predictor_df_str = """dw_ek_borger,timestamp,value,
                        1,2021-12-30 00:00:01, 1
                        1,2021-12-30 00:00:02, 2
                        """

    assert_flattened_predictor_as_expected(
        prediction_times_df_str=prediction_times_str,
        predictor_df_str=predictor_df_str,
        resolve_multiple="sum",
        lookbehind_days=2,
        expected_flattened_values=[3],
    )


def test_resolve_multiple_count():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,value,
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


def test_resolve_multiple_bool():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple="bool",
        lookahead_days=2,
        expected_flattened_values=[1, 0],
    )


def test_resolve_multiple_change_per_day():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-01 00:00:00, 1
                        1,2022-01-02 00:00:00, 2
                        2,2022-01-01 00:00:00, 1
                        2,2022-01-08 00:00:00, 2
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple="change_per_day",
        lookahead_days=4,
        expected_flattened_values=[1, np.NaN],
    )


def test_resolve_multiple_change_per_day_unordered():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-02 00:00:00, 2
                        1,2022-01-01 00:00:00, 1
                        2,2022-01-02 00:00:00, 2
                        2,2022-01-01 00:00:00, 1
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple="change_per_day",
        lookahead_days=4,
        expected_flattened_values=[1, 1],
    )


def test_resolve_multiple_change_per_day_negative():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-02 00:00:00, 2
                        1,2022-01-01 00:00:00, 1
                        2,2022-01-02 00:00:00, 1
                        2,2022-01-01 00:00:00, 2
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple="change_per_day",
        lookahead_days=4,
        expected_flattened_values=[1, -1],
    )


def test_resolve_multiple_change_per_day_too_few_datapoints():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-01 00:00:00, 1
                        1,2022-01-02 00:00:00, 2
                        2,2022-01-01 00:00:00, 1
                        2,2022-01-08 00:00:00, 2
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple="change_per_day",
        lookahead_days=4,
        expected_flattened_values=[1, 99999],
        fallback=99999,
    )


def test_resolve_multiple_mean_of_multiple_columns():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val1,val2
                        1,2022-01-01 00:00:00,1,2
                        1,2022-01-02 00:00:00,2,3
                        2,2022-01-01 00:00:00,3,4
                        2,2022-01-08 00:00:00,4,5
                        """

    assert_flattened_outcome_as_expected(
        prediction_times_df_str=prediction_times_str,
        outcome_df_str=event_times_str,
        resolve_multiple="mean",
        lookahead_days=4,
        expected_flattened_values=[[1.5, 2.5], [3.0, 4.0]],
        fallback=99999,
        values_colname=["val1", "val2"],
    )
