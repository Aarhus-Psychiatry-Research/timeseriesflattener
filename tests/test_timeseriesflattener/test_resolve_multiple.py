"""Tests of resolve_multiple strategies."""
# pylint: disable=missing-function-docstring

import numpy as np
from timeseriesflattener.feature_spec_objects import OutcomeSpec, PredictorSpec
from timeseriesflattener.testing.utils_for_testing import (
    assert_flattened_data_as_expected,
    str_to_df,
)


def test_resolve_multiple_catalogue():
    """Test that resolve_multiple functions can be retrieved from catalogue."""
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="min",
            interval_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[1],
    )


def test_resolve_multiple_max():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="max",
            interval_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[2],
    )


def test_resolve_multiple_min():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="min",
            interval_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[1],
    )


def test_resolve_multiple_avg():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 08:00:00
                            """
    predictor_df_str = """entity_id,timestamp,value,
                        1,2021-12-30 00:00:01, 1
                        1,2021-12-30 00:00:02, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=PredictorSpec(
            feature_name="value",
            values_df=str_to_df(predictor_df_str),
            resolve_multiple_fn="mean",
            interval_days=2,
            fallback=0,
        ),
        expected_values=[1.5],
    )


def test_resolve_multiple_latest():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:03, 3
                        1,2022-01-01 00:00:02, 2
                        2,2022-01-01 00:00:01, 3
                        2,2022-01-01 00:00:03, 9
                        2,2022-01-01 00:00:02, 6
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="latest",
            interval_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[3, 9],
    )


def test_resolve_multiple_latest_no_values():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="latest",
            interval_days=2,
            fallback=np.nan,
            incident=False,
        ),
        expected_values=[2, np.nan],
    )


def test_resolve_multiple_latest_one_vlaue():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="latest",
            interval_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[1],
    )


def test_resolve_multiple_earliest():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:03, 3
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        2,2022-01-01 00:00:03, 3
                        2,2022-01-01 00:00:01, 1
                        2,2022-01-01 00:00:02, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="earliest",
            interval_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[1, 1],
    )


def test_resolve_multiple_sum():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """
    predictor_df_str = """entity_id,timestamp,value,
                        1,2021-12-30 00:00:01, 1
                        1,2021-12-30 00:00:02, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=PredictorSpec(
            feature_name="value",
            values_df=str_to_df(predictor_df_str),
            resolve_multiple_fn="sum",
            interval_days=2,
            fallback=0,
        ),
        expected_values=[3],
    )


def test_resolve_multiple_count():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="count",
            interval_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[2],
    )


def test_resolve_multiple_bool():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        1,2022-01-01 00:00:02, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="bool",
            interval_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[1, 0],
    )


def test_resolve_multiple_change_per_day():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:00, 1
                        1,2022-01-02 00:00:00, 2
                        2,2022-01-01 00:00:00, 1
                        2,2023-01-08 00:00:00, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="change_per_day",
            interval_days=4,
            fallback=np.NaN,
            incident=False,
        ),
        expected_values=[1, np.NaN],
    )


def test_resolve_multiple_change_per_day_unordered():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-02 00:00:00, 2
                        1,2022-01-01 00:00:00, 1
                        2,2022-01-02 00:00:00, 2
                        2,2022-01-01 00:00:00, 1
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="change_per_day",
            interval_days=4,
            fallback=np.NaN,
            incident=False,
        ),
        expected_values=[1, 1],
    )


def test_resolve_multiple_change_per_day_negative():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-02 00:00:00, 2
                        1,2022-01-01 00:00:00, 1
                        2,2022-01-02 00:00:00, 1
                        2,2022-01-01 00:00:00, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="change_per_day",
            interval_days=4,
            fallback=np.NaN,
            incident=False,
        ),
        expected_values=[1, -1],
    )


def test_resolve_multiple_change_per_day_too_few_datapoints():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:00, 1
                        1,2022-01-02 00:00:00, 2
                        2,2022-01-01 00:00:00, 1
                        2,2022-01-08 00:00:00, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="change_per_day",
            interval_days=4,
            fallback=99999,
            incident=False,
        ),
        expected_values=[1, 99999],
    )


def test_resolve_multiple_variance():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:00, 1
                        1,2022-01-02 00:00:00, 2
                        2,2022-01-01 00:00:00, 1
                        2,2022-01-08 00:00:00, 2
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="variance",
            interval_days=4,
            fallback=np.NaN,
            incident=False,
        ),
        expected_values=[0.5, np.NaN],
    )


def test_resolve_multiple_concatenate():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:00,the patient
                        1,2022-01-02 00:00:00,is feeling ill
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_name="value",
            values_df=str_to_df(event_times_str),
            resolve_multiple_fn="concatenate",
            interval_days=4,
            fallback=np.NaN,
            incident=False,
        ),
        expected_values=["the patient is feeling ill"],
    )
