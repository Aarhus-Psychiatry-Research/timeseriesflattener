"""Tests of aggregation strategies."""


import numpy as np

from timeseriesflattener.aggregation_fns import (
    boolean,
    change_per_day,
    count,
    earliest,
    latest,
    maximum,
    mean,
    minimum,
    summed,
    variance,
)
from timeseriesflattener.feature_specs.single_specs import OutcomeSpec, PredictorSpec
from timeseriesflattener.testing.utils_for_testing import (
    assert_flattened_data_as_expected,
    str_to_df,
)


def test_aggregation_max():
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
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=maximum,
            lookahead_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[2],
    )


def test_aggregation_min():
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
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=minimum,
            lookahead_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[1],
    )


def test_aggregation_avg():
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
            feature_base_name="value",
            timeseries_df=str_to_df(predictor_df_str),
            aggregation_fn=mean,
            lookbehind_days=2,
            fallback=0,
        ),
        expected_values=[1.5],
    )


def test_aggregation_latest():
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
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=latest,
            lookahead_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[3, 9],
    )


def test_aggregation_latest_no_values():
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
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=latest,
            lookahead_days=2,
            fallback=np.nan,
            incident=False,
        ),
        expected_values=[2, np.nan],
    )


def test_aggregation_latest_one_vlaue():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=latest,
            lookahead_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[1],
    )


def test_aggregation_earliest():
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
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=earliest,
            lookahead_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[1, 1],
    )


def test_aggregation_sum():
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
            feature_base_name="value",
            timeseries_df=str_to_df(predictor_df_str),
            aggregation_fn=summed,
            lookbehind_days=2,
            fallback=0,
        ),
        expected_values=[3],
    )


def test_aggregation_count():
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
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=count,
            lookahead_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[2],
    )


def test_aggregation_bool():
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
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=boolean,
            lookahead_days=2,
            fallback=0,
            incident=False,
        ),
        expected_values=[1, 0],
    )


def test_aggregation_change_per_day():
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
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=change_per_day,
            lookahead_days=4,
            fallback=np.NaN,
            incident=False,
        ),
        expected_values=[1, np.NaN],
    )


def test_aggregation_change_per_day_unordered():
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
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=change_per_day,
            lookahead_days=4,
            fallback=np.NaN,
            incident=False,
        ),
        expected_values=[1, 1],
    )


def test_aggregation_change_per_day_negative():
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
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=change_per_day,
            lookahead_days=4,
            fallback=np.NaN,
            incident=False,
        ),
        expected_values=[1, -1],
    )


def test_aggregation_change_per_day_too_few_datapoints():
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
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=change_per_day,
            lookahead_days=4,
            fallback=99999,
            incident=False,
        ),
        expected_values=[1, 99999],
    )


def test_aggregation_change_per_day_only_one_observation():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:00, 1
                        1,2022-01-02 00:00:00, 2
                        2,2022-01-01 00:00:00, 1
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=change_per_day,
            lookahead_days=4,
            fallback=0,
            incident=False,
        ),
        expected_values=[1, 0],
    )


def test_aggregation_variance():
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
            feature_base_name="value",
            timeseries_df=str_to_df(event_times_str),
            aggregation_fn=variance,
            lookahead_days=4,
            fallback=np.NaN,
            incident=False,
        ),
        expected_values=[0.5, np.NaN],
    )
