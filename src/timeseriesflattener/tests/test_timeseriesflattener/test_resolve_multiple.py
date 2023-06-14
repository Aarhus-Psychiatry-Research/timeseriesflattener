"""Tests of resolve_multiple strategies."""


import numpy as np

from timeseriesflattener.aggregation_functions import (
    boolean,
    change_per_day,
    concatenate,
    count,
    earliest,
    latest,
    maximum,
    mean,
    mean_number_of_characters,
    minimum,
    type_token_ratio,
    variance,
)
from timeseriesflattener.feature_specs.single_specs import (
    OutcomeSpec,
    PredictorSpec,
)
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
            base_values_df=str_to_df(event_times_str),
            feature_base_name="outcome",
            aggregation_fn=minimum,
            lookahead_days=2,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=maximum,
            lookahead_days=2,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=minimum,
            lookahead_days=2,
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
            feature_base_name="value",
            base_values_df=str_to_df(predictor_df_str),
            aggregation_fn=mean,
            lookbehind_days=2,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=latest,
            lookahead_days=2,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=latest,
            lookahead_days=2,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=latest,
            lookahead_days=2,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=earliest,
            lookahead_days=2,
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
            feature_base_name="value",
            base_values_df=str_to_df(predictor_df_str),
            aggregation_fn=sum,
            lookbehind_days=2,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=count,
            lookahead_days=2,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=boolean,
            lookahead_days=2,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=change_per_day,
            lookahead_days=4,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=change_per_day,
            lookahead_days=4,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=change_per_day,
            lookahead_days=4,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=change_per_day,
            lookahead_days=4,
            fallback=99999,
            incident=False,
        ),
        expected_values=[1, 99999],
    )


def test_resolve_multiple_change_per_day_only_one_observation():
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
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=change_per_day,
            lookahead_days=4,
            fallback=0,
            incident=False,
        ),
        expected_values=[1, 0],
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=variance,
            lookahead_days=4,
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=concatenate,
            lookahead_days=4,
            fallback=np.NaN,
            incident=False,
        ),
        expected_values=["the patient is feeling ill"],
    )


def test_resolve_multiple_mean_len():
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
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=mean_number_of_characters,
            lookahead_days=4,
            fallback=np.NaN,
            incident=False,
        ),
        expected_values=[12.5],
    )


def test_resolve_multiple_type_token_ratio():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            """
    event_times_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:00,The patient feels very tired!
                        1,2022-01-02 00:00:00,The patient is tired tired.
                        2,2022-01-01 00:00:00,The patient feels very happy!
                        2,2022-01-02 00:00:00,The patient is feeling tired.
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_str,
        output_spec=OutcomeSpec(
            feature_base_name="value",
            base_values_df=str_to_df(event_times_str),
            aggregation_fn=type_token_ratio,
            lookahead_days=4,
            fallback=np.NaN,
            incident=False,
        ),
        expected_values=[0.6, 0.8],
    )
