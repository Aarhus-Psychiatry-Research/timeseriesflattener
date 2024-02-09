"""Tests for adding values to a flattened dataset."""


import numpy as np
import pandas as pd
import pytest

from timeseriesflattener import TimeseriesFlattener
from timeseriesflattener.aggregation_fns import latest, maximum, minimum
from timeseriesflattener.feature_specs.single_specs import OutcomeSpec, PredictorSpec, StaticSpec
from timeseriesflattener.testing.utils_for_testing import (
    assert_flattened_data_as_expected,
    str_to_df,
)


# Predictors
def test_predictor_after_prediction_time():
    prediction_times_df = str_to_df(
        """entity_id,timestamp,
    1,2021-12-31 00:00:00
    """
    )
    predictor_df = str_to_df(
        """entity_id,timestamp,value,
    1,2022-01-01 00:00:01, 1.0
    """
    )

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df,
        output_spec=PredictorSpec(
            timeseries_df=predictor_df,
            lookbehind_days=2,
            aggregation_fn=maximum,
            fallback=np.NaN,
            feature_base_name="value",
        ),
        expected_values=[np.NaN],
    )


def test_predictor_before_prediction():
    prediction_times_df = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """
    predictor_df_str = """entity_id,timestamp,value,
                        1,2021-12-30 22:59:59, 1
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df,
        output_spec=PredictorSpec(
            timeseries_df=str_to_df(predictor_df_str),
            lookbehind_days=2,
            aggregation_fn=maximum,
            fallback=np.NaN,
            feature_base_name="value",
        ),
        expected_values=[1],
    )


def test_multiple_citizens_predictor():
    prediction_times_df_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            1,2022-01-02 00:00:00
                            5,2022-01-02 00:00:00
                            5,2022-01-05 00:00:00
                            6,2022-01-05 00:00:00
                            """
    predictor_df_str = """entity_id,timestamp,value,
                        1,2021-12-30 00:00:01, 0
                        1,2022-01-01 00:00:00, 1
                        5,2022-01-01 00:00:00, 0
                        5,2022-01-04 00:00:01, 2
                        7,2022-01-05 00:00:00, 5
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=PredictorSpec(
            timeseries_df=str_to_df(predictor_df_str),
            lookbehind_days=2,
            fallback=np.NaN,
            feature_base_name="value",
            aggregation_fn=maximum,
        ),
        expected_values=[0, 1, 0, 2, np.NaN],
    )


def test_predictor_with_interval_lookperiod():
    prediction_times_df_str = """entity_id,timestamp,
                            1,2022-01-01 00:00:00
                            """
    predictor_df_str = """entity_id,timestamp,value,
                        1,2021-12-30 00:00:01, 2
                        1,2021-12-15 00:00:00, 1
                        """
    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=PredictorSpec(
            timeseries_df=str_to_df(predictor_df_str),
            lookbehind_days=(5, 30),
            fallback=np.NaN,
            feature_base_name="value",
            aggregation_fn=maximum,
        ),
        expected_values=[1],
    )


# Outcomes
def test_event_after_prediction_time():
    prediction_times_df_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """
    outcome_df_str = """entity_id,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=OutcomeSpec(
            timeseries_df=str_to_df(outcome_df_str),
            lookahead_days=2,
            aggregation_fn=maximum,
            incident=True,
            fallback=np.NaN,
            feature_base_name="value",
        ),
        expected_values=[1],
    )


def test_event_before_prediction():
    prediction_times_df_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """
    outcome_df_str = """entity_id,timestamp,value,
                        1,2021-12-30 23:59:59, 1.0
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=OutcomeSpec(
            timeseries_df=str_to_df(outcome_df_str),
            lookahead_days=2,
            aggregation_fn=maximum,
            incident=False,
            fallback=np.NaN,
            feature_base_name="value",
        ),
        expected_values=[np.NaN],
    )


def test_multiple_citizens_outcome():
    prediction_times_df_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            1,2022-01-02 00:00:00
                            5,2025-01-02 00:00:00
                            5,2025-08-05 00:00:00
                            """
    outcome_df_str = """entity_id,timestamp,value
                        1,2021-12-31 00:00:01, 1.0
                        1,2023-01-02 00:00:00, 1.0
                        5,2025-01-03 00:00:00, 1.0
                        5,2022-01-05 00:00:01, 1.0
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=OutcomeSpec(
            timeseries_df=str_to_df(outcome_df_str),
            lookahead_days=2,
            aggregation_fn=maximum,
            incident=False,
            fallback=np.NaN,
            feature_base_name="value",
        ),
        expected_values=[1, np.NaN, 1, np.NaN],
    )


def test_outcome_with_interval_lookperiod_outside():
    prediction_times_df_str = """entity_id,timestamp,
                            1,2022-01-01 00:00:00
                            """
    outcome_df_str = """entity_id,timestamp,value,
                        1,2022-01-02 00:00:00, 1
                        """
    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=OutcomeSpec(
            timeseries_df=str_to_df(outcome_df_str),
            lookahead_days=(2, 10),
            fallback=0,
            incident=True,
            feature_base_name="value",
            aggregation_fn=maximum,
        ),
        expected_values=[0],
    )


def test_outcome_interval_lookperiod_inside():
    prediction_times_df_str = """entity_id,timestamp,
                            1,2022-01-01 00:00:00
                            """
    outcome_df_str = """entity_id,timestamp,value,
                        1,2022-01-03 00:00:00, 1
                        """
    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=OutcomeSpec(
            timeseries_df=str_to_df(outcome_df_str),
            lookahead_days=(1, 10),
            fallback=0,
            incident=True,
            feature_base_name="value",
            aggregation_fn=maximum,
        ),
        expected_values=[1],
    )


def test_citizen_without_outcome():
    prediction_times_df_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """
    outcome_df_str = """entity_id,timestamp,value,
                        0,2021-12-31 00:00:01, 1.0
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=OutcomeSpec(
            timeseries_df=str_to_df(outcome_df_str),
            lookahead_days=2,
            aggregation_fn=maximum,
            incident=False,
            fallback=np.NaN,
            feature_base_name="value",
        ),
        expected_values=[np.NaN],
    )


def test_static_predictor():
    prefix = "meta"
    feature_name = "date_of_birth"
    output_col_name = f"{prefix}_{feature_name}"

    prediction_times_df = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            1,2021-12-31 00:00:01
                            1,2021-12-31 00:00:02
                            """
    static_predictor = f"""entity_id,{feature_name}
                        1,1994-12-31 00:00:01
                        """

    dataset = TimeseriesFlattener(
        prediction_times_df=str_to_df(prediction_times_df),
        drop_pred_times_with_insufficient_look_distance=False,
    )

    dataset.add_spec(
        StaticSpec(  # type: ignore
            timeseries_df=str_to_df(static_predictor), feature_base_name=feature_name, prefix=prefix
        )
    )

    expected_values = pd.DataFrame(
        {output_col_name: ["1994-12-31 00:00:01", "1994-12-31 00:00:01", "1994-12-31 00:00:01"]}
    )

    pd.testing.assert_series_equal(
        left=dataset.get_df()[output_col_name].reset_index(drop=True),
        right=expected_values[output_col_name].reset_index(drop=True),
        check_dtype=False,
    )


def test_add_age():
    prediction_times_df = """entity_id,timestamp,
                            1,1994-12-31 00:00:00
                            1,2021-12-30 00:00:00
                            1,2021-12-31 00:00:00
                            """
    static_predictor = """entity_id,date_of_birth
                        1,1994-12-31 00:00:00
                        """

    dataset = TimeseriesFlattener(
        prediction_times_df=str_to_df(prediction_times_df),
        drop_pred_times_with_insufficient_look_distance=False,
    )

    output_prefix = "eval"

    dataset.add_age(
        date_of_birth_df=str_to_df(static_predictor),
        date_of_birth_col_name="date_of_birth",
        output_prefix=output_prefix,
    )

    expected_values = pd.DataFrame({f"{output_prefix}_age_in_years": [0.0, 27.0, 27.0]})

    pd.testing.assert_series_equal(
        left=dataset.get_df()["eval_age_in_years"].reset_index(drop=True),
        right=expected_values[f"{output_prefix}_age_in_years"].reset_index(drop=True),
        check_dtype=False,
    )


def test_add_age_error():
    prediction_times_df = """entity_id,timestamp,
                            1,1994-12-31 00:00:00
                            1,2021-11-28 00:00:00
                            1,2021-12-31 00:00:00
                            """
    static_predictor = """entity_id,date_of_birth
                        1,XYZ
                        """

    dataset = TimeseriesFlattener(
        prediction_times_df=str_to_df(prediction_times_df),
        drop_pred_times_with_insufficient_look_distance=False,
    )

    with pytest.raises(ValueError, match=".*Recommend converting.*"):
        dataset.add_age(
            date_of_birth_df=str_to_df(static_predictor), date_of_birth_col_name="date_of_birth"
        )


def test_incident_addition_with_multiple_timestamps_raises_meaningful_error():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            """

    event_times_str = """entity_id,timestamp,value,
                        1,2021-12-31 00:00:01, 1
                        1,2021-12-31 00:00:01, 1
                        """

    prediction_times_df = str_to_df(prediction_times_str)
    event_times_df = str_to_df(event_times_str)

    flattened_dataset = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        entity_id_col_name="entity_id",
        n_workers=4,
        drop_pred_times_with_insufficient_look_distance=False,
    )

    flattened_dataset.add_spec(
        spec=OutcomeSpec(
            timeseries_df=event_times_df,
            lookahead_days=2,
            incident=True,
            fallback=np.NaN,
            feature_base_name="value",
            aggregation_fn=maximum,
        )
    )

    with pytest.raises(ValueError, match="Since incident = True"):
        flattened_dataset.get_df().reset_index(drop=True)


def test_incident_outcome_removing_prediction_times():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            1,2023-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            2,2023-12-30 00:00:00
                            3,2023-12-31 00:00:00
                            """

    event_times_str = """entity_id,timestamp,value,
                        1,2021-12-31 00:00:01, 1
                        2,2021-12-31 00:00:01, 1
                        """

    expected_df_str = """entity_id,timestamp,outc_value_within_2_days_maximum_fallback_nan_dichotomous,
                        1,2021-12-31 00:00:00, 1.0
                        2,2021-12-31 00:00:00, 1.0
                        3,2023-12-31 00:00:00, 0.0
                        """

    prediction_times_df = str_to_df(prediction_times_str)
    event_times_df = str_to_df(event_times_str)
    expected_df = str_to_df(expected_df_str)

    flattened_dataset = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        entity_id_col_name="entity_id",
        n_workers=4,
        drop_pred_times_with_insufficient_look_distance=False,
    )

    flattened_dataset.add_spec(
        spec=OutcomeSpec(
            timeseries_df=event_times_df,
            lookahead_days=2,
            incident=True,
            fallback=np.NaN,
            feature_base_name="value",
            aggregation_fn=maximum,
        )
    )

    outcome_df = flattened_dataset.get_df().reset_index(drop=True)

    for col in expected_df.columns:
        pd.testing.assert_series_equal(outcome_df[col], expected_df[col], check_dtype=False)


def test_add_multiple_static_predictors():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-12-31 00:00:00
                            1,2023-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            2,2023-12-31 00:00:00
                            3,2023-12-31 00:00:00
                            """

    event_times_str = """entity_id,timestamp,value,
                        1,2021-12-31 00:00:01, 1
                        2,2021-12-31 00:00:01, 1
                        """

    expected_df_str = """entity_id,timestamp,outc_value_within_2_days_maximum_fallback_0_dichotomous,pred_age_in_years,pred_male
                        1,2021-12-31 00:00:00, 1.0,22.00,1
                        2,2021-12-31 00:00:00, 1.0,22.00,0
                        3,2023-12-31 00:00:00, 0.0,23.99,1
                        """

    birthdates_df_str = """entity_id,date_of_birth,
    1,2000-01-01,
    2,2000-01-02,
    3,2000-01-03"""

    male_df_str = """entity_id,male,
    1,1
    2,0
    3,1"""

    prediction_times_df = str_to_df(prediction_times_str)
    event_times_df = str_to_df(event_times_str)
    expected_df = str_to_df(expected_df_str)
    birthdates_df = str_to_df(birthdates_df_str)
    male_df = str_to_df(male_df_str)

    flattened_dataset = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        entity_id_col_name="entity_id",
        n_workers=4,
        drop_pred_times_with_insufficient_look_distance=False,
    )

    output_spec = OutcomeSpec(
        timeseries_df=event_times_df,
        lookahead_days=2,
        aggregation_fn=maximum,
        fallback=0,
        incident=True,
        feature_base_name="value",
    )

    flattened_dataset.add_spec(
        spec=[
            output_spec,
            StaticSpec(  # type: ignore
                timeseries_df=male_df, feature_base_name="male", prefix="pred"
            ),
        ]
    )

    flattened_dataset.add_age(
        date_of_birth_col_name="date_of_birth", date_of_birth_df=birthdates_df
    )

    outcome_df = flattened_dataset.get_df()

    for col in (
        "entity_id",
        "timestamp",
        "outc_value_within_2_days_maximum_fallback_0_dichotomous",
        "pred_age_in_years",
        "pred_male",
    ):
        pd.testing.assert_series_equal(
            outcome_df[col].reset_index(drop=True),
            expected_df[col].reset_index(drop=True),
            check_dtype=False,
        )


def test_add_temporal_predictors_then_temporal_outcome():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-11-05 00:00:00
                            2,2021-11-05 00:00:00
                            """

    predictors_df_str = """entity_id,timestamp,value,
                        1,2020-11-05 00:00:01, 1
                        2,2020-11-05 00:00:01, 1
                        2,2021-01-15 00:00:01, 3
                        """

    event_times_str = """entity_id,timestamp,value,
                        1,2021-11-05 00:00:01, 1
                        2,2021-11-05 00:00:01, 1
                        """

    expected_df_str = """entity_id,timestamp,prediction_time_uuid
                            2,2021-11-05,2-2021-11-05-00-00-00
                            1,2021-11-05,1-2021-11-05-00-00-00
                        """

    prediction_times_df = str_to_df(prediction_times_str)
    predictors_df = str_to_df(predictors_df_str)
    event_times_df = str_to_df(event_times_str)
    expected_df = str_to_df(expected_df_str)

    flattened_dataset = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        entity_id_col_name="entity_id",
        n_workers=4,
        drop_pred_times_with_insufficient_look_distance=False,
    )

    flattened_dataset.add_spec(
        spec=[
            PredictorSpec(
                timeseries_df=predictors_df,
                lookbehind_days=365,
                aggregation_fn=minimum,
                fallback=np.nan,
                feature_base_name="value",
            ),
            OutcomeSpec(
                timeseries_df=event_times_df,
                lookahead_days=2,
                aggregation_fn=maximum,
                fallback=0,
                incident=True,
                feature_base_name="value",
            ),
        ]
    )

    outcome_df = flattened_dataset.get_df().set_index("entity_id").sort_index()
    expected_df = expected_df.set_index("entity_id").sort_index()

    for col in expected_df.columns:
        pd.testing.assert_series_equal(
            outcome_df[col], expected_df[col], check_index=False, check_dtype=False
        )


def test_add_temporal_incident_binary_outcome():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-11-05 00:00:00
                            1,2021-11-01 00:00:00
                            1,2023-11-05 00:00:00
                            """

    event_times_str = """entity_id,timestamp,value,
                        1,2021-11-06 00:00:01, 1
                        """

    expected_df_str = """outc_value_within_2_days_maximum_fallback_nan_dichotomous,
    1
    0"""

    prediction_times_df = str_to_df(prediction_times_str)
    event_times_df = str_to_df(event_times_str)
    expected_df = str_to_df(expected_df_str)

    flattened_dataset = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        entity_id_col_name="entity_id",
        n_workers=4,
        drop_pred_times_with_insufficient_look_distance=False,
    )

    flattened_dataset.add_spec(
        spec=OutcomeSpec(
            timeseries_df=event_times_df,
            lookahead_days=2,
            incident=True,
            fallback=np.NaN,
            feature_base_name="value",
            aggregation_fn=maximum,
        )
    )

    outcome_df = flattened_dataset.get_df()

    for col in [c for c in expected_df.columns if "outc" in c]:
        for df in (outcome_df, expected_df):
            # Windows and Linux have different default dtypes for ints,
            # which is not a meaningful error here. So we force the dtype.
            if df[col].dtype == "int64":
                df[col] = df[col].astype("int32")

        pd.testing.assert_series_equal(outcome_df[col], expected_df[col])


def test_add_outcome_timestamps():
    prediction_times_str = """entity_id,timestamp,
                            1,2021-11-05 00:00:00
                            1,2021-11-01 00:00:00
                            1,2023-11-05 00:00:00
                            """

    event_times_str = """entity_id,timestamp,value,
                        1,2021-11-06 00:00:01,2021-11-06 00:00:01
                        1,2021-11-13 00:00:01,2021-11-13 00:00:01
                        """

    expected_df_str = """entity_id,outc_timestamp_within_10_days_latest_fallback_nan_dichotomous,
    0,2021-11-13 00:00:01
    1,2021-11-06 00:00:01
    2,
    """

    prediction_times_df = str_to_df(prediction_times_str)
    event_times_df = str_to_df(event_times_str)
    expected_df = str_to_df(expected_df_str)
    expected_df["outc_timestamp_within_10_days_latest_fallback_nan_dichotomous"] = (
        expected_df["outc_timestamp_within_10_days_latest_fallback_nan_dichotomous"]
        .astype(str)
        .replace("NaT", np.NaN)
    )

    flattened_dataset = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        entity_id_col_name="entity_id",
        n_workers=4,
        drop_pred_times_with_insufficient_look_distance=False,
    )

    flattened_dataset.add_spec(
        spec=OutcomeSpec(
            timeseries_df=event_times_df,
            lookahead_days=10,
            incident=False,
            fallback=np.NaN,
            feature_base_name="timestamp",
            aggregation_fn=latest,
        )
    )

    outcome_df = flattened_dataset.get_df()

    for col in [c for c in expected_df.columns if "outc" in c]:
        for df in (outcome_df, expected_df):
            # Windows and Linux have different default dtypes for ints,
            # which is not a meaningful error here. So we force the dtype.
            if df[col].dtype == "int64":
                df[col] = df[col].astype("int32")

        pd.testing.assert_series_equal(outcome_df[col], expected_df[col])


def test_drop_pred_times_with_insufficient_look_distance():
    prediction_times_df_str = """entity_id,timestamp,
                            1,2022-01-01 00:00:00
                            1,2022-01-02 00:00:00
                            1,2022-01-03 00:00:00
                            1,2022-01-04 00:00:00
                            """
    outcome_df_str = """entity_id,timestamp,value,
                        1,2022-01-04 00:00:00, 1
                        """
    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=OutcomeSpec(
            timeseries_df=str_to_df(outcome_df_str),
            lookahead_days=(2),
            fallback=0,
            incident=True,
            feature_base_name="value",
            aggregation_fn=maximum,
        ),
        expected_values=[0, 0],
        drop_pred_times_with_insufficient_look_distance=True,
    )
