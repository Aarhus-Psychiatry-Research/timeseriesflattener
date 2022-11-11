"""Tests for adding values to a flattened dataset."""

# pylint: disable=missing-function-docstring

import numpy as np
import pandas as pd
import pytest

from psycop_feature_generation.loaders.raw.load_text import (  # noqa pylint: disable=unused-import; load_synth_notes,
    _chunk_text,
)
from psycop_feature_generation.timeseriesflattener import FlattenedDataset
from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    OutcomeSpec,
    PredictorGroupSpec,
    PredictorSpec,
    StaticSpec,
)
from psycop_feature_generation.utils import data_loaders
from psycop_feature_generation.utils_for_testing import (
    assert_flattened_data_as_expected,
    str_to_df,
)

# pylint: disable=import-error
# from tests.test_data.test_hf.test_hf_embeddings import TEST_HF_EMBEDDINGS
# from tests.test_data.test_tfidf.test_tfidf_vocab import TEST_TFIDF_VOCAB


# Predictors
def test_predictor_after_prediction_time():
    prediction_times_df = str_to_df(
        """dw_ek_borger,timestamp,
    1,2021-12-31 00:00:00
    """,
    )
    predictor_df = str_to_df(
        """dw_ek_borger,timestamp,value,
    1,2022-01-01 00:00:01, 1.0
    """,
    )

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df,
        output_spec=PredictorSpec(
            values_df=predictor_df,
            interval_days=2,
            resolve_multiple_fn_name="max",
            fallback=np.NaN,
            feature_name="value",
        ),
        expected_values=[np.NaN],
    )


def test_predictor_before_prediction():
    prediction_times_df = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    predictor_df_str = """dw_ek_borger,timestamp,value,
                        1,2021-12-30 22:59:59, 1
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df,
        output_spec=PredictorSpec(
            values_df=str_to_df(predictor_df_str),
            interval_days=2,
            resolve_multiple_fn_name="max",
            fallback=np.NaN,
            feature_name="value",
        ),
        expected_values=[1],
    )


def test_multiple_citizens_predictor():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            1,2022-01-02 00:00:00
                            5,2022-01-02 00:00:00
                            5,2022-01-05 00:00:00
                            6,2022-01-05 00:00:00
                            """
    predictor_df_str = """dw_ek_borger,timestamp,value,
                        1,2021-12-30 00:00:01, 0
                        1,2022-01-01 00:00:00, 1
                        5,2022-01-01 00:00:00, 0
                        5,2022-01-04 00:00:01, 2
                        7,2022-01-05 00:00:00, 5
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=PredictorSpec(
            values_df=str_to_df(predictor_df_str),
            interval_days=2,
            fallback=np.NaN,
            feature_name="value",
            resolve_multiple_fn_name="max",
        ),
        expected_values=[0, 1, 0, 2, np.NaN],
    )


# Outcomes
def test_event_after_prediction_time():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    outcome_df_str = """dw_ek_borger,timestamp,value,
                        1,2022-01-01 00:00:01, 1
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=OutcomeSpec(
            values_df=str_to_df(outcome_df_str),
            interval_days=2,
            resolve_multiple_fn_name="max",
            incident=True,
            fallback=np.NaN,
            feature_name="value",
        ),
        expected_values=[1],
    )


def test_event_before_prediction():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    outcome_df_str = """dw_ek_borger,timestamp,value,
                        1,2021-12-30 23:59:59, 1.0
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=OutcomeSpec(
            values_df=str_to_df(outcome_df_str),
            interval_days=2,
            resolve_multiple_fn_name="max",
            incident=True,
            fallback=np.NaN,
            feature_name="value",
        ),
        expected_values=[np.NaN],
    )


def test_multiple_citizens_outcome():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            1,2022-01-02 00:00:00
                            5,2025-01-02 00:00:00
                            5,2025-08-05 00:00:00
                            """
    outcome_df_str = """dw_ek_borger,timestamp,value
                        1,2021-12-31 00:00:01, 1.0
                        1,2023-01-02 00:00:00, 1.0
                        5,2025-01-03 00:00:00, 1.0
                        5,2022-01-05 00:00:01, 1.0
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=OutcomeSpec(
            values_df=str_to_df(outcome_df_str),
            interval_days=2,
            resolve_multiple_fn_name="max",
            incident=True,
            fallback=np.NaN,
            feature_name="value",
        ),
        expected_values=[1, np.NaN, 1, np.NaN],
    )


def test_citizen_without_outcome():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    outcome_df_str = """dw_ek_borger,timestamp,value,
                        0,2021-12-31 00:00:01, 1.0
                        """

    assert_flattened_data_as_expected(
        prediction_times_df=prediction_times_df_str,
        output_spec=OutcomeSpec(
            values_df=str_to_df(outcome_df_str),
            interval_days=2,
            resolve_multiple_fn_name="max",
            incident=True,
            fallback=np.NaN,
            feature_name="value",
        ),
        expected_values=[np.NaN],
    )


def test_static_predictor():
    prediction_times_df = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            1,2021-12-31 00:00:01
                            1,2021-12-31 00:00:02
                            """
    static_predictor = """dw_ek_borger,date_of_birth
                        1,1994-12-31 00:00:01
                        """

    dataset = FlattenedDataset(prediction_times_df=str_to_df(prediction_times_df))
    dataset.add_static_info(
        static_spec=StaticSpec(
            values_df=str_to_df(static_predictor),
            input_col_name_override="date_of_birth",
            output_col_name_override="date_of_birth",
        )
    )

    expected_values = pd.DataFrame(
        {
            "date_of_birth": [
                "1994-12-31 00:00:01",
                "1994-12-31 00:00:01",
                "1994-12-31 00:00:01",
            ],
        },
    )

    pd.testing.assert_series_equal(
        left=dataset.df["date_of_birth"].reset_index(drop=True),
        right=expected_values["date_of_birth"].reset_index(drop=True),
        check_dtype=False,
    )


def test_add_age():
    prediction_times_df = """dw_ek_borger,timestamp,
                            1,1994-12-31 00:00:00
                            1,2021-12-30 00:00:00
                            1,2021-12-31 00:00:00
                            """
    static_predictor = """dw_ek_borger,date_of_birth
                        1,1994-12-31 00:00:00
                        """

    dataset = FlattenedDataset(prediction_times_df=str_to_df(prediction_times_df))
    dataset.add_age_and_date_of_birth(
        id2date_of_birth=str_to_df(static_predictor),
        date_of_birth_col_name="date_of_birth",
    )

    expected_values = pd.DataFrame(
        {
            "pred_age_in_years": [
                0.0,
                27.0,
                27.0,
            ],
        },
    )

    pd.testing.assert_series_equal(
        left=dataset.df["pred_age_in_years"].reset_index(drop=True),
        right=expected_values["pred_age_in_years"].reset_index(drop=True),
        check_dtype=False,
    )


def test_add_age_error():
    prediction_times_df = """dw_ek_borger,timestamp,
                            1,1994-12-31 00:00:00
                            1,2021-11-28 00:00:00
                            1,2021-12-31 00:00:00
                            """
    static_predictor = """dw_ek_borger,date_of_birth
                        1,94-12-31 00:00:00
                        """

    dataset = FlattenedDataset(prediction_times_df=str_to_df(prediction_times_df))

    with pytest.raises(ValueError):
        dataset.add_age_and_date_of_birth(
            id2date_of_birth=str_to_df(static_predictor),
            date_of_birth_col_name="date_of_birth",
        )


def test_incident_outcome_removing_prediction_times():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            1,2023-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            2,2023-12-30 00:00:00
                            3,2023-12-31 00:00:00
                            """

    event_times_str = """dw_ek_borger,timestamp,value,
                        1,2021-12-31 00:00:01, 1
                        2,2021-12-31 00:00:01, 1
                        """

    expected_df_str = """dw_ek_borger,timestamp,outc_value_within_2_days_max_fallback_nan_dichotomous,
                        1,2021-12-31 00:00:00, 1.0
                        2,2021-12-31 00:00:00, 1.0
                        3,2023-12-31 00:00:00, 0.0
                        """

    prediction_times_df = str_to_df(prediction_times_str)
    event_times_df = str_to_df(event_times_str)
    expected_df = str_to_df(expected_df_str)

    flattened_dataset = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
        n_workers=4,
    )

    flattened_dataset.add_temporal_outcome(
        output_spec=OutcomeSpec(
            values_df=event_times_df,
            interval_days=2,
            incident=True,
            fallback=np.NaN,
            feature_name="value",
            resolve_multiple_fn_name="max",
            col_main="value",
        ),
    )

    outcome_df = flattened_dataset.df.reset_index(drop=True)

    for col in expected_df.columns:
        pd.testing.assert_series_equal(
            outcome_df[col],
            expected_df[col],
            check_dtype=False,
        )


def test_add_multiple_static_predictors():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            1,2023-12-31 00:00:00
                            2,2021-12-31 00:00:00
                            2,2023-12-31 00:00:00
                            3,2023-12-31 00:00:00
                            """

    event_times_str = """dw_ek_borger,timestamp,value,
                        1,2021-12-31 00:00:01, 1
                        2,2021-12-31 00:00:01, 1
                        """

    expected_df_str = """dw_ek_borger,timestamp,outc_value_within_2_days_max_fallback_0_dichotomous,pred_age_in_years,pred_male
                        1,2021-12-31 00:00:00, 1.0,22.00,1
                        2,2021-12-31 00:00:00, 1.0,22.00,0
                        3,2023-12-31 00:00:00, 0.0,23.99,1
                        """

    birthdates_df_str = """dw_ek_borger,date_of_birth,
    1,2000-01-01,
    2,2000-01-02,
    3,2000-01-03"""

    male_df_str = """dw_ek_borger,male,
    1,1
    2,0
    3,1"""

    prediction_times_df = str_to_df(prediction_times_str)
    event_times_df = str_to_df(event_times_str)
    expected_df = str_to_df(expected_df_str)
    birthdates_df = str_to_df(birthdates_df_str)
    male_df = str_to_df(male_df_str)

    flattened_dataset = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
        n_workers=4,
    )

    output_spec = OutcomeSpec(
        col_main="value",
        values_df=event_times_df,
        interval_days=2,
        resolve_multiple_fn_name="max",
        fallback=0,
        incident=True,
        feature_name="value",
    )

    flattened_dataset.add_temporal_outcome(
        output_spec=output_spec,
    )

    flattened_dataset.add_age_and_date_of_birth(
        date_of_birth_col_name="date_of_birth", id2date_of_birth=birthdates_df
    )
    flattened_dataset.add_static_info(static_spec=StaticSpec(values_df=male_df))

    outcome_df = flattened_dataset.df

    for col in (
        "dw_ek_borger",
        "timestamp",
        "outc_value_within_2_days_max_fallback_0_dichotomous",
        "pred_age_in_years",
        "pred_male",
    ):
        pd.testing.assert_series_equal(
            outcome_df[col],
            expected_df[col],
            check_dtype=False,
        )


def test_add_temporal_predictors_then_temporal_outcome():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-11-05 00:00:00
                            2,2021-11-05 00:00:00
                            """

    predictors_df_str = """dw_ek_borger,timestamp,value,
                        1,2020-11-05 00:00:01, 1
                        2,2020-11-05 00:00:01, 1
                        2,2021-01-15 00:00:01, 3
                        """

    event_times_str = """dw_ek_borger,timestamp,value,
                        1,2021-11-05 00:00:01, 1
                        2,2021-11-05 00:00:01, 1
                        """

    expected_df_str = """dw_ek_borger,timestamp,prediction_time_uuid
                            2,2021-11-05,2-2021-11-05-00-00-00
                            1,2021-11-05,1-2021-11-05-00-00-00
                        """

    prediction_times_df = str_to_df(prediction_times_str)
    predictors_df = str_to_df(predictors_df_str)
    event_times_df = str_to_df(event_times_str)
    expected_df = str_to_df(expected_df_str)

    flattened_dataset = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
        n_workers=4,
    )

    predictor_spec_list = PredictorGroupSpec(
        values_df=[predictors_df],
        interval_days=[1, 365, 720],
        resolve_multiple_fn_name=["min"],
        fallback=[np.nan],
        allowed_nan_value_prop=[0],
        feature_name="value",
    ).create_combinations()

    flattened_dataset.add_temporal_predictors_from_pred_specs(
        predictor_specs=predictor_spec_list,
    )

    flattened_dataset.add_temporal_outcome(
        output_spec=OutcomeSpec(
            col_main="value",
            values_df=event_times_df,
            interval_days=2,
            resolve_multiple_fn_name="max",
            fallback=0,
            incident=True,
            feature_name="value",
        ),
    )

    outcome_df = flattened_dataset.df.set_index("dw_ek_borger").sort_index()
    expected_df = expected_df.set_index("dw_ek_borger").sort_index()

    for col in expected_df.columns:
        pd.testing.assert_series_equal(
            outcome_df[col],
            expected_df[col],
            check_index=False,
            check_dtype=False,
        )


def test_add_temporal_incident_binary_outcome():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-11-05 00:00:00
                            1,2021-11-01 00:00:00
                            1,2023-11-05 00:00:00
                            """

    event_times_str = """dw_ek_borger,timestamp,value,
                        1,2021-11-06 00:00:01, 1
                        """

    expected_df_str = """outc_value_within_2_days_max_fallback_nan_dichotomous,
    1
    0"""

    prediction_times_df = str_to_df(prediction_times_str)
    event_times_df = str_to_df(event_times_str)
    expected_df = str_to_df(expected_df_str)

    flattened_dataset = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
        n_workers=4,
    )

    flattened_dataset.add_temporal_outcome(
        output_spec=OutcomeSpec(
            values_df=event_times_df,
            interval_days=2,
            incident=True,
            fallback=np.NaN,
            feature_name="value",
            resolve_multiple_fn_name="max",
            col_main="t2d",
        ),
    )

    outcome_df = flattened_dataset.df

    for col in [c for c in expected_df.columns if "outc" in c]:
        for df in (outcome_df, expected_df):
            # Windows and Linux have different default dtypes for ints,
            # which is not a meaningful error here. So we force the dtype.
            if df[col].dtype == "int64":
                df[col] = df[col].astype("int32")
        pd.testing.assert_series_equal(outcome_df[col], expected_df[col])


# def test_add_tfidf_text_data():
#     prediction_times_str = """dw_ek_borger,timestamp,
#                             746430.0,1970-05-01 00:00:00
#                             765709.0,1971-05-14 22:04:00
#                             """

#     prediction_times_df = str_to_df(prediction_times_str)

#     flattened_dataset = FlattenedDataset(
#         prediction_times_df=prediction_times_df,
#         timestamp_col_name="timestamp",
#         id_col_name="dw_ek_borger",
#         n_workers=4,
#     )

#     synth_notes_df = data_loaders.get_all()["synth_notes"](featurizer="tfidf")

#     predictor_specs = PredictorGroupSpec(
#         values_df=[synth_notes_df],
#         interval_days=[1, 365, 720],
#         resolve_multiple_fn_name=["min"],
#         fallback=[np.nan],
#         allowed_nan_value_prop=[0],
#         loader_kwargs=[{"featurizer": "tfidf"}],
#         feature_name="tfidf",
#     ).create_combinations()

#     flattened_dataset.add_temporal_predictors_from_pred_specs(
#         predictor_specs=predictor_specs,
#     )

#     outcome_df = flattened_dataset.df

#     assert outcome_df.shape == (2, 33)

#     # 20 nas = 2 ids * 10 predictors with lookbehind 1 day. First get sum of each column. Then get sum of the row.
#     assert outcome_df.isna().sum().sum() == 20

#     # assert len()


# @pytest.mark.slow  # Only run if --runslow is passed to pytest
# def test_add_hf_text_data():
#     prediction_times_str = """dw_ek_borger,timestamp,
#                             746430.0,1970-05-01 00:00:00
#                             765709.0,1971-05-14 22:04:00
#                             """

#     prediction_times_df = str_to_df(prediction_times_str)

#     flattened_dataset = FlattenedDataset(
#         prediction_times_df=prediction_times_df,
#         timestamp_col_name="timestamp",
#         id_col_name="dw_ek_borger",
#         n_workers=4,
#     )

#     predictor_list = create_feature_combinations(
#         [
#             {
#                 "predictor_df": "synth_notes",
#                 "lookbehind_days": [1, 365, 720],
#                 "resolve_multiple_fn_name": "min",
#                 "fallback": np.nan,
#                 "loader_kwargs": {
#                     "featurizer": "huggingface",
#                     "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#                 },
#                 "new_col_name": [TEST_HF_EMBEDDINGS],
#             },
#         ],
#     )

#     flattened_dataset.add_temporal_predictors_from_list_of_argument_dictionaries(
#         predictors=predictor_list,
#     )

#     outcome_df = flattened_dataset.df

#     assert outcome_df.shape == (2, 1155)

#     # 768 nas = 2 ids * 384 predictors with lookbehind 1 day. First get sum of each column. Then get sum of the row.
#     assert outcome_df.isna().sum().sum() == 768


def test_chunk_text():
    text = "This is a test. This is another test. This is a third test. This is a fourth test."
    expected = [
        "This is a test.",
        "This is another test.",
        "This is a third",
        "test. This is a",
    ]

    assert _chunk_text(text, 4) == expected
