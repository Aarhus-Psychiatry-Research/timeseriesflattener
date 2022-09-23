"""Tests feature creation dict."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from utils_for_testing import str_to_df

from psycopmlutils.timeseriesflattener.create_feature_combinations import (
    create_feature_combinations,
)
from psycopmlutils.timeseriesflattener.flattened_dataset import FlattenedDataset


def test_generate_two_features_from_dict():
    """Test generation of features from a dictionary."""

    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """

    event_times_str = """dw_ek_borger,timestamp,value,
                        1,2021-12-30 00:00:01, 1
                        1,2021-12-29 00:00:02, 2
                        """

    expected_df_str = """dw_ek_borger,timestamp,pred_event_times_df_within_1_days_max_fallback_0,pred_event_times_df_within_2_days_max_fallback_0,pred_event_times_df_within_3_days_max_fallback_0,pred_event_times_df_within_4_days_max_fallback_0
                        1,2021-12-31 00:00:00,1,2,2,2
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

    predictor_list = create_feature_combinations(
        [
            {
                "predictor_df": "event_times_df",
                "lookbehind_days": [1, 2, 3, 4],
                "resolve_multiple": "max",
                "fallback": 0,
                "source_values_col_name": "val",
            },
        ],
    )

    flattened_dataset.add_temporal_predictors_from_list_of_argument_dictionaries(
        predictors=predictor_list,
        predictor_dfs={"event_times_df": event_times_df},
    )

    for col in [
        "dw_ek_borger",
        "timestamp",
        "pred_event_times_df_within_1_days_max_fallback_0",
        "pred_event_times_df_within_2_days_max_fallback_0",
    ]:
        pd.testing.assert_series_equal(flattened_dataset.df[col], expected_df[col])


def test_output_independent_of_order_of_input():
    """Test generation of features from a dictionary."""

    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            1,2021-12-31 00:00:01
                            2,2021-12-31 00:00:00
                            2,2021-12-31 00:00:01
                            """

    prediction_times_str2 = """dw_ek_borger,timestamp,
                            2,2021-12-31 00:00:00
                            2,2021-12-31 00:00:01
                            1,2021-12-31 00:00:00
                            1,2021-12-31 00:00:01
                            """

    prediction_times_df = str_to_df(prediction_times_str)
    prediction_times_df2 = str_to_df(prediction_times_str2)

    flattened_dataset1 = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
        n_workers=1,
    )

    flattened_dataset2 = FlattenedDataset(
        prediction_times_df=prediction_times_df2,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
        n_workers=1,
    )

    predictor_list = create_feature_combinations(
        [
            {
                "predictor_df": "predictor_df",
                "lookbehind_days": [1, 2, 3, 4],
                "resolve_multiple": "max",
                "fallback": 0,
                "source_values_col_name": "val",
            },
        ],
    )

    predictor_list2 = create_feature_combinations(
        [
            {
                "predictor_df": "predictor_df",
                "lookbehind_days": [1, 2, 3, 4],
                "resolve_multiple": "max",
                "fallback": 0,
                "source_values_col_name": "val",
            },
        ],
    )

    predictor_str = """dw_ek_borger,timestamp,value,
                        1,2021-12-30 00:00:01, 1
                        1,2021-12-29 00:00:02, 4
                        2,2021-12-30 00:00:01, 2
                        2,2021-12-29 00:00:02, 3
                        """

    predictor_df = str_to_df(predictor_str)

    flattened_dataset1.add_temporal_predictors_from_list_of_argument_dictionaries(
        predictors=predictor_list,
        predictor_dfs={"predictor_df": predictor_df},
    )

    flattened_dataset2.add_temporal_predictors_from_list_of_argument_dictionaries(
        predictors=predictor_list2,
        predictor_dfs={"predictor_df": predictor_df},
    )

    # We don't care about indeces. Sort to match the ordering.
    assert_frame_equal(
        flattened_dataset1.df.sort_values(["dw_ek_borger", "timestamp"]).reset_index(
            drop=True,
        ),
        flattened_dataset2.df.sort_values(["dw_ek_borger", "timestamp"]).reset_index(
            drop=True,
        ),
        check_index_type=False,
        check_like=True,
    )


def test_add_df_from_catalogue():
    """Test generation of features from a dictionary."""

    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """

    expected_df_str = """dw_ek_borger,timestamp,pred_load_event_times_within_1_days_max_fallback_0,pred_load_event_times_within_2_days_max_fallback_0,
                        1,2021-12-31 00:00:00,1,2,
    """

    prediction_times_df = str_to_df(prediction_times_str)
    expected_df = str_to_df(expected_df_str)

    flattened_dataset = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
        n_workers=4,
    )

    predictor_list = create_feature_combinations(
        [
            {
                "predictor_df": "load_event_times",
                "lookbehind_days": [1, 2, 3, 4],
                "resolve_multiple": "max",
                "fallback": 0,
                "source_values_col_name": "val",
            },
        ],
    )

    flattened_dataset.add_temporal_predictors_from_list_of_argument_dictionaries(
        predictors=predictor_list,
    )

    for col in [
        "dw_ek_borger",
        "timestamp",
        "pred_load_event_times_within_1_days_max_fallback_0",
        "pred_load_event_times_within_2_days_max_fallback_0",
    ]:
        pd.testing.assert_series_equal(flattened_dataset.df[col], expected_df[col])


def test_wrong_formatting():
    """Test generation of features from a dictionary."""

    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """

    prediction_times_df = str_to_df(prediction_times_str)

    predictor_str = """dw_ek_borger,timestamp,value,
                        1,2021-12-30 00:00:01, 1
                        1,2021-12-29 00:00:02, 2
                        2,2021-12-30 00:00:01, 1
                        2,2021-12-29 00:00:02, 2
                        """

    predictor_df = str_to_df(predictor_str)

    flattened_dataset = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
        n_workers=4,
    )

    with pytest.raises(ValueError):
        unresolvable_df = [
            {
                "predictor_df": "df_doesn_not_exist",
                "lookbehind_days": 1,
                "resolve_multiple": "max",
                "fallback": 0,
                "source_values_col_name": "val",
            },
        ]

        flattened_dataset.add_temporal_predictors_from_list_of_argument_dictionaries(
            predictors=unresolvable_df,
        )

    with pytest.raises(ValueError):
        unresolvable_resolve_multiple = [
            {
                "predictor_df": "predictor_df",
                "lookbehind_days": 1,
                "resolve_multiple": "does_not_exist",
                "fallback": 0,
                "source_values_col_name": "val",
            },
        ]

        flattened_dataset.add_temporal_predictors_from_list_of_argument_dictionaries(
            predictors=unresolvable_resolve_multiple,
            predictor_dfs={"predictor_df": predictor_df},
        )

    with pytest.raises(ValueError):
        lookbehind_days_not_int = [
            {
                "predictor_df": "predictor_df",
                "lookbehind_days": "1",
                "resolve_multiple": "does_not_exist",
                "fallback": 0,
                "source_values_col_name": "val",
            },
        ]

        flattened_dataset.add_temporal_predictors_from_list_of_argument_dictionaries(
            predictors=lookbehind_days_not_int,
            predictor_dfs={"predictor_df": predictor_df},
        )
