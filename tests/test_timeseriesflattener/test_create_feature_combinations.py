"""Tests for feature_combination creation."""

# pylint: disable=missing-function-docstring

from psycopmlutils.timeseriesflattener.create_feature_combinations import (
    create_feature_combinations,
)


def test_skip_all_if_no_need_to_process():
    feature_spec_list = [
        {
            "predictor_df": "prediction_times_df",
            "source_values_col_name": "val",
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 0,
        },
    ]

    assert create_feature_combinations(feature_spec_list) == feature_spec_list


def test_skip_one_if_no_need_to_process():
    feature_spec_list = [
        {
            "predictor_df": "prediction_times_df",
            "source_values_col_name": "val",
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 0,
        },
        {
            "predictor_df": "prediction_times_df",
            "source_values_col_name": "val",
            "lookbehind_days": [1],
            "resolve_multiple": "max",
            "fallback": 0,
        },
    ]

    expected_output = [
        {
            "predictor_df": "prediction_times_df",
            "source_values_col_name": "val",
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 0,
        },
        {
            "predictor_df": "prediction_times_df",
            "source_values_col_name": "val",
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 0,
        },
    ]

    assert create_feature_combinations(feature_spec_list) == expected_output


def test_create_feature_combinations():
    feature_spec_list = [
        {
            "predictor_df": "prediction_times_df",
            "source_values_col_name": "val",
            "lookbehind_days": [1, 30],
            "resolve_multiple": "max",
            "fallback": 0,
        },
    ]

    expected_output = [
        {
            "predictor_df": "prediction_times_df",
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 0,
            "source_values_col_name": "val",
        },
        {
            "predictor_df": "prediction_times_df",
            "lookbehind_days": 30,
            "resolve_multiple": "max",
            "fallback": 0,
            "source_values_col_name": "val",
        },
    ]

    assert create_feature_combinations(arg_sets=feature_spec_list) == expected_output


def test_create_multiple_feature_combinations():
    predictor_dict = [
        {
            "predictor_df": "prediction_times_df",
            "source_values_col_name": "val",
            "lookbehind_days": [1, 30],
            "resolve_multiple": "max",
            "fallback": [0, 1],
        },
    ]

    expected_output = [
        {
            "predictor_df": "prediction_times_df",
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 0,
            "source_values_col_name": "val",
        },
        {
            "predictor_df": "prediction_times_df",
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 1,
            "source_values_col_name": "val",
        },
        {
            "predictor_df": "prediction_times_df",
            "lookbehind_days": 30,
            "resolve_multiple": "max",
            "fallback": 0,
            "source_values_col_name": "val",
        },
        {
            "predictor_df": "prediction_times_df",
            "lookbehind_days": 30,
            "resolve_multiple": "max",
            "fallback": 1,
            "source_values_col_name": "val",
        },
    ]

    assert create_feature_combinations(arg_sets=predictor_dict) == expected_output


def test_create_multiple_feature_combinations_from_multiple_columns():
    predictor_dict = [
        {
            "predictor_df": "prediction_times_df",
            "source_values_col_name": "val",
            "lookbehind_days": [1, 30],
            "resolve_multiple": "max",
            "fallback": [0, 1],
        },
        {
            "predictor_df": "prediction_times_df2",
            "source_values_col_name": "val",
            "lookbehind_days": [1, 30],
            "resolve_multiple": "max",
            "fallback": [0, 1],
        },
    ]

    expected_output = [
        {
            "predictor_df": "prediction_times_df",
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 0,
            "source_values_col_name": "val",
        },
        {
            "predictor_df": "prediction_times_df",
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 1,
            "source_values_col_name": "val",
        },
        {
            "predictor_df": "prediction_times_df",
            "lookbehind_days": 30,
            "resolve_multiple": "max",
            "fallback": 0,
            "source_values_col_name": "val",
        },
        {
            "predictor_df": "prediction_times_df",
            "lookbehind_days": 30,
            "resolve_multiple": "max",
            "fallback": 1,
            "source_values_col_name": "val",
        },
        {
            "predictor_df": "prediction_times_df2",
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 0,
            "source_values_col_name": "val",
        },
        {
            "predictor_df": "prediction_times_df2",
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 1,
            "source_values_col_name": "val",
        },
        {
            "predictor_df": "prediction_times_df2",
            "lookbehind_days": 30,
            "resolve_multiple": "max",
            "fallback": 0,
            "source_values_col_name": "val",
        },
        {
            "predictor_df": "prediction_times_df2",
            "lookbehind_days": 30,
            "resolve_multiple": "max",
            "fallback": 1,
            "source_values_col_name": "val",
        },
    ]

    assert create_feature_combinations(arg_sets=predictor_dict) == expected_output
