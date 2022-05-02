from psycopmlutils.timeseriesflattener.create_feature_combinations import (
    create_feature_combinations,
    dict_has_list_in_any_value,
    list_has_dict_with_list_as_value,
)


def test_skip_all_if_no_need_to_process():
    input = [
        {
            "predictor_df": "prediction_times_df",
            "source_values_col_name": "val",
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 0,
        }
    ]

    assert create_feature_combinations(input) == input


def test_skip_one_if_no_need_to_process():
    input = [
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

    assert create_feature_combinations(input) == expected_output


def test_list_has_dict_with_list_as_val():
    test_pos_dataset = [
        {
            "lookbehind_days": [1],
            "resolve_multiple": "max",
            "fallback": 0,
            "source_values_col_name": "val",
        }
    ]

    assert list_has_dict_with_list_as_value(test_pos_dataset)

    test_neg_dataset = [
        {
            "lookbehind_days": 1,
            "resolve_multiple": "max",
            "fallback": 0,
            "source_values_col_name": "val",
        }
    ]

    assert not list_has_dict_with_list_as_value(test_neg_dataset)


def test_dict_has_list_as_val():
    test_pos_dict = {
        "lookbehind_days": [1, 30],
        "resolve_multiple": "max",
        "fallback": [0, 1],
        "source_values_col_name": "val",
    }

    assert dict_has_list_in_any_value(test_pos_dict)

    test_neg_dict = {
        "lookbehind_days": 1,
        "resolve_multiple": "max",
        "fallback": 0,
        "source_values_col_name": "val",
    }

    assert not dict_has_list_in_any_value(test_neg_dict)


def test_create_feature_combinations():
    input = [
        {
            "predictor_df": "prediction_times_df",
            "source_values_col_name": "val",
            "lookbehind_days": [1, 30],
            "resolve_multiple": "max",
            "fallback": 0,
        }
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

    assert create_feature_combinations(arg_sets=input) == expected_output


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
