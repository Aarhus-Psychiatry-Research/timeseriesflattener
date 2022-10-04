"""Example for loading diagnoses."""

import psycop_feature_generation.loaders.raw.load_diagnoses as d
from psycop_feature_generation.data_checks.raw.check_predictor_lists import (
    check_feature_combinations_return_correct_dfs,
)

if __name__ == "__main__":
    df = d.sleep_apnea(n_rows=100)

    input_dict = [{"predictor_df": "sleep_apnea", "allowed_nan_value_prop": 0.01}]

    check_feature_combinations_return_correct_dfs(
        predictor_dict_list=input_dict,
        n_rows=100,
    )
