"""Example for loading diagnoses."""

import psycopmlutils.loaders.raw.load_diagnoses as d
from psycopmlutils.data_checks.raw.check_predictor_lists import (
    check_feature_combinations_return_correct_dfs,
)

if __name__ == "__main__":
    df = d.sleep_apnea(n=100)

    input_dict = [{"predictor_df": "sleep_apnea"}]

    check_feature_combinations_return_correct_dfs(
        predictor_dict_list=input_dict,
        n_rows=100,
    )
