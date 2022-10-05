"""Example for loading structured SFIs."""

import psycop_feature_generation.loaders.raw.load_structured_sfi as struct_sfi_loader
from psycop_feature_generation.data_checks.raw.check_predictor_lists import (
    check_feature_combinations_return_correct_dfs,
)

if __name__ == "__main__":
    df = struct_sfi_loader.selvmordsrisiko(n_rows=1000)

    input_dict = [
        {
            "predictor_df": "selvmordsrisiko",
            "allowed_nan_value_prop": 0.01,
        },
    ]

    check_feature_combinations_return_correct_dfs(
        predictor_dict_list=input_dict,
        n_rows=1000,
    )
