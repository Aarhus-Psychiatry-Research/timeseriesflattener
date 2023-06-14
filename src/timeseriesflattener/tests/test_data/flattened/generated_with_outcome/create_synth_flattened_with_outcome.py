"""Generate synth data with outcome."""
from pathlib import Path

import numpy as np
from psycop_ml_utils.synth_data_generator.synth_prediction_times_generator import (
    generate_synth_data,
)

if __name__ == "__main__":
    column_specifications = {
        "citizen_ids": {"column_type": "uniform_int", "min": 0, "max": 1_200_001},
        "timestamp": {"column_type": "datetime_uniform", "min": 0, "max": 5 * 365},
        "timestamp_outcome": {
            "column_type": "datetime_uniform",
            "min": 1 * 365,
            "max": 6 * 365,
        },
        "pred_hba1c_within_100_days_max_fallback_np.nan": {
            "column_type": "normal",
            "mean": 48,
            "sd": 5,
            "fallback": np.nan,
        },
        "pred_hdl_within_100_days_max_fallback_np.nan": {
            "column_type": "normal",
            "mean": 1,
            "sd": 0.5,
            "min": 0,
            "fallback": np.nan,
        },
    }

    synth_df = generate_synth_data(
        predictors=column_specifications,
        outcome_column_name="outc_dichotomous_t2d_within_30_days_max_fallback_0",
        n_samples=10_000,
        logistic_outcome_model="1*pred_hba1c_within_100_days_max_fallback_nan+1*pred_hdl_within_100_days_max_fallback_nan",
        prob_outcome=0.08,
    )

    synth_df.describe()

    save_path = Path(__file__).parent.parent.parent.parent
    synth_df.to_csv(save_path / "tests" / "test_data" / "synth_prediction_data.csv")
