"""Generate dataframe with prediction times."""

from psycopmlutils.synth_data_generator.synth_col_generators import (
    generate_data_columns,
)
from psycopmlutils.utils import PROJECT_ROOT

if __name__ == "__main__":
    # Get project root directory

    column_specs = {
        "dw_ek_borger": {
            "column_type": "uniform_int",
            "min": 0,
            "max": 10_000,
        },
        "timestamp": {
            "column_type": "datetime_uniform",
            "min": -5 * 365,
            "max": 0 * 365,
        },
    }

    df = generate_data_columns(
        predictors=column_specs,
        n_samples=10_000,
    )

    df.to_csv(
        PROJECT_ROOT / "tests" / "test_data" / "raw" / "synth_prediction_times.csv",
        index=False,
    )
