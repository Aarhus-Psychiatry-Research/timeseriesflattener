"""Generate raw float dataframe."""

from psycop_ml_utils.synth_data_generator.synth_col_generators import (
    generate_data_columns,
)
from timeseriesflattener.utils import PROJECT_ROOT

if __name__ == "__main__":
    # Get project root directory

    column_specs = [
        {
            "entity_id": {
                "column_type": "uniform_int",
                "min": 0,
                "max": 10_000,
            },
        },
        {
            "timestamp": {
                "column_type": "datetime_uniform",
                "min": -5 * 365,
                "max": 0 * 365,
            },
        },
        {"value": {"column_type": "uniform_float", "min": 0, "max": 10}},
    ]

    for i in (1, 2):
        df = generate_data_columns(
            predictors=column_specs,
            n_samples=100_000,
        )

        df.to_csv(
            PROJECT_ROOT / "tests" / "test_data" / "raw" / f"synth_raw_float_{i}.csv",
            index=False,
        )
