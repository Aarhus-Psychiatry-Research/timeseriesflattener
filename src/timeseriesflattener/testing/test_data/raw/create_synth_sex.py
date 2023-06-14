"""Generate raw binary dataframe."""

from pathlib import Path

from psycop_ml_utils.synth_data_generator.synth_col_generators import (
    generate_data_columns,
)

if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).resolve().parents[3]

    column_specs = [
        {
            "entity_id": {
                "column_type": "uniform_int",
                "min": 0,
                "max": 10_000,
            },
        },
        {"female": {"column_type": "uniform_int", "min": 0, "max": 2}},
    ]

    df = generate_data_columns(
        predictors=column_specs,
        n_samples=100_000,
    )

    df = df.groupby("entity_id").last().reset_index()

    df.to_csv(
        project_root / "tests" / "test_data" / "raw" / "synth_sex.csv",
        index=False,
    )
