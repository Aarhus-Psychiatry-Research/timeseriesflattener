"""Example for how to check feature set integrity."""

from pathlib import Path

from psycopmlutils.data_checks.flattened.data_integrity import (
    save_feature_set_integrity_from_dir,
)

if __name__ == "__main__":
    subdir = Path(
        "E:/shared_resources/feature_sets/t2d/adminmanber_260_features_2022_08_26_14_10/",
    )

    save_feature_set_integrity_from_dir(
        feature_set_csv_dir=subdir,
        split_names=["train", "val", "test"],
    )
