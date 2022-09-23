"""Example for how to describe features."""

from pathlib import Path

from psycopmlutils.data_checks.flattened.feature_describer import (
    save_feature_description_from_dir,
)
from src.application.t2d.features_blood_samples import get_lab_feature_spec
from src.application.t2d.features_diagnoses import get_diagnosis_feature_spec
from src.application.t2d.features_medications import get_medication_feature_spec

if __name__ == "__main__":
    feature_set_dir = Path(
        "C:/shared_resources/feature_sets/t2d/adminmanber_260_features_2022_08_26_14_10/",
    )

    feature_set_path = (
        feature_set_dir
        / "adminmanber_psycop_t2d_260_features_2022_08_26_14_10_train.csv"
    )
    out_dir = feature_set_dir / "feature_description"

    resolve_multiple = ["latest", "max", "min", "mean"]
    lookbehind_days = [365, 730, 1825, 9999]

    LAB_PREDICTORS = get_lab_feature_spec(
        resolve_multiple=resolve_multiple,
        lookbehind_days=lookbehind_days,
    )

    DIAGNOSIS_PREDICTORS = get_diagnosis_feature_spec(
        resolve_multiple=resolve_multiple,
        lookbehind_days=lookbehind_days,
        fallback=0,
    )

    MEDICATION_PREDICTORS = get_medication_feature_spec(
        lookbehind_days=lookbehind_days,
        resolve_multiple=["count"],
        fallback=0,
    )

    PREDICTOR_LIST = MEDICATION_PREDICTORS + DIAGNOSIS_PREDICTORS + LAB_PREDICTORS

    save_feature_description_from_dir(
        feature_set_csv_dir=feature_set_dir,
        predictor_dicts=PREDICTOR_LIST,
    )
