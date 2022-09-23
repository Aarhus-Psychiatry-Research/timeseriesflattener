"""Feature specifications for T2D, diagnoses."""

from typing import Optional


def get_diagnosis_feature_spec(  # pylint: disable=dangerous-default-value
    lookbehind_days=None,  # Not a problem here, since the function is only ever called once.
    resolve_multiple=None,
    fallback: Optional[any] = 0,
):
    """Create diagnosis feature combinations.

    Args:
        lookbehind_days (list, optional): list of lookbehind days. Defaults to [365, 730, 1825, 9999].
        resolve_multiple (list, optional): list of resolve multiple options. Defaults to ["mean", "max", "min"].
        fallback (any, optional): Fallback value. Defaults to 0.

    Returns:
        _type_: _description_
    """

    if lookbehind_days is None:
        lookbehind_days = [
            365,
            730,
            1825,
            9999,
        ]

    if resolve_multiple is None:
        resolve_multiple = ["mean", "max", "min"]

    dfs = [
        "essential_hypertension",
        "hyperlipidemia",
        "polycystic_ovarian_syndrome",
        "sleep_apnea",
    ]

    # As list comprehension:
    return [
        {
            "predictor_df": df,
            "lookbehind_days": lookbehind_days,
            "resolve_multiple": resolve_multiple,
            "fallback": fallback,
            "allowed_nan_value_prop": 0.0,
        }
        for df in dfs
    ]
