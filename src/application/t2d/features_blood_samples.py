"""Feature specificatin for T2D blood samples."""

from typing import Optional

import numpy as np


def get_lab_feature_spec(  # pylint: disable=dangerous-default-value
    # Not a problem since the function is only called once.
    lookbehind_days: Optional[list[int]] = None,
    resolve_multiple: Optional[list[str]] = None,
    values_to_load="all",
) -> list:
    """Get feature specification for T2D blood samples.

    Args:
        lookbehind_days (list[int], optional): Defaults to None.
        resolve_multiple (list[str], optional): Defaults to None.
        values_to_load (str, optional): Defaults to "all".

    Returns:
        list: Feature specification.
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
        "hba1c",
        "alat",
        "hdl",
        "ldl",
        "scheduled_glc",
        "unscheduled_p_glc",
        "triglycerides",
        "fasting_ldl",
        "alat",
        "crp",
        "egfr",
        "albumine_creatinine_ratio",
    ]

    return [
        {
            "predictor_df": df,
            "lookbehind_days": lookbehind_days,
            "resolve_multiple": resolve_multiple,
            "fallback": np.nan,
            "allowed_nan_value_prop": 0.0,
            "values_to_load": values_to_load,
        }
        for df in dfs
    ]
