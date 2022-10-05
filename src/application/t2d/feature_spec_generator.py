from typing import Any, Callable, Iterable, Optional, Union


def generate_feature_specification(  # pylint: disable=dangerous-default-value
    dfs: Iterable[str],
    lookbehind_days: Iterable = (
        365,
        730,
        1825,
        9999,
    ),  # Not a problem here, since the function is only ever called once.
    resolve_multiple: Iterable[Union[Callable, str]] = ("mean", "max", "min"),
    fallback: Optional[Any] = 0,
    allowed_nan_value_prop: float = 0.0,
    values_to_load: Optional[str] = None,
):
    """Generate feature specifications.

    Args:
        dfs (Iterable[str]): Dataframes to generate feature specifications for.
        lookbehind_days (Iterable, optional): Iterable of lookbehind days. Defaults to (365, 730, 1825, 9999).
        resolve_multiple (Iterable, optional): Iterable of resolve multiple options. Defaults to ("mean", "max", "min").
        fallback (any, optional): Fallback value. Defaults to 0.
        allowed_nan_value_prop (float, optional): Allowed proportion of NaN values. Defaults to 0.0.
        values_to_load (str, optional): Which values to load for medications. Takes "all", "numerical" or "numerical_and_coerce". Defaults to None.

    Returns:
        list[dict[str, Any]]: List of feature specifications.
    """
    return [
        {
            "predictor_df": df,
            "lookbehind_days": lookbehind_days,
            "resolve_multiple": resolve_multiple,
            "fallback": fallback,
            "allowed_nan_value_prop": allowed_nan_value_prop,
            "values_to_load": values_to_load,
        }
        for df in dfs
    ]
