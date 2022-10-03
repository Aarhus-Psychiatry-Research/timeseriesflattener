"""Functionality for taking a dictionary of feature combinations where values
are lists, and then creating each possible permutation."""

import itertools
from typing import Any, Union

from psycop_feature_generation.utils import assert_no_duplicate_dicts_in_list


def create_feature_combinations_from_dict(
    d: dict[str, Union[str, list]],
) -> list[dict[str, Union[str, float, int]]]:
    """Create feature combinations from a dictionary of feature specifications.
    Only unpacks the top level of lists.

    Args:
        d (dict[str]): A dictionary of feature specifications.

    Returns:
        list[dict[str]]: list of all possible combinations of the arguments.
    """

    # Make all elements iterable
    d = {k: v if isinstance(v, list) else [v] for k, v in d.items()}
    keys, values = zip(*d.items())
    # Create all combinations of top level elements
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts


def create_feature_combinations(
    arg_sets: Union[list[dict[str, Union[str, list]]], dict[str, Union[str, list]]],
) -> list[dict[str, Any]]:
    """Create feature combinations from a dictionary or list of dictionaries of
    feature specifications.

    Args:
        arg_sets (Union[list[dict[str, Union[str, list]]], dict[str, Union[str, list]]]):
            dict/list of dicts containing arguments for .add_predictor.

    Returns:
        list[dict[str, Union[str, float, int]]]: All possible combinations of
            arguments.

    Example:
        >>> input = [
        >>>     {
        >>>         "predictor_df": "prediction_times_df",
        >>>         "source_values_col_name": "val",
        >>>         "lookbehind_days": [1, 30],
        >>>         "resolve_multiple": "max",
        >>>         "fallback": 0,
        >>>     }
        >>> ]
        >>> print(create_feature_combinations(arg_sets=input))
        >>> [
        >>>     {
        >>>         "predictor_df": "prediction_times_df",
        >>>         "lookbehind_days": 1,
        >>>         "resolve_multiple": "max",
        >>>         "fallback": 0,
        >>>         "source_values_col_name": "val",
        >>>     },
        >>>     {
        >>>         "predictor_df": "prediction_times_df",
        >>>         "lookbehind_days": 30,
        >>>         "resolve_multiple": "max",
        >>>         "fallback": 0,
        >>>         "source_values_col_name": "val",
        >>>     },
        >>> ]
    """
    if isinstance(arg_sets, dict):
        arg_sets = [arg_sets]
    feature_combinations = []
    for arg_set in arg_sets:
        feature_combinations.extend(create_feature_combinations_from_dict(arg_set))

    assert_no_duplicate_dicts_in_list(feature_combinations)

    return feature_combinations


__all__ = [
    "create_feature_combinations",
    "create_feature_combinations_from_dict",
]
