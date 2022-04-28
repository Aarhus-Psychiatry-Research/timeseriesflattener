from typing import Dict, List, Union


def list_has_dict_with_list_as_value(
    list_of_dicts: List[Dict[str, Union[str, list]]]
) -> bool:
    """Checks if any dict in a list of dicts has a value that is a list.

    Args:
        list_of_dicts (List[Dict[str, Union[str, list]]]): A list of dicts.

    Returns:
        bool
    """
    for dict in list_of_dicts:
        if dict_has_list_in_any_value(dict):
            return True

    return False


def dict_has_list_in_any_value(dict: Dict[str, Union[str, list]]) -> bool:
    """
    Checks if a dict has any values that are lists
    """
    for value in dict.values():
        if type(value) == list:
            return True
    return False


def create_feature_combinations(
    arg_sets: List[Dict[str, Union[str, list]]],
) -> List[Dict[str, Union[str, float, int]]]:
    """Create feature combinations from a dictionary of feature specifications. See example for shape.

    Args:
        arg_sets (List[Dict[str, Union[str, list]]]): A set of argument sets for .add_predictor. See example for shape.

    Returns:
        List[Dict[str, Union[str, float, int]]]: List of all possible combinations of the arguments.

    Example:
        >>> input = [
        >>>     {
        >>>         "predictor_df": "prediction_times_df",
        >>>         "source_values_col_name": "val",
        >>>         "lookbehind_days": [1, 30],
        >>>         "resolve_multiple": "get_max_value_from_list_of_events",
        >>>         "fallback": 0,
        >>>     }
        >>> ]
        >>> print(create_feature_combinations(arg_sets=input))
        >>> [
        >>>     {
        >>>         "predictor_df": "prediction_times_df",
        >>>         "lookbehind_days": 1,
        >>>         "resolve_multiple": "get_max_value_from_list_of_events",
        >>>         "fallback": 0,
        >>>         "source_values_col_name": "val",
        >>>     },
        >>>     {
        >>>         "predictor_df": "prediction_times_df",
        >>>         "lookbehind_days": 30,
        >>>         "resolve_multiple": "get_max_value_from_list_of_events",
        >>>         "fallback": 0,
        >>>         "source_values_col_name": "val",
        >>>     },
        >>> ]
    """
    output_arg_sets = []

    if not list_has_dict_with_list_as_value(arg_sets):
        # If no arg_sets contain lists as values, no need for further processing
        return arg_sets
    else:
        for arg_set in arg_sets:
            if not dict_has_list_in_any_value(arg_set):
                # If arg_set contains no lists, just append it to output_list
                output_arg_sets.append(arg_set)
            else:
                for arg_name, arg_value in arg_set.items():
                    hit_arg_with_list_as_value = False

                    if isinstance(arg_value, list):
                        hit_arg_with_list_as_value = True

                        for item in arg_value:
                            i_arg_set = arg_set.copy()
                            i_arg_set[arg_name] = item

                            output_arg_sets.append(i_arg_set)

                    if hit_arg_with_list_as_value:
                        # Break here to avoid adding duplicates.
                        # If e.g. two args with list, [0,1] and [0,1], not breaking would result in
                        # four new arg_sets: [0,0], [1,0], [0,0] and [1,1]
                        break

        return create_feature_combinations(output_arg_sets)
