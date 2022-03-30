from typing import Dict, List, Union, Tuple
from datetime import datetime
from timeseriesflattener.flattened_dataset import FlattenedDataset


def get_max_value_from_list_of_events(
    list_of_events: List[Tuple[Union[datetime, float]]]
) -> float:
    """Gets the max value from a list of events.

    Args:
        list_of_events (List[Tuple[Union[datetime, float]]]): A list of events.
            Shaped like [(timestamp1: val1), (timestamp2: val2)]

    Returns:
        float: The max value.
    """
    max_val = max([event[1] for event in list_of_events])

    return max_val


def get_min_value_from_list_of_events(
    list_of_events: List[Tuple[Union[datetime, float]]]
) -> float:
    """Gets the min value from a list of events.

    Args:
        list_of_events (List[Tuple[Union[datetime, float]]]): A list of events.
            Shaped like [(timestamp1: val1), (timestamp2: val2)]

    Returns:
        float: The min value.
    """
    return min([event[1] for event in list_of_events])


def get_avg_value_from_list_of_events(
    list_of_events: List[Tuple[Union[datetime, float]]]
) -> float:
    """Gets the min value from a list of events.

    Args:
        list_of_events (List[Tuple[Union[datetime, float]]]): A list of events.
            Shaped like [(timestamp1: val1), (timestamp2: val2)]

    Returns:
        float: The avg value.
    """
    vals = [event[1] for event in list_of_events]

    return sum(vals) / len(vals)


def get_latest_value_from_list_of_events(
    list_of_events: List[Tuple[Union[datetime, float]]]
) -> float:
    """Sorts the list by the timestamp and gets the value for the list with the earliest timestamp.

    Args:
        list_of_events (List[Tuple[Union[datetime, float]]]): A list of events.
            Shaped like [(timestamp1: val1), (timestamp2: val2)]

    Returns:
        float: The value for the latest timestamp.
    """
    list_of_events.sort(key=lambda event: event[0], reverse=True)

    return list_of_events[0][1]


def get_earliest_value_from_list_of_events(
    list_of_events: List[Tuple[Union[datetime, float]]]
) -> float:
    """Sorts the list by the timestamp and gets the value for the list with the earliest timestamp.

    Args:
        list_of_events (List[Tuple[Union[datetime, float]]]): A list of events.
            Shaped like [(timestamp1: val1), (timestamp2: val2)]

    Returns:
        float: The value for the earliest timestamp.
    """
    list_of_events.sort(key=lambda event: event[0], reverse=False)

    return list_of_events[0][1]
