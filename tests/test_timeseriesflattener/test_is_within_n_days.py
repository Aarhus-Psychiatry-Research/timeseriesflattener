from datetime import datetime
import pytest
from timeseriesflattener.flattened_dataset import is_within_n_days

# Seems basic, but had an error here, so included as regression tests.


def test_lookahead():
    assert is_within_n_days(
        direction="ahead",
        prediction_timestamp=datetime(2022, 1, 1),
        event_timestamp=datetime(2022, 1, 2),
        interval_days=2,
    )


def test_lookbehind():
    assert is_within_n_days(
        direction="behind",
        prediction_timestamp=datetime(2022, 1, 2),
        event_timestamp=datetime(2022, 1, 1),
        interval_days=2,
    )


def test_equal():
    assert not is_within_n_days(
        direction="behind",
        prediction_timestamp=datetime(2022, 1, 2),
        event_timestamp=datetime(2022, 1, 2),
        interval_days=2,
    )

    assert not is_within_n_days(
        direction="ahead",
        prediction_timestamp=datetime(2022, 1, 2),
        event_timestamp=datetime(2022, 1, 2),
        interval_days=2,
    )


def test_valueerror():
    with pytest.raises(ValueError):
        is_within_n_days(
            direction="up",
            prediction_timestamp=datetime(2022, 1, 2),
            event_timestamp=datetime(2022, 1, 2),
            interval_days=2,
        )
