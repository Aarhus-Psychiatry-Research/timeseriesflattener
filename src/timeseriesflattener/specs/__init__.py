from __future__ import annotations

import datetime as dt


def _lookdistance_to_timedelta_days(
    lookdistance: float | tuple[float, float],
) -> tuple[dt.timedelta, dt.timedelta]:
    if isinstance(lookdistance, tuple):
        return (dt.timedelta(days=lookdistance[0]), dt.timedelta(days=lookdistance[1]))
    return (dt.timedelta(days=0), dt.timedelta(days=lookdistance))
