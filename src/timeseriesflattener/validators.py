from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from iterpy.iter import Iter

if TYPE_CHECKING:
    from .intermediary import AggregatedValueFrame, TimeDeltaFrame, TimeMaskedFrame
    from .specs.value import ValueFrame
    from .specs.outcome import BooleanOutcomeSpec, OutcomeSpec
    from .specs.prediction_times import PredictionTimeFrame
    from .specs.temporal import PredictorSpec
    from .specs.static import StaticFrame
    from .specs.timedelta import TimeDeltaSpec
    from .specs.timestamp import TimestampValueFrame


@dataclass(frozen=True)
class SpecColumnError(Exception):
    description: str


def validate_col_name_columns_exist(
    obj: PredictionTimeFrame
    | ValueFrame
    | TimeMaskedFrame
    | AggregatedValueFrame
    | TimeDeltaFrame
    | TimestampValueFrame
    | PredictorSpec
    | OutcomeSpec
    | BooleanOutcomeSpec
    | TimeDeltaSpec
    | StaticFrame,
):
    missing_columns = (
        Iter(dir(obj))
        .filter(lambda attr_name: attr_name.endswith("_col_name"))
        .map(lambda attr_name: getattr(obj, attr_name))
        .filter(lambda col_name: col_name not in obj.df.columns)
        .to_list()
    )

    if len(missing_columns) > 0:
        raise SpecColumnError(
            f"""Missing columns: {missing_columns} in dataframe.
Current columns are: {obj.df.columns}."""
        )
