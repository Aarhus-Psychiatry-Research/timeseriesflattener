from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from iterpy.iter import Iter

if TYPE_CHECKING:
    from ._intermediary_frames import AggregatedValueFrame, TimeDeltaFrame, TimeMaskedFrame
    from .feature_specs.meta import ValueFrame
    from .feature_specs.outcome import BooleanOutcomeSpec, OutcomeSpec
    from .feature_specs.prediction_times import PredictionTimeFrame
    from .feature_specs.predictor import PredictorSpec
    from .feature_specs.static import StaticFrame
    from .feature_specs.timedelta import TimeDeltaSpec
    from .feature_specs.timestamp_frame import TimestampValueFrame


@dataclass(frozen=True)
class SpecColumnError(Exception):
    description: str


def _validate_col_name_columns_exist(
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
