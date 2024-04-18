from __future__ import annotations

from typing import TYPE_CHECKING, Union
import datetime as dt

from .feature_specs.static import StaticSpec
from .feature_specs.timedelta import TimeDeltaSpec
from .spec_processors.static import process_static_spec
from .spec_processors.temporal import process_temporal_spec
from .spec_processors.timedelta import process_timedelta_spec

if TYPE_CHECKING:
    from ._intermediary_frames import ProcessedFrame
    from .feature_specs.prediction_times import PredictionTimeFrame
    from .flattener import ValueSpecification


def process_spec(
    spec: ValueSpecification,
    predictiontime_frame: PredictionTimeFrame,
    step_size: dt.timedelta | None = None,
) -> ProcessedFrame:
    if isinstance(spec, TimeDeltaSpec):
        return process_timedelta_spec(spec, predictiontime_frame)
    if isinstance(spec, StaticSpec):
        return process_static_spec(spec, predictiontime_frame)

    return process_temporal_spec(
        spec=spec, predictiontime_frame=predictiontime_frame, step_size=step_size
    )
