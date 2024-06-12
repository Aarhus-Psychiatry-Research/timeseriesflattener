from __future__ import annotations
from typing import Union
from timeseriesflattener.specs.outcome import BooleanOutcomeSpec, OutcomeSpec
from timeseriesflattener.specs.prediction_times import PredictionTimeFrame
import datetime as dt

from timeseriesflattener.intermediary import ProcessedFrame
from timeseriesflattener.processors.static import process_static_spec
from timeseriesflattener.processors.temporal import process_temporal_spec
from timeseriesflattener.processors.timedelta import process_timedelta_spec
from timeseriesflattener.specs.static import StaticSpec
from timeseriesflattener.specs.temporal import PredictorSpec
from timeseriesflattener.specs.timedelta import TimeDeltaSpec

ValueSpecification = Union[
    PredictorSpec, OutcomeSpec, BooleanOutcomeSpec, TimeDeltaSpec, StaticSpec
]


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
