from timeseriesflattener import PredictionTimeFrame, ValueSpecification
from ..intermediary import 
import datetime as dt

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
