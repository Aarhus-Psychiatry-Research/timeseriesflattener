from __future__ import annotations

from typing import TYPE_CHECKING

from .._intermediary_frames import ProcessedFrame

if TYPE_CHECKING:
    from ..feature_specs.prediction_times import PredictionTimeFrame
    from ..feature_specs.static import StaticSpec


def process_static_spec(
    spec: StaticSpec, predictiontime_frame: PredictionTimeFrame
) -> ProcessedFrame:
    new_col_name = (
        f"{spec.column_prefix}_{spec.value_frame.value_col_name}_fallback_{spec.fallback}"
    )
    prediction_times_with_time_from_event = (
        predictiontime_frame.df.join(
            spec.value_frame.df, on=predictiontime_frame.entity_id_col_name, how="left"
        )
        .rename({spec.value_frame.value_col_name: new_col_name})
        .select(predictiontime_frame.pred_time_uuid_col_name, new_col_name)
    )

    return ProcessedFrame(
        df=prediction_times_with_time_from_event,
        pred_time_uuid_col_name=predictiontime_frame.pred_time_uuid_col_name,
    )
