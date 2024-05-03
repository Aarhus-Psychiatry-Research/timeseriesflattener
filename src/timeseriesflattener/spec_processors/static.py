from __future__ import annotations

from typing import TYPE_CHECKING

from .._intermediary_frames import ProcessedFrame

if TYPE_CHECKING:
    from ..feature_specs.prediction_times import PredictionTimeFrame
    from ..feature_specs.static import StaticSpec


def process_static_spec(
    spec: StaticSpec, predictiontime_frame: PredictionTimeFrame
) -> ProcessedFrame:
    old2new_colname = {
        value_col_name: f"{spec.column_prefix}_{value_col_name}_fallback_{spec.fallback}"
        for value_col_name in spec.value_frame.value_col_names
    }
    prediction_times_with_time_from_event = (
        predictiontime_frame.df.join(
            spec.value_frame.df, on=predictiontime_frame.entity_id_col_name, how="left"
        )
        .rename(old2new_colname)
        .select(predictiontime_frame.prediction_time_uuid_col_name, *old2new_colname.values())
    )

    return ProcessedFrame(
        df=prediction_times_with_time_from_event,
        prediction_time_uuid_col_name=predictiontime_frame.prediction_time_uuid_col_name,
    )
