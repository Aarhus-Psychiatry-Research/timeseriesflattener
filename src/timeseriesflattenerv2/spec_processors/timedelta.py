from typing import TYPE_CHECKING

import polars as pl

from .._intermediary_frames import ProcessedFrame

if TYPE_CHECKING:
    from ..feature_specs.prediction_times import PredictionTimeFrame
    from ..feature_specs.timedelta import TimeDeltaSpec


def process_timedelta_spec(
    spec: "TimeDeltaSpec", predictiontime_frame: "PredictionTimeFrame"
) -> ProcessedFrame:
    new_col_name = f"{spec.column_prefix}_{spec.output_name}_fallback_{spec.fallback}"
    prediction_times_with_time_from_event = (
        predictiontime_frame.df.join(
            spec.df.rename({spec.init_frame.value_timestamp_col_name: "_event_time"}),
            on=predictiontime_frame.entity_id_col_name,
            how="left",
        )
        .with_columns(
            (
                (pl.col(predictiontime_frame.timestamp_col_name) - pl.col("_event_time")).dt.days()
                / 365
            )
            .fill_null(spec.fallback)
            .alias(new_col_name)
        )
        .select(predictiontime_frame.pred_time_uuid_col_name, new_col_name)
    )

    return ProcessedFrame(
        df=prediction_times_with_time_from_event,
        pred_time_uuid_col_name=predictiontime_frame.pred_time_uuid_col_name,
    )
