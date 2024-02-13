import datetime as dt
from typing import Sequence

import polars as pl
from iterpy.iter import Iter

from ._horisontally_concat import horizontally_concatenate_dfs
from .feature_specs import (
    AggregatedValueFrame,
    Aggregator,
    LookDistance,
    OutcomeSpec,
    PredictionTimeFrame,
    PredictorSpec,
    SlicedFrame,
    TimedeltaFrame,
    ValueFrame,
    ValueSpecification,
    ValueType,
)


def process_spec(predictiontime_frame: PredictionTimeFrame, spec: ValueSpecification) -> ValueFrame:
    aggregated_value_frames = (
        Iter(_normalise_lookdistances(spec))
        .map(
            lambda distance: _slice_and_aggregate_spec(
                timedelta_frame=_get_timedelta_frame(
                    predictiontime_frame=predictiontime_frame, value_frame=spec.value_frame
                ),
                distance=distance,
                aggregators=spec.aggregators,
                fallback=spec.fallback,
                column_prefix=spec.column_prefix,
            )
        )
        .flatten()
    )

    return ValueFrame(
        init_df=horizontally_concatenate_dfs(
            [AggValueFrame.df for AggValueFrame in aggregated_value_frames.to_list()],
            pred_time_uuid_col_name=predictiontime_frame.pred_time_uuid_col_name,
        ),
        entity_id_col_name=spec.value_frame.entity_id_col_name,
        value_timestamp_col_name=spec.value_frame.value_timestamp_col_name,
        value_col_name=spec.value_frame.value_col_name,
    )


def _get_timedelta_frame(
    predictiontime_frame: PredictionTimeFrame, value_frame: ValueFrame
) -> TimedeltaFrame:
    # Join the prediction time dataframe
    joined_frame = predictiontime_frame.df.join(
        value_frame.df, on=predictiontime_frame.entity_id_col_name, how="left"
    )

    # Get timedelta
    timedelta_frame = joined_frame.with_columns(
        (
            pl.col(value_frame.value_timestamp_col_name)
            - pl.col(predictiontime_frame.timestamp_col_name)
        ).alias("time_from_prediction_to_value")
    )

    return TimedeltaFrame(timedelta_frame, value_col_name=value_frame.value_col_name)


def _normalise_lookdistances(spec: ValueSpecification) -> Sequence[LookDistance]:
    if isinstance(spec, PredictorSpec):
        lookdistances = [-distance for distance in spec.lookbehind_distances]
    elif isinstance(spec, OutcomeSpec):
        lookdistances = spec.lookahead_distances
    else:
        raise ValueError("Unknown spec type")
    return lookdistances


def _slice_frame(
    timedelta_frame: TimedeltaFrame,
    lookdistance: LookDistance,
    column_prefix: str,
    value_col_name: str,
) -> SlicedFrame:
    new_colname = f"{column_prefix}_{value_col_name}_within_{abs(lookdistance.days)}_days"

    timedelta_col = pl.col(timedelta_frame.timedelta_col_name)

    lookbehind = lookdistance < dt.timedelta(0)
    no_predictor_value = timedelta_col.is_null()

    # The predictor case
    if lookbehind:
        after_lookbehind_start = lookdistance <= timedelta_col
        before_pred_time = timedelta_col <= dt.timedelta(0)
        sliced_frame = timedelta_frame.df.filter(
            (after_lookbehind_start).and_(before_pred_time).or_(no_predictor_value)
        )
    # The outcome case
    else:
        after_pred_time = dt.timedelta(0) <= timedelta_col
        before_lookahead_end = timedelta_col <= lookdistance
        sliced_frame = timedelta_frame.df.filter(
            (after_pred_time).and_(before_lookahead_end).or_(no_predictor_value)
        )

    return SlicedFrame(
        init_df=sliced_frame.rename({timedelta_frame.value_col_name: new_colname}),
        pred_time_uuid_col_name=timedelta_frame.pred_time_uuid_col_name,
        value_col_name=new_colname,
    )


def _aggregate_within_slice(
    sliced_frame: SlicedFrame, aggregators: Sequence[Aggregator], fallback: ValueType
) -> Sequence[AggregatedValueFrame]:
    grouped_frame = sliced_frame.init_df.groupby(
        sliced_frame.pred_time_uuid_col_name, maintain_order=True
    )

    aggregated_value_frames = [
        agg.apply(grouped_frame, column_name=sliced_frame.value_col_name) for agg in aggregators
    ]

    with_fallback = [frame.fill_nulls(fallback=fallback) for frame in aggregated_value_frames]

    return with_fallback


def _slice_and_aggregate_spec(
    timedelta_frame: TimedeltaFrame,
    distance: LookDistance,
    aggregators: Sequence[Aggregator],
    fallback: ValueType,
    column_prefix: str,
) -> Sequence[AggregatedValueFrame]:
    sliced_frame = _slice_frame(
        timedelta_frame, distance, column_prefix, timedelta_frame.value_col_name
    )
    return _aggregate_within_slice(sliced_frame, aggregators, fallback=fallback)
