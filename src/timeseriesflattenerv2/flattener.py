from dataclasses import dataclass
from typing import Sequence

import polars as pl
from iterpy._iter import Iter

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


def _aggregate_within_slice(
    sliced_frame: SlicedFrame, aggregators: Sequence[Aggregator], fallback: ValueType
) -> Sequence[AggregatedValueFrame]:
    aggregated_value_frames = [
        agg.apply(SlicedFrame(sliced_frame.df), column_name=sliced_frame.value_col_name)
        for agg in aggregators
    ]

    with_fallback = [frame.fill_nulls(fallback=fallback) for frame in aggregated_value_frames]

    return [
        AggregatedValueFrame(
            df=frame.df,
            pred_time_uuid_col_name=sliced_frame.pred_time_uuid_col_name,
            value_col_name=sliced_frame.value_col_name,
        )
        for frame in with_fallback
    ]


def _slice_frame(timedelta_frame: TimedeltaFrame, distance: LookDistance) -> SlicedFrame:
    sliced_frame = timedelta_frame.df.filter(pl.col(timedelta_frame.timedelta_col_name) <= distance)

    return SlicedFrame(
        df=sliced_frame,
        pred_time_uuid_col_name=timedelta_frame.pred_time_uuid_col_name,
        value_col_name=timedelta_frame.value_col_name,
    )


def _slice_and_aggregate_spec(
    timedelta_frame: TimedeltaFrame,
    distance: LookDistance,
    aggregators: Sequence[Aggregator],
    fallback: ValueType,
) -> Sequence[AggregatedValueFrame]:
    sliced_frame = _slice_frame(timedelta_frame, distance)
    return _aggregate_within_slice(sliced_frame, aggregators, fallback=fallback)


def _normalise_lookdistances(spec: ValueSpecification) -> Sequence[LookDistance]:
    if isinstance(spec, PredictorSpec):
        lookdistances = [-distance for distance in spec.lookbehind_distances]
    elif isinstance(spec, OutcomeSpec):
        lookdistances = spec.lookahead_distances
    else:
        raise ValueError("Unknown spec type")
    return lookdistances


def _horizontally_concatenate_dfs(dfs: Sequence[pl.LazyFrame]) -> pl.LazyFrame:
    # Run some checks on the dfs
    return pl.concat(dfs, how="horizontal")


def _get_timedelta_frame(
    predictiontime_frame: PredictionTimeFrame, value_frame: ValueFrame
) -> TimedeltaFrame:
    # Join the prediction time dataframe
    joined_frame = predictiontime_frame.to_lazyframe_with_uuid().join(
        value_frame.df, on=predictiontime_frame.entity_id_col_name
    )

    # Get timedelta
    timedelta_frame = joined_frame.with_columns(
        (
            pl.col(value_frame.value_timestamp_col_name)
            - pl.col(predictiontime_frame.timestamp_col_name)
        ).alias("time_from_prediction_to_value")
    )

    return TimedeltaFrame(timedelta_frame)


def _process_spec(
    predictiontime_frame: PredictionTimeFrame, spec: ValueSpecification
) -> ValueFrame:
    lookdistances = _normalise_lookdistances(spec)
    timedelta_frame = _get_timedelta_frame(
        predictiontime_frame=predictiontime_frame, value_frame=spec.value_frame
    )

    aggregated_value_frames = (
        Iter(lookdistances)
        .map(
            lambda distance: _slice_and_aggregate_spec(
                timedelta_frame=timedelta_frame,
                distance=distance,
                aggregators=spec.aggregators,
                fallback=spec.fallback,
            )
        )
        .flatten()
    )

    return ValueFrame(
        df=_horizontally_concatenate_dfs([f.df for f in aggregated_value_frames.to_list()]),
        value_type=spec.value_frame.value_type,
        entity_id_col_name=spec.value_frame.entity_id_col_name,
        value_timestamp_col_name=spec.value_frame.value_timestamp_col_name,
    )


@dataclass
class Flattener:
    predictiontime_frame: PredictionTimeFrame

    def aggregate_timeseries(self, specs: Sequence[ValueSpecification]) -> AggregatedValueFrame:
        dfs = (
            Iter(specs)
            .map(
                lambda spec: _process_spec(
                    predictiontime_frame=self.predictiontime_frame, spec=spec
                )
            )
            .map(lambda x: x.df)
            .to_list()
        )
        return AggregatedValueFrame(df=_horizontally_concatenate_dfs(dfs))