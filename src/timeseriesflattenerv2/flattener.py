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
)


def _aggregate_within_slice(
    sliced_frame: SlicedFrame, aggregators: Sequence[Aggregator]
) -> Iter[AggregatedValueFrame]:
    aggregated_value_frames = [
        aggregator.apply(SlicedFrame(sliced_frame.df), column_name=sliced_frame.value_col_name)
        for aggregator in aggregators
    ]

    return Iter(
        AggregatedValueFrame(
            df=frame.df,
            pred_time_uuid_col_name=sliced_frame.pred_time_uuid_col_name,
            value_col_name=sliced_frame.value_col_name,
        )
        for frame in aggregated_value_frames
    )


def _slice_frame(timedelta_frame: TimedeltaFrame, distance: LookDistance) -> SlicedFrame:
    sliced_frame = timedelta_frame.df.filter(pl.col(timedelta_frame.timedelta_col_name) <= distance)

    return SlicedFrame(
        df=sliced_frame,
        pred_time_uuid_col_name=timedelta_frame.pred_time_uuid_col_name,
        value_col_name=timedelta_frame.value_col_name,
    )


def _slice_and_aggregate_spec(
    timedelta_frame: TimedeltaFrame, distance: LookDistance, aggregators: Sequence[Aggregator]
) -> Iter[AggregatedValueFrame]:
    sliced_frame = _slice_frame(timedelta_frame, distance)
    return _aggregate_within_slice(sliced_frame, aggregators)


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


@dataclass
class Flattener:
    predictiontime_frame: PredictionTimeFrame

    def _get_timedelta_frame(self, spec: ValueSpecification) -> TimedeltaFrame:
        # Join the prediction time dataframe
        joined_frame = self.predictiontime_frame.to_lazyframe_with_uuid().join(
            spec.value_frame.df, on=self.predictiontime_frame.entity_id_col_name
        )

        # Get timedelta
        timedelta_frame = joined_frame.with_columns(
            (
                pl.col(spec.value_frame.value_timestamp_col_name)
                - pl.col(self.predictiontime_frame.timestamp_col_name)
            ).alias("time_from_prediction_to_value")
        )

        return TimedeltaFrame(timedelta_frame)

    def _process_spec(self, spec: ValueSpecification) -> ValueFrame:
        lookdistances = _normalise_lookdistances(spec)
        timedelta_frame = self._get_timedelta_frame(spec)

        aggregated_value_frames = (
            Iter(lookdistances)
            .map(
                lambda distance: _slice_and_aggregate_spec(
                    timedelta_frame=timedelta_frame, distance=distance, aggregators=spec.aggregators
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

    def aggregate_timeseries(self, specs: Sequence[ValueSpecification]) -> AggregatedValueFrame:
        dfs = Iter(specs).map(self._process_spec).map(lambda x: x.df).to_list()
        return AggregatedValueFrame(df=_horizontally_concatenate_dfs(dfs))
