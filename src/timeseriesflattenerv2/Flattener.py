import datetime as dt
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


@dataclass
class Flattener:
    predictiontime_frame: PredictionTimeFrame

    def _aggregate_within_slice(
        self,
        timedelta_frame: TimedeltaFrame,
        distance: LookDistance,
        aggregators: Sequence[Aggregator],
    ) -> Iter[AggregatedValueFrame]:
        filtered_by_distance = timedelta_frame.df.filter(
            pl.col(timedelta_frame.timedelta_col_name) <= distance
        )

        aggregated_value_frames: Sequence[AggregatedValueFrame] = [
            aggregator.apply(
                SlicedFrame(filtered_by_distance),
                column_name=timedelta_frame.value_col_name,
            )
            for aggregator in aggregators
        ]

        return Iter(
            AggregatedValueFrame(
                df=frame.df,
                pred_time_uuid_col_name=timedelta_frame.pred_time_uuid_col_name,
                value_col_name=timedelta_frame.value_col_name,
            )
            for frame in aggregated_value_frames
        )

    def _slice_and_aggregate_spec(self, spec: ValueSpecification) -> ValueFrame:
        if isinstance(spec, PredictorSpec):
            lookdistances = [-distance for distance in spec.lookbehind_distances]
        elif isinstance(spec, OutcomeSpec):
            lookdistances = spec.lookahead_distances
        else:
            raise ValueError("Unknown spec type")

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

        aggregated_value_frames = (
            Iter(lookdistances)
            .map(
                lambda distance: self._aggregate_within_slice(
                    TimedeltaFrame(df=timedelta_frame),
                    distance=distance,
                    aggregators=spec.aggregators,
                )
            )
            .flatten()
        )

        return ValueFrame(
            df=pl.concat(
                [f.df for f in aggregated_value_frames.to_list()], how="horizontal"
            ),
            value_type=spec.value_frame.value_type,
            entity_id_col_name=spec.value_frame.entity_id_col_name,
            value_timestamp_col_name=spec.value_frame.value_timestamp_col_name,
        )

    def aggregate_timeseries(
        self, specs: Sequence[ValueSpecification]
    ) -> AggregatedValueFrame:
        dfs = (
            Iter(specs)
            .map(self._slice_and_aggregate_spec)
            .map(lambda x: x.df)
            .to_list()
        )
        return AggregatedValueFrame(df=pl.concat(dfs, how="horizontal"))
