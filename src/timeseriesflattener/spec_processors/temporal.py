from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Callable, Union

import polars as pl
import polars.selectors as cs
from iterpy.iter import Iter

from .._intermediary_frames import ProcessedFrame, TimeDeltaFrame, TimeMaskedFrame
from ..feature_specs.outcome import BooleanOutcomeSpec, OutcomeSpec
from ..feature_specs.predictor import PredictorSpec
from ..frame_utilities._horisontally_concat import horizontally_concatenate_dfs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..aggregators import Aggregator
    from ..feature_specs.meta import LookPeriod, ValueFrame, ValueType
    from ..feature_specs.prediction_times import PredictionTimeFrame


def _get_timedelta_frame(
    predictiontime_frame: PredictionTimeFrame, value_frame: ValueFrame
) -> TimeDeltaFrame:
    # Join the prediction time dataframe
    if predictiontime_frame.timestamp_col_name == value_frame.value_timestamp_col_name:
        """If the timestamp col names are the same, they cause conflicts when joining."""
        predictiontime_timestamp_col_name = f"_{predictiontime_frame.timestamp_col_name}"
        join_patient_times = predictiontime_frame.df.rename(
            {predictiontime_frame.timestamp_col_name: predictiontime_timestamp_col_name}
        )
    else:
        predictiontime_timestamp_col_name = predictiontime_frame.timestamp_col_name
        join_patient_times = predictiontime_frame.df

    joined_frame = join_patient_times.join(
        value_frame.df, on=predictiontime_frame.entity_id_col_name, how="left"
    )

    # Get timedelta
    timedelta_frame = joined_frame.with_columns(
        (
            pl.col(value_frame.value_timestamp_col_name) - pl.col(predictiontime_timestamp_col_name)
        ).alias("time_from_prediction_to_value")
    )

    return TimeDeltaFrame(
        timedelta_frame,
        value_col_names=value_frame.value_col_names,
        pred_time_uuid_col_name=predictiontime_frame.pred_time_uuid_col_name,
        value_timestamp_col_name=value_frame.value_timestamp_col_name,
    )


def _null_values_outside_lookwindow(
    df: pl.LazyFrame, lookwindow_predicate: pl.Expr, cols_to_null: Sequence[str]
) -> pl.LazyFrame:
    for col_to_null in cols_to_null:
        df = df.with_columns(
            pl.when(lookwindow_predicate).then(pl.col(col_to_null)).otherwise(None)
        )
    return df


def _mask_outside_lookperiod(
    timedelta_frame: TimeDeltaFrame,
    lookperiod: LookPeriod,
    column_prefix: str,
    value_col_names: Sequence[str],
) -> TimeMaskedFrame:
    timedelta_col = pl.col(timedelta_frame.timedelta_col_name)

    after_lookperiod_start = lookperiod.first <= timedelta_col
    before_lookperiod_end = timedelta_col <= lookperiod.last
    within_lookwindow = after_lookperiod_start.and_(before_lookperiod_end)

    masked_frame = _null_values_outside_lookwindow(
        df=timedelta_frame.df,
        lookwindow_predicate=within_lookwindow,
        cols_to_null=[*timedelta_frame.value_col_names, timedelta_frame.timedelta_col_name],
    )

    is_lookbehind = lookperiod.first < dt.timedelta(0)
    # The predictor case
    if is_lookbehind:
        lookperiod_string = f"{abs(lookperiod.last.days)}_to_{abs(lookperiod.first.days)}_days"
    # The outcome case
    else:
        lookperiod_string = f"{lookperiod.first.days}_to_{lookperiod.last.days}_days"

    # TODO: #436 base suffix on the type of timedelta (days, hours, minutes)
    new_colnames = [
        f"{column_prefix}_{value_col_name}_within_{lookperiod_string}"
        for value_col_name in value_col_names
    ]

    return TimeMaskedFrame(
        init_df=masked_frame.rename(dict(zip(value_col_names, new_colnames))),
        pred_time_uuid_col_name=timedelta_frame.pred_time_uuid_col_name,
        value_col_names=new_colnames,
        timestamp_col_name=timedelta_frame.value_timestamp_col_name,
    )


def _aggregate_masked_frame(
    masked_frame: TimeMaskedFrame, aggregators: Sequence[Aggregator], fallback: ValueType
) -> pl.LazyFrame:
    aggregator_expressions = [
        aggregator(value_col_name)
        for aggregator in aggregators
        for value_col_name in masked_frame.value_col_names
    ]

    grouped_frame = masked_frame.init_df.group_by(
        masked_frame.pred_time_uuid_col_name, maintain_order=True
    ).agg(aggregator_expressions)

    value_columns = (
        Iter(grouped_frame.columns)
        .filter(
            lambda col: any(
                value_col_name in col for value_col_name in masked_frame.value_col_names
            )
        )
        .map(lambda old_name: (old_name, f"{old_name}_fallback_{fallback}"))
    )
    rename_mapping = dict(value_columns)

    with_fallback = grouped_frame.with_columns(
        cs.contains(masked_frame.value_col_names).fill_null(fallback)
    ).rename(rename_mapping)

    return with_fallback


TimeMasker = Callable[[TimeDeltaFrame], TimeMaskedFrame]
MaskedAggregator = Callable[[TimeMaskedFrame], pl.LazyFrame]


def _slice_and_aggregate_spec(
    timedelta_frame: TimeDeltaFrame, masked_aggregator: MaskedAggregator, time_masker: TimeMasker
) -> pl.LazyFrame:
    sliced_frame = time_masker(timedelta_frame)
    return masked_aggregator(sliced_frame)


TemporalSpec = Union[PredictorSpec, OutcomeSpec, BooleanOutcomeSpec]


def process_temporal_spec(
    spec: TemporalSpec, predictiontime_frame: PredictionTimeFrame
) -> ProcessedFrame:
    aggregated_value_frames = (
        Iter(spec.normalised_lookperiod)
        .map(
            lambda lookperiod: _slice_and_aggregate_spec(
                timedelta_frame=_get_timedelta_frame(
                    predictiontime_frame=predictiontime_frame, value_frame=spec.value_frame
                ),
                masked_aggregator=lambda sliced_frame: _aggregate_masked_frame(
                    aggregators=spec.aggregators, fallback=spec.fallback, masked_frame=sliced_frame
                ),
                time_masker=lambda timedelta_frame: _mask_outside_lookperiod(
                    timedelta_frame=timedelta_frame,
                    lookperiod=lookperiod,
                    column_prefix=spec.column_prefix,
                    value_col_names=spec.value_frame.value_col_names,
                ),
            )
        )
        .flatten()
    )

    return ProcessedFrame(
        df=horizontally_concatenate_dfs(
            dfs=aggregated_value_frames.to_list(),
            pred_time_uuid_col_name=predictiontime_frame.pred_time_uuid_col_name,
        ),
        pred_time_uuid_col_name=predictiontime_frame.pred_time_uuid_col_name,
    )
