import datetime as dt
from typing import Callable, Sequence

import polars as pl
import polars.selectors as cs
from iterpy.iter import Iter

from ._horisontally_concat import horizontally_concatenate_dfs
from .feature_specs import (
    Aggregator,
    LookPeriod,
    PredictionTimeFrame,
    ProcessedFrame,
    TimedeltaFrame,
    TimeMaskedFrame,
    ValueFrame,
    ValueSpecification,
    ValueType,
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

    return TimedeltaFrame(
        timedelta_frame,
        value_col_name=value_frame.value_col_name,
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
    timedelta_frame: TimedeltaFrame, lookperiod: LookPeriod, column_prefix: str, value_col_name: str
) -> TimeMaskedFrame:
    timedelta_col = pl.col(timedelta_frame.timedelta_col_name)

    after_lookperiod_start = lookperiod.first <= timedelta_col
    before_lookperiod_end = timedelta_col <= lookperiod.last
    within_lookwindow = after_lookperiod_start.and_(before_lookperiod_end)

    masked_frame = _null_values_outside_lookwindow(
        df=timedelta_frame.df,
        lookwindow_predicate=within_lookwindow,
        cols_to_null=[timedelta_frame.value_col_name, timedelta_frame.timedelta_col_name],
    )

    is_lookbehind = lookperiod.first < dt.timedelta(0)
    # The predictor case
    if is_lookbehind:
        lookperiod_string = f"{abs(lookperiod.last.days)}_to_{abs(lookperiod.first.days)}_days"
    # The outcome case
    else:
        lookperiod_string = f"{lookperiod.first.days}_to_{lookperiod.last.days}_days"

    # TODO: #436 base suffix on the type of timedelta (days, hours, minutes)
    new_colname = f"{column_prefix}_{value_col_name}_within_{lookperiod_string}"

    return TimeMaskedFrame(
        init_df=masked_frame.rename({timedelta_frame.value_col_name: new_colname}),
        pred_time_uuid_col_name=timedelta_frame.pred_time_uuid_col_name,
        value_col_name=new_colname,
        timestamp_col_name=timedelta_frame.value_timestamp_col_name,
    )


def _aggregate_masked_frame(
    masked_frame: TimeMaskedFrame, aggregators: Sequence[Aggregator], fallback: ValueType
) -> pl.LazyFrame:
    aggregator_expressions = [aggregator(masked_frame.value_col_name) for aggregator in aggregators]

    grouped_frame = masked_frame.init_df.group_by(
        masked_frame.pred_time_uuid_col_name, maintain_order=True
    ).agg(aggregator_expressions)

    value_columns = (
        Iter(grouped_frame.columns)
        .filter(lambda col: masked_frame.value_col_name in col)
        .map(lambda old_name: (old_name, f"{old_name}_fallback_{fallback}"))
    )
    rename_mapping = dict(value_columns)

    with_fallback = grouped_frame.with_columns(
        cs.contains(masked_frame.value_col_name).fill_null(fallback)
    ).rename(rename_mapping)

    return with_fallback


TimeMasker = Callable[[TimedeltaFrame], TimeMaskedFrame]
MaskedAggregator = Callable[[TimeMaskedFrame], pl.LazyFrame]


def _slice_and_aggregate_spec(
    timedelta_frame: TimedeltaFrame, masked_aggregator: MaskedAggregator, time_masker: TimeMasker
) -> pl.LazyFrame:
    sliced_frame = time_masker(timedelta_frame)
    return masked_aggregator(sliced_frame)


def process_spec(
    spec: ValueSpecification, predictiontime_frame: PredictionTimeFrame
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
                    value_col_name=spec.value_frame.value_col_name,
                ),
            )
        )
        .flatten()
    )

    return ProcessedFrame(
        df=horizontally_concatenate_dfs(
            aggregated_value_frames.to_list(),
            pred_time_uuid_col_name=predictiontime_frame.pred_time_uuid_col_name,
        ),
        pred_time_uuid_col_name=predictiontime_frame.pred_time_uuid_col_name,
    )
