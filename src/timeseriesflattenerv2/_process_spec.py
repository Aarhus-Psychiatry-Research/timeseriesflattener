import datetime as dt
from typing import Sequence

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

    return TimedeltaFrame(timedelta_frame, value_col_name=value_frame.value_col_name)


def _null_values_outside_lookwindow(
    df: pl.LazyFrame, lookwindow_predicate: pl.Expr, cols_to_null: Sequence[str]
) -> pl.LazyFrame:
    for col_to_null in cols_to_null:
        df = df.with_columns(
            pl.when(lookwindow_predicate).then(pl.col(col_to_null)).otherwise(None)
        )
    return df


def _slice_frame(
    timedelta_frame: TimedeltaFrame, lookperiod: LookPeriod, column_prefix: str, value_col_name: str
) -> TimeMaskedFrame:
    # TODO: #436 base suffix on the type of timedelta (days, hours, minutes)

    timedelta_col = pl.col(timedelta_frame.timedelta_col_name)

    is_lookbehind = lookperiod.first < dt.timedelta(0)
    new_colname_prefix = f"{column_prefix}_{value_col_name}_within_"

    # The predictor case
    if is_lookbehind:
        new_colname = (
            new_colname_prefix + f"{abs(lookperiod.last.days)}_to_{abs(lookperiod.first.days)}_days"
        )
        after_lookbehind_start = lookperiod.first <= timedelta_col
        before_prediction_time = timedelta_col <= lookperiod.last

        within_lookbehind = after_lookbehind_start.and_(before_prediction_time)
        sliced_frame = _null_values_outside_lookwindow(
            df=timedelta_frame.df,
            lookwindow_predicate=within_lookbehind,
            cols_to_null=[timedelta_frame.value_col_name, timedelta_frame.timedelta_col_name],
        )
    # The outcome case
    else:
        new_colname = new_colname_prefix + f"{lookperiod.first.days}_to_{lookperiod.last.days}_days"
        after_prediction_time = lookperiod.first <= timedelta_col
        before_lookahead_end = timedelta_col <= lookperiod.last

        within_lookahead = after_prediction_time.and_(before_lookahead_end)
        sliced_frame = _null_values_outside_lookwindow(
            df=timedelta_frame.df,
            lookwindow_predicate=within_lookahead,
            cols_to_null=[timedelta_frame.value_col_name, timedelta_frame.timedelta_col_name],
        )

    return TimeMaskedFrame(
        init_df=sliced_frame.rename({timedelta_frame.value_col_name: new_colname}),
        pred_time_uuid_col_name=timedelta_frame.pred_time_uuid_col_name,
        value_col_name=new_colname,
    )


def _aggregate_within_slice(
    sliced_frame: TimeMaskedFrame, aggregators: Sequence[Aggregator], fallback: ValueType
) -> pl.LazyFrame:
    aggregator_expressions = [aggregator(sliced_frame.value_col_name) for aggregator in aggregators]

    grouped_frame = sliced_frame.init_df.group_by(
        sliced_frame.pred_time_uuid_col_name, maintain_order=True
    ).agg(aggregator_expressions)

    value_columns = (
        Iter(grouped_frame.columns)
        .filter(lambda col: sliced_frame.value_col_name in col)
        .map(lambda old_name: (old_name, f"{old_name}_fallback_{fallback}"))
    )
    rename_mapping = dict(value_columns)

    with_fallback = grouped_frame.with_columns(
        cs.contains(sliced_frame.value_col_name).fill_null(fallback)
    ).rename(rename_mapping)

    return with_fallback


def _slice_and_aggregate_spec(
    timedelta_frame: TimedeltaFrame,
    lookperiod: LookPeriod,
    aggregators: Sequence[Aggregator],
    fallback: ValueType,
    column_prefix: str,
) -> pl.LazyFrame:
    sliced_frame = _slice_frame(
        timedelta_frame, lookperiod, column_prefix, timedelta_frame.value_col_name
    )
    return _aggregate_within_slice(sliced_frame, aggregators, fallback=fallback)


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
                lookperiod=lookperiod,
                aggregators=spec.aggregators,
                fallback=spec.fallback,
                column_prefix=spec.column_prefix,
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
