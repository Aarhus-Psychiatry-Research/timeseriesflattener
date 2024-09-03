from __future__ import annotations

import datetime as dt
from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING

from timeseriesflattener.specs import _lookdistance_to_timedelta_days

from ..aggregators import (
    AggregatorName,
    strings_to_aggregators,
    validate_compatible_fallback_type_for_aggregator,
)
from ..validators import validate_col_name_columns_exist
from .value import ValueFrame, lookdistance_to_normalised_lookperiod

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from ..aggregators import Aggregator


@dataclass
class PredictorSpec:
    """Specification for a temporal predictor.

    The value_frame must contain columns:
        entity_id_col_name: The name of the column containing the entity ids.
        value_timestamp_col_name: The name of the column containing the timestamps for each value.
        additional columns containing values to aggregate. The name of the columns will be used for feature naming.
    """

    value_frame: ValueFrame
    lookbehind_distances: InitVar[Sequence[dt.timedelta | tuple[dt.timedelta, dt.timedelta]]]
    aggregators: Sequence[Aggregator]
    fallback: int | float | str | None
    column_prefix: str = "pred"

    def __post_init__(
        self, lookbehind_distances: Sequence[dt.timedelta | tuple[dt.timedelta, dt.timedelta]]
    ):
        self.normalised_lookperiod = [
            lookdistance_to_normalised_lookperiod(lookdistance=lookdistance, direction="behind")
            for lookdistance in lookbehind_distances
        ]
        validate_col_name_columns_exist(obj=self)
        for aggregator in self.aggregators:
            validate_compatible_fallback_type_for_aggregator(
                aggregator=aggregator, fallback=self.fallback
            )

    @property
    def df(self) -> pl.DataFrame:
        return self.value_frame.df

    @staticmethod
    def from_primitives(
        df: pl.DataFrame,
        entity_id_col_name: str,
        lookbehind_days: Sequence[float | tuple[float, float]],
        aggregators: Sequence[AggregatorName],
        value_timestamp_col_name: str = "timestamp",
        column_prefix: str = "pred",
        fallback: int | float | str | None = 0,
    ) -> PredictorSpec:
        """Create a PredictorSpec from primitives."""
        lookbehind_distances = [_lookdistance_to_timedelta_days(d) for d in lookbehind_days]

        return PredictorSpec(
            value_frame=ValueFrame(
                init_df=df,
                entity_id_col_name=entity_id_col_name,
                value_timestamp_col_name=value_timestamp_col_name,
            ),
            lookbehind_distances=lookbehind_distances,
            aggregators=strings_to_aggregators(aggregators, value_timestamp_col_name),
            fallback=fallback,
            column_prefix=column_prefix,
        )
