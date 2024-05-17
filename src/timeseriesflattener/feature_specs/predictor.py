from __future__ import annotations

import datetime as dt
from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING

from .._frame_validator import _validate_col_name_columns_exist
from .meta import ValueFrame, _lookdistance_to_normalised_lookperiod

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from ..aggregators import Aggregator


@dataclass
class PredictorSpec:
    """Specification for a predictor.

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
            _lookdistance_to_normalised_lookperiod(lookdistance=lookdistance, direction="behind")
            for lookdistance in lookbehind_distances
        ]
        _validate_col_name_columns_exist(obj=self)

    @property
    def df(self) -> pl.LazyFrame:
        return self.value_frame.df
