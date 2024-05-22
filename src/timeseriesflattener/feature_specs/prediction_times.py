from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from .._frame_validator import _validate_col_name_columns_exist
from ..frame_utilities.anyframe_to_lazyframe import _anyframe_to_lazyframe

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class PredictionTimeFrame:
    """Specification for prediction times, i.e. the times for which predictions are made.

    init_df must be a dataframe (pandas or polars) containing columns:
        entity_id_col_name: The name of the column containing the entity ids.
        timestamp_col_name: The name of the column containing the timestamps for when to make a prediction.
    """

    init_df: InitVar[pl.LazyFrame | pl.DataFrame | pd.DataFrame]
    entity_id_col_name: str = "entity_id"
    timestamp_col_name: str = "pred_timestamp"
    prediction_time_uuid_col_name: str = "prediction_time_uuid"
    coerce_to_lazy: InitVar[bool] = True

    def __post_init__(
        self, init_df: pl.LazyFrame | pl.DataFrame | pd.DataFrame, coerce_to_lazy: bool
    ):
        if coerce_to_lazy:
            self.df = _anyframe_to_lazyframe(init_df)
        else:
            self.df: pl.LazyFrame = init_df

        self.df = self.df.with_columns(
            pl.concat_str(
                pl.col(self.entity_id_col_name), pl.lit("-"), pl.col(self.timestamp_col_name)
            ).alias(self.prediction_time_uuid_col_name)
        )

        _validate_col_name_columns_exist(obj=self)

    def collect(self) -> pl.DataFrame:
        if isinstance(self.df, pl.DataFrame):
            return self.df
        return self.df.collect()

    def required_columns(self) -> Sequence[str]:
        return [self.entity_id_col_name]
