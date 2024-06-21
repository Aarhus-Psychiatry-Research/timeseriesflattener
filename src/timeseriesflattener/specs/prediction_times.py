from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from ..validators import validate_col_name_columns_exist
from ..utils import anyframe_to_pl_frame

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class PredictionTimeFrame:
    """Specification for prediction times, i.e. the times for which predictions are made.

    init_df must be a dataframe (pandas or polars) containing columns:
        entity_id_col_name: The name of the column containing the entity ids.
        timestamp_col_name: The name of the column containing the timestamps for when to make a prediction.
    """

    init_df: InitVar[pl.DataFrame | pd.DataFrame]
    entity_id_col_name: str = "entity_id"
    timestamp_col_name: str = "pred_timestamp"
    prediction_time_uuid_col_name: str = "prediction_time_uuid"

    def __post_init__(self, init_df: pl.DataFrame | pd.DataFrame):
        # Sort to ensure alignment when processing multiple specs and concatenating in the end.
        self.df = anyframe_to_pl_frame(init_df).sort(self.timestamp_col_name)

        self.df = self.df.with_columns(
            pl.concat_str(
                pl.col(self.entity_id_col_name), pl.lit("-"), pl.col(self.timestamp_col_name)
            ).alias(self.prediction_time_uuid_col_name)
        )

        validate_col_name_columns_exist(obj=self)

    def collect(self) -> pl.DataFrame:
        return self.df

    def required_columns(self) -> Sequence[str]:
        return [self.entity_id_col_name]
