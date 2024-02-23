from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from ..feature_specs.meta import InitDF_T


def _anyframe_to_lazyframe(init_df: InitDF_T) -> pl.LazyFrame:
    if isinstance(init_df, pl.LazyFrame):
        return init_df
    if isinstance(init_df, pl.DataFrame):
        return init_df.lazy()
    if isinstance(init_df, pd.DataFrame):
        return pl.from_pandas(init_df).lazy()
    raise ValueError(f"Unsupported type: {type(init_df)}.")


def _anyframe_to_eagerframe(init_df: InitDF_T) -> pl.DataFrame:
    return _anyframe_to_lazyframe(init_df).collect()
