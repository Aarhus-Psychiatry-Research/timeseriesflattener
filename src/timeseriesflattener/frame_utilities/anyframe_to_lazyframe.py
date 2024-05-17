from __future__ import annotations


import pandas as pd
import polars as pl


def _anyframe_to_lazyframe(init_df: pl.LazyFrame | pl.DataFrame | pd.DataFrame) -> pl.LazyFrame:
    if isinstance(init_df, pl.LazyFrame):
        return init_df
    if isinstance(init_df, pl.DataFrame):
        return init_df.lazy()
    if isinstance(init_df, pd.DataFrame):
        return pl.from_pandas(init_df).lazy()
    raise ValueError(f"Unsupported type: {type(init_df)}.")


def _anyframe_to_eagerframe(init_df: pl.LazyFrame | pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    return _anyframe_to_lazyframe(init_df).collect()
