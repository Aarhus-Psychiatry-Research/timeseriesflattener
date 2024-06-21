from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from iterpy.iter import Iter

if TYPE_CHECKING:
    from collections.abc import Sequence

from typing import TYPE_CHECKING

import polars as pl
from iterpy.iter import Iter
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


def horizontally_concatenate_dfs(
    dfs: Sequence[pl.DataFrame], prediction_time_uuid_col_name: str
) -> pl.DataFrame:
    dfs_without_identifiers = (
        Iter(dfs).map(lambda df: df.drop([prediction_time_uuid_col_name])).to_list()
    )

    return pl.concat([dfs[0], *dfs_without_identifiers[1:]], how="horizontal")


def anyframe_to_pl_frame(init_df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    if isinstance(init_df, pl.DataFrame):
        return init_df
    if isinstance(init_df, pd.DataFrame):
        return pl.from_pandas(init_df)
    raise ValueError(f"Unsupported type: {type(init_df)}.")
