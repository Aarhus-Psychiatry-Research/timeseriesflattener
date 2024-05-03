from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from iterpy.iter import Iter

if TYPE_CHECKING:
    from collections.abc import Sequence


def horizontally_concatenate_dfs(
    dfs: Sequence[pl.LazyFrame], prediction_time_uuid_col_name: str
) -> pl.LazyFrame:
    dfs_without_identifiers = (
        Iter(dfs).map(lambda df: df.drop([prediction_time_uuid_col_name])).to_list()
    )

    return pl.concat([dfs[0], *dfs_without_identifiers[1:]], how="horizontal")
