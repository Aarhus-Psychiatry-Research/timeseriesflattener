from typing import Sequence

import polars as pl
from iterpy.iter import Iter


def horizontally_concatenate_dfs(
    dfs: Sequence[pl.LazyFrame], pred_time_uuid_col_name: str
) -> pl.LazyFrame:
    dfs_without_identifiers = Iter(dfs).map(lambda df: df.drop([pred_time_uuid_col_name])).to_list()

    return pl.concat([dfs[0], *dfs_without_identifiers[1:]], how="horizontal")
