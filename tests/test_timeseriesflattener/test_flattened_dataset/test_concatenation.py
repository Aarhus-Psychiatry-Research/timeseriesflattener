"""Testing of rapid concatenation"""
import random
import time
import uuid
from typing import Any, Callable, List

import pandas as pd
import pytest
from pandas import DataFrame
from timeseriesflattener.flattened_dataset import TimeseriesFlattener


def benchmark(func: Callable, *args: Any, **kwargs: Any) -> float:
    """Benchmark a function."""
    start = time.perf_counter()
    func(*args, **kwargs)
    end = time.perf_counter()
    print(f"{func.__name__} took {end - start:.4f} seconds to execute.")
    return end - start


def generate_test_df(uuids: List[str], col_values: List[int]) -> DataFrame:
    """Generate a test df with a random integer column and a uuid index."""
    df = pd.DataFrame()

    # create a column with random integers between 1 and 1000_000
    df["random_int"] = col_values

    # Set index to uuids
    df.index = uuids  # type: ignore

    # Sort df by indices
    print("Generated df")

    return df


def test_benchmark_full_index_comparison_before_concatenate():
    """Benchmark the full index comparison before concatenating."""
    # create an empty dataframe

    n_rows = 20_000

    # Generate 16 digit alphanumeric uuids
    uuids = [uuid.uuid4().hex[:16] for _ in range(n_rows)]
    col_values = [1 for _ in range(n_rows)]

    dfs = [generate_test_df(uuids=uuids, col_values=col_values) for _ in range(1_000)]

    # 0.004 seconds for 9 dfs when sampling 5_000 rows
    # 0.033 seconds for 9 dfs when sampling 100_000 rows
    # 7.622 seconds for 100 dfs when sampling 2_000_000 rows
    compute_seconds = benchmark(
        TimeseriesFlattener._check_dfs_are_ready_for_concat,
        dfs,
    )

    assert compute_seconds < 2


def test_error_raised_with_unaligend_rows():
    """Test that an error is raised when the rows are not aligned"""
    n_rows = 2_000_000
    uuids = [uuid.uuid4().hex[:16] for _ in range(n_rows)]
    random_ints = [1 for _ in range(n_rows)]

    df1 = generate_test_df(uuids=uuids, col_values=random_ints)

    # Rewrite a uuids at a random index
    random_index = random.randint(0, n_rows - 1)
    uuids[random_index] = uuid.uuid4().hex[:16]

    df2 = generate_test_df(uuids=uuids, col_values=random_ints)

    with pytest.raises(ValueError, match="not aligned"):
        TimeseriesFlattener._check_dfs_are_ready_for_concat(dfs=[df1, df2])
