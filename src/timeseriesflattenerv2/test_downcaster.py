import polars as pl

from .feature_specs import _downcast_dataframe


def test_downcasting():
    input_df = pl.LazyFrame({"value_col_name": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    result = _downcast_dataframe(input_df).collect()
    assert result["value_col_name"].dtype == pl.UInt8
