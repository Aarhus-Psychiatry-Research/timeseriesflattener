import polars as pl

from .feature_specs import _downcast_dataframe


def test_downcasting():
    input_df = pl.LazyFrame({"int_col": [1, 2, 3], "float_col": [1.0, 2.0, 3.0]})
    result = _downcast_dataframe(input_df).collect()
    assert result["int_col"].dtype == pl.UInt8
    assert result["float_col"].dtype == pl.Float32
