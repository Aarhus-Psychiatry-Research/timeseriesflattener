"""Tests of check_raw_df."""

import pytest
from utils_for_testing import str_to_df  # noqa pylint: disable=import-error

from psycopmlutils.data_checks.raw.check_raw_df import check_raw_df

# pylint: disable=missing-function-docstring


def test_raw_df_has_rows():
    df_str = """dw_ek_borger,timestamp,value
            """

    df = str_to_df(df_str)

    with pytest.raises(ValueError, match="No rows returned"):
        check_raw_df(df)


def test_raw_df_has_required_cols():
    df_str = """dw_ek_borger,timstamp,value
            """

    df = str_to_df(df_str)

    with pytest.raises(ValueError, match="not in columns"):
        check_raw_df(df)


def test_raw_df_has_datetime_formatting():
    df_str = """dw_ek_borger,timestamp,value
                1,2021-01-01 00:00:00,1
            """

    df = str_to_df(df_str, convert_timestamp_to_datetime=False)

    with pytest.raises(ValueError, match="invalid datetime"):
        check_raw_df(df)


def test_raw_df_has_expected_val_dtype():
    df_str = """dw_ek_borger,timestamp,value
                1,2021-01-01 00:00:00,a
            """

    df = str_to_df(df_str)

    with pytest.raises(ValueError, match="value: dtype"):
        check_raw_df(df)


def test_raw_df_has_invalid_na_prop():
    """If raw df has a nan prop above 0.0, return an error."""
    df_str = """dw_ek_borger,timestamp,value
                1,2021-01-01 00:00:00,np.nan
            """

    df = str_to_df(df_str)

    with pytest.raises(ValueError, match="NaN"):
        check_raw_df(df)


def test_raw_df_has_duplicates():
    df_str = """dw_ek_borger,timestamp,value
                1,2021-01-01 00:00:00,np.nan
                1,2021-01-01 00:00:00,np.nan
            """

    df = str_to_df(df_str)

    with pytest.raises(ValueError, match="NaN"):
        check_raw_df(df)
