import pytest
from pandas import DataFrame

from timeseriesflattener.df_transforms import df_with_multiple_values_to_named_dataframes
from timeseriesflattener.testing.utils_for_testing import str_to_df


@pytest.fixture()
def df_with_multiple_values() -> DataFrame:
    df_str = """entity_id,timestamp,value_1,value_2,
                1,2021-12-30 00:00:01, 1, 2
                1,2021-12-29 00:00:02, 2, 3"""
    return str_to_df(df_str)


def test_df_with_multiple_values_to_named_dataframes(df_with_multiple_values: DataFrame) -> None:
    dfs = df_with_multiple_values_to_named_dataframes(
        df=df_with_multiple_values,
        entity_id_col_name="entity_id",
        timestamp_col_name="timestamp",
        name_prefix="test_",
    )

    assert len(dfs) == 2
    assert dfs[0].df.shape == (2, 3)
    assert dfs[0].name == "test_value_1"
    assert dfs[1].name == "test_value_2"

    assert dfs[0].df.equals(
        str_to_df(
            """entity_id,timestamp,value,
                    1,2021-12-30 00:00:01, 1
                    1,2021-12-29 00:00:02, 2"""
        )
    )
    assert dfs[1].df.equals(
        str_to_df(
            """entity_id,timestamp,value,
                    1,2021-12-30 00:00:01, 2
                    1,2021-12-29 00:00:02, 3"""
        )
    )
