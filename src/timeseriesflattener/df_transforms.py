from typing import List

from pandas import DataFrame

from timeseriesflattener.feature_specs.group_specs import NamedDataframe


def df_with_multiple_values_to_named_dataframes(
    df: DataFrame,
    entity_id_col_name: str = "entity_id",
    timestamp_col_name: str = "timestamp",
    name_prefix: str = "value_",
) -> List[NamedDataframe]:
    """Split a dataframe with multiple values into a list of dataframes with one value each."""
    mandatory_columns = [entity_id_col_name, timestamp_col_name]
    if any(col not in list(df.columns) for col in mandatory_columns):
        raise ValueError(
            f"entity_id_col_name and timestamp_col_name must be columns in the dataframe. Available columns are {df.columns}."
        )

    value_cols = [col for col in list(df.columns) if col not in mandatory_columns]
    return [
        NamedDataframe(
            df=df[[*mandatory_columns, value_col_name]].rename(columns={value_col_name: "value"}),
            name=name_prefix + str(value_col_name),
        )
        for value_col_name in value_cols
    ]
