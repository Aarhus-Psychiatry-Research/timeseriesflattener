"""This module contains functions for handling dataframes with multiindex columns."""
from typing import Callable, List, Optional

import pandas as pd

from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
    TemporalSpec,
)


class ColumnHandler:
    """Class for handling dataframes with multiindex columns."""

    @staticmethod
    def rename_value_column(
        df: pd.DataFrame,
        output_spec: AnySpec,
    ) -> pd.DataFrame:
        """Renames the value column to the column name specified in the output_spec.
        Handles the case where the output_spec has a multiindex.

        Args:
            output_spec (TemporalSpec): Output specification
            df (pd.DataFrame): Dataframe with value column
        """
        df = df.rename(columns={"value": output_spec.get_output_col_name()})

        return df

    @staticmethod
    def replace_na_in_spec_col_with_fallback(
        df: pd.DataFrame,
        output_spec: TemporalSpec,
    ) -> pd.DataFrame:
        """Adds fallback to value columns in df. If the value column is a multiindex,
        adds fallback to all columns in the multiindex. Otherwise, adds fallback to the
        single value column.

        Args:
            df (pd.DataFrame): Dataframe with value column
            output_spec (TemporalSpec): Output specification
        """
        if "value" in df.columns and isinstance(df["value"], pd.DataFrame):
            df["value"] = df["value"].fillna(output_spec.fallback)  # type: ignore
        else:
            df[output_spec.get_output_col_name()] = df[
                output_spec.get_output_col_name()
            ].fillna(
                output_spec.fallback,  # type: ignore
            )
        return df

    @staticmethod
    def get_value_col_str_name(
        df: Optional[pd.DataFrame] = None,
        output_spec: Optional[TemporalSpec] = None,
    ) -> List[str]:
        """Returns the name of the value column in df. If df has a multiindex,
        returns a list of all column names in the 'value' multiindex.

        Args:
            df (pd.DataFrame): Dataframe to get value column name from.
            output_spec (TemporalSpec): Output specification"""
        if df is not None or output_spec is not None:
            return (
                df["value"].columns.tolist()  # type: ignore
                if isinstance(df.columns, pd.MultiIndex)  # type: ignore
                else [output_spec.get_output_col_name()]  # type: ignore
            )
        raise ValueError("Either df or output_spec must be provided.")
