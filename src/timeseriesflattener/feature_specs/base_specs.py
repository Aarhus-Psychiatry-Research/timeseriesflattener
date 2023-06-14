"""Templates for feature specifications."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import pandas as pd
from pydantic import Field

log = logging.getLogger(__name__)


class AnySpec(ABC):
    class Doc:
        short_description = "A base class for all feature specifications."

    base_values_df: Optional[pd.DataFrame] = Field(  # type: ignore
        default=None,
        description="Dataframe with the values.",
    )

    feature_base_name: str = Field(
        description="""The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_baase_name>_<metadata>.""",
    )

    prefix: str = Field(
        description="""The prefix used for column name generation, e.g.
            <prefix>_<feature_name>_<metadata>.""",
    )

    @abstractmethod
    def get_output_col_name(self) -> str:
        ...

    def __eq__(self, other: Any) -> bool:
        """Add equality check for dataframes.

        Trying to run `spec in list_of_specs` works for all attributes except for df, since the truth value of a dataframe is ambiguous.
        To remedy this, we use pandas' .equals() method for comparing the dfs, and get the combined truth value.
        """
        try:
            other_attributes_equal = all(
                getattr(self, attr) == getattr(other, attr)
                for attr in self.__dict__
                if attr != "base_values_df"
            )
        except AttributeError:
            return False

        dfs_equal = self.base_values_df.equals(other.values_df)  # type: ignore

        return other_attributes_equal and dfs_equal


class StaticSpec(AnySpec):
    class Doc:
        short_description = """Specification for a static feature."""

    def get_output_col_name(self) -> str:
        return f"{self.prefix}_{self.feature_base_name}"


class TemporalSpec(AnySpec, ABC):
    """Base class for temporal features."""

    class Doc:
        short_description = """The minimum specification required for collapsing a temporal
        feature, whether looking ahead or behind. Mostly used for inheritance below."""

    interval_days: Optional[float] = Field(
        description="""How far to look in the given direction (ahead for outcomes,
            behind for predictors)""",
    )

    aggregation_fn: Callable = Field(
        description="""A function used for aggregating multiple values within the
            interval_days.""",
    )

    fallback: Union[float, str] = Field(
        description="""Which value to use if no values are found within interval_days.""",
    )

    def get_output_col_name(self) -> str:
        """Generate the column name for the output column.
        If interval days is a float, the decimal point is changed to an underscore.

        Args:
            additional_feature_name (Optional[str]): additional feature name to
                append to the column name.
        """
        col_str = f"{self.prefix}_{self.feature_base_name}_within_{str(self.interval_days)}_days_{self.aggregation_fn.__name__}_fallback_{self.fallback}"
        return col_str
