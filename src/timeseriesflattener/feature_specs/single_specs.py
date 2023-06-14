from dataclasses import dataclass
from typing import Callable, Optional, Union

import pandas as pd
from pydantic import Field

BASE_VALUES_DEF = Field(
    default=None,
    description="Dataframe with the values.",
)
FEATURE_BASE_NAME_DEF = Field(
    description="""The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_baase_name>_<metadata>.""",
)
PRED_PREFIX_DEF = Field(
    default="pred",
    description="""The prefix used for column name generation, e.g.
            <prefix>_<feature_name>_<metadata>.""",
)
OUTC_PREFIX_DEF = Field(
    default="outc",
    description="""The prefix used for column name generation, e.g.
            <prefix>_<feature_name>_<metadata>.""",
)
AGGREGATION_FN_DEFINITION = Field(
    description="""How to aggregate multiple values within a window. Can be a string, a function, or a list of functions.""",
)
FALLBACK_DEFINITION = Field(
    description="""Value to return if no values is found within window.""",
)
LOOKAHEAD_DAYS_DEF = Field(
    description="""How far ahead from the prediction time to look for outcome values""",
)
LOOKBEHIND_DAYS_DEF = Field(
    description="""How far behind to look for values""",
)


@dataclass(frozen=True)
class StaticSpec:
    base_values_df: pd.DataFrame = BASE_VALUES_DEF
    prefix: str = PRED_PREFIX_DEF
    feature_base_name: str = FEATURE_BASE_NAME_DEF

    class Doc:
        short_description = """Specification for a static feature."""

    def get_output_col_name(self) -> str:
        return f"{self.prefix}_{self.feature_base_name}"


@dataclass(frozen=True)
class PredictorSpec:
    """Specification for a predictor."""

    base_values_df: pd.DataFrame = BASE_VALUES_DEF
    prefix: str = PRED_PREFIX_DEF
    feature_base_name: str = FEATURE_BASE_NAME_DEF
    aggregation_fn: Callable = AGGREGATION_FN_DEFINITION
    fallback: Union[str, int] = FALLBACK_DEFINITION
    lookbehind_days: float = LOOKBEHIND_DAYS_DEF

    class Doc:
        short_description = """Specification for a single predictor."""

    def get_output_col_name(self) -> str:
        """Generate the column name for the output column.
        If interval days is a float, the decimal point is changed to an underscore.

        Args:
            additional_feature_name (Optional[str]): additional feature name to
                append to the column name.
        """
        col_str = f"{self.prefix}_{self.feature_base_name}_within_{str(self.lookbehind_days)}_days_{self.aggregation_fn.__name__}_fallback_{self.fallback}"
        return col_str


@dataclass(frozen=True)
class TextPredictorSpec:
    """Specification for a text predictor, where the df has been resolved."""

    base_values_df: pd.DataFrame = BASE_VALUES_DEF
    prefix: str = PRED_PREFIX_DEF
    feature_base_name: str = FEATURE_BASE_NAME_DEF
    aggregation_fn: Callable = AGGREGATION_FN_DEFINITION
    fallback: Union[str, int] = FALLBACK_DEFINITION
    lookbehind_days: float = Field(
        description="""How far behind to look for values""",
    )

    class Doc:
        short_description = (
            """Specification for a text predictor, where the df has been resolved."""
        )

    embedding_fn: Callable = Field(
        description="""A function used for embedding the text. Should take a
        pandas series of strings and return a pandas dataframe of embeddings.
        Defaults to: None.""",
    )
    embedding_fn_kwargs: Optional[dict] = Field(
        default=None,
        description="""Optional kwargs passed onto the embedding_fn.""",
    )
    resolve_multiple_fn: Union[Callable, str] = Field(
        default="concatenate",
        description="""A function used for resolving multiple values within the
        interval_days, i.e. how to combine texts within the lookbehind window.
        Defaults to: 'concatenate'. Other possible options are 'latest' and
        'earliest'.""",
    )

    def get_output_col_name(self) -> str:
        """Generate the column name for the output column.
        If interval days is a float, the decimal point is changed to an underscore.

        Args:
            additional_feature_name (Optional[str]): additional feature name to
                append to the column name.
        """
        col_str = f"{self.prefix}_{self.feature_base_name}_within_{str(self.lookbehind_days)}_days_{self.aggregation_fn.__name__}_fallback_{self.fallback}"
        return col_str


@dataclass(frozen=True)
class OutcomeSpec:
    base_values_df: pd.DataFrame = BASE_VALUES_DEF
    prefix: str = PRED_PREFIX_DEF
    feature_base_name: str = FEATURE_BASE_NAME_DEF
    lookahead_days: float = LOOKAHEAD_DAYS_DEF
    aggregation_fn: Callable = AGGREGATION_FN_DEFINITION
    fallback: Union[str, int] = FALLBACK_DEFINITION

    class Doc:
        short_description = (
            """Specification for a single outcome, where the df has been resolved."""
        )

    incident: bool = Field(
        description="""Whether the outcome is incident or not.
            I.e., incident outcomes are outcomes you can only experience once.
            For example, type 2 diabetes is incident. Incident outcomes can be handled
            in a vectorised way during resolution, which is faster than non-incident outcomes.""",
    )

    def get_output_col_name(self) -> str:
        """Get the column name for the output column."""
        col_str = f"{self.prefix}_{self.feature_base_name}_within_{str(self.lookahead_days)}_days_{self.aggregation_fn.__name__}_fallback_{self.fallback}"
        return col_str

    def is_dichotomous(self) -> bool:
        """Check if the outcome is dichotomous."""
        return len(self.base_values_df["value"].unique()) <= 2


TemporalSpec = Union[PredictorSpec, OutcomeSpec, TextPredictorSpec]
AnySpec = Union[StaticSpec, TemporalSpec]
