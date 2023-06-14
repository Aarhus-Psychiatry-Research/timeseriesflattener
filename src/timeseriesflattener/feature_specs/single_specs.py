from typing import Callable, Optional, Union

import pandas as pd
from pydantic import Field
from timeseriesflattener.aggregation_functions import concatenate
from timeseriesflattener.utils.pydantic_basemodel import BaseModel

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


class StaticSpec(BaseModel):
    """Specification for a static feature.

    Args:
        base_values_df: Dataframe with the values. Should contain columns named "entity_id", "value" and "timestamp".
        feature_base_name: The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_baase_name>_<metadata>.
        prefix: The prefix used for column name generation, e.g.
            <prefix>_<feature_name>_<metadata>. Defaults to "pred".
    """

    base_values_df: pd.DataFrame
    feature_base_name: str
    prefix: str = "pred"

    def get_output_col_name(self) -> str:
        return f"{self.prefix}_{self.feature_base_name}"


class OutcomeSpec(BaseModel):
    """Specification for a static feature.

    Args:
        base_values_df: Dataframe with the values. Should contain columns named "entity_id", "value" and "timestamp".
        feature_base_name: The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_baase_name>_<metadata>.
        lookahead_days: How far ahead from the prediction time to look for outcome values.
        aggregation_fn: How to aggregate multiple values within lookahead days. Should take a grouped dataframe as input and return a single value.
        fallback: Value to return if no values is found within window.
        incident: Whether the outcome is incident or not. E.g. type 2 diabetes is incident because you can only experience it once.
            Incident outcomes can be handled in a vectorised way during resolution, which is faster than non-incident outcomes.
            Requires that each entity only occurs once in the base_values_df.
        prefix: The prefix used for column name generation, e.g.
            <prefix>_<feature_name>_<metadata>. Defaults to "pred".
    """

    base_values_df: pd.DataFrame
    feature_base_name: str
    lookahead_days: Union[int, float]
    aggregation_fn: Callable
    fallback: Union[int, float, str]
    incident: bool
    prefix: str = "outc"

    Field(
        description="""Whether the outcome is incident or not.
            I.e., incident outcomes are outcomes you can only experience once.
            For example, type 2 diabetes is incident. Incident outcomes can be handled
            in a vectorised way during resolution, which is faster than non-incident outcomes.""",
    )

    def get_output_col_name(self) -> str:
        """Get the column name for the output column."""
        col_str = f"{self.prefix}_{self.feature_base_name}_within_{str(self.lookahead_days)}_days_{self.aggregation_fn.__name__}_fallback_{self.fallback}"

        if self.is_dichotomous:
            col_str += "_dichotomous"

        return col_str

    def is_dichotomous(self) -> bool:
        """Check if the outcome is dichotomous."""
        return len(self.base_values_df["value"].unique()) <= 2


class PredictorSpec(BaseModel):
    """Specification for a static feature.

    Args:
        base_values_df: Dataframe with the values. Should contain columns named "entity_id", "value" and "timestamp".
        feature_base_name: The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_baase_name>_<metadata>.
        lookahead_days: How far ahead from the prediction time to look for outcome values.
        aggregation_fn: How to aggregate multiple values within lookahead days. Should take a grouped dataframe as input and return a single value.
        fallback: Value to return if no values is found within window.
        incident: Whether the outcome is incident or not. E.g. type 2 diabetes is incident because you can only experience it once.
            Incident outcomes can be handled in a vectorised way during resolution, which is faster than non-incident outcomes.
            Requires that each entity only occurs once in the base_values_df.
        prefix: The prefix used for column name generation, e.g.
            <prefix>_<feature_name>_<metadata>. Defaults to "pred".
    """

    base_values_df: pd.DataFrame
    feature_base_name: str
    aggregation_fn: Callable
    fallback: Union[int, float, str]
    lookbehind_days: Union[int, float]
    prefix: str = "pred"

    def get_output_col_name(self) -> str:
        """Generate the column name for the output column."""
        col_str = f"{self.prefix}_{self.feature_base_name}_within_{str(self.lookbehind_days)}_days_{self.aggregation_fn.__name__}_fallback_{self.fallback}"

        return col_str


class TextPredictorSpec(BaseModel):
    """Specification for a text predictor, where the df has been resolved.

    Args:
        base_values_df: Dataframe with the values.
        feature_base_name: The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_baase_name>_<metadata>.
        aggregation_fn: How to aggregate multiple values within a window. Can be a string, a function, or a list of functions.
        fallback: Value to return if no values is found within window.
        lookbehind_days: How far behind to look for values. Defaults to LOOKBEHIND_DAYS_DEF.
        prefix: The prefix used for column name generation, e.g.
            <prefix>_<feature_name>_<metadata>. Defaults to "pred".
        embedding_fn: A function used for embedding the text. Should take a
            pandas series of strings and return a pandas dataframe of embeddings.
            Defaults to None.
        embedding_fn_kwargs: Optional kwargs passed onto the embedding_fn.
            Defaults to None.
        resolve_multiple_fn: A function used for resolving multiple
            values within a window. Defaults to concatenate.

    """

    base_values_df: pd.DataFrame
    feature_base_name: str
    fallback: Union[int, float, str]
    embedding_fn: Callable
    embedding_fn_kwargs: Optional[dict] = None
    lookbehind_days: Union[int, float]
    prefix: str = "pred"

    aggregation_fn: Callable = concatenate

    def get_output_col_name(self, additional_feature_name: Optional[str] = None) -> str:
        """Generate the column name for the output column.
        If interval days is a float, the decimal point is changed to an underscore.

        Args:
            additional_feature_name (Optional[str]): additional feature name to
                append to the column name.
        """
        feature_name = self.feature_base_name
        if additional_feature_name is not None:
            feature_name += f"-{additional_feature_name}"

        col_str = f"{self.prefix}_{feature_name}_within_{str(self.lookbehind_days)}_days_{self.aggregation_fn.__name__}_fallback_{self.fallback}"

        return col_str


TemporalSpec = Union[PredictorSpec, OutcomeSpec, TextPredictorSpec]
AnySpec = Union[StaticSpec, TemporalSpec]
