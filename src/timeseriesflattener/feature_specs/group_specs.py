import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

from pydantic import Field
from timeseriesflattener.aggregation_functions import concatenate
from timeseriesflattener.feature_specs.base_group_spec import (
    GroupSpec,
    NamedDataframe,
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
    OutcomeSpec,
    PredictorSpec,
    TextPredictorSpec,
)
from timeseriesflattener.utils.pydantic_basemodel import BaseModel


def create_feature_combinations_from_dict(
    dictionary: Dict[str, Union[str, list]],
) -> List[Dict[str, Union[str, float, int]]]:
    """Create feature combinations from a dictionary of feature specifications.
    Only unpacks the top level of lists.
    Args:
        dictionary (Dict[str]): A dictionary of feature specifications.
    Returns
    -------
        List[Dict[str]]: list of all possible combinations of the arguments.
    """
    # Make all elements iterable
    dictionary = {
        k: v if isinstance(v, (list, tuple)) else [v] for k, v in dictionary.items()
    }
    keys, values = zip(*dictionary.items())

    # Create all combinations of top level elements
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutations_dicts  # type: ignore


def create_specs_from_group(
    feature_group_spec: GroupSpec,
    output_class: AnySpec,
) -> List[AnySpec]:
    """Create a list of specs from a GroupSpec."""
    # Create all combinations of top level elements
    # For each attribute in the FeatureGroupSpec

    feature_group_spec_dict = feature_group_spec.__dict__

    permuted_dicts = create_feature_combinations_from_dict(
        dictionary=feature_group_spec_dict,
    )

    return [output_class(**d) for d in permuted_dicts]  # type: ignore


class PredictorGroupSpec(BaseModel):
    # Shared attributes from GroupSpec
    prefix: str = "pred"
    lookbehind_days: List[float]
    named_dataframes: Sequence[NamedDataframe]
    aggregation_fns: Sequence[Callable]
    fallback: Sequence[Union[Callable, str, float]]

    def create_combinations(self) -> List[PredictorSpec]:
        """Create all combinations from the group spec."""
        combination_dict = create_feature_combinations_from_dict(
            dictionary=self.__dict__,
        )

        return [
            PredictorSpec(
                prefix=d["prefix"],  # type: ignore
                base_values_df=d["named_dataframes"].df,  # type: ignore
                feature_base_name=d["named_dataframes"].name,  # type: ignore
                lookbehind_days=d["lookbehind_days"],  # type: ignore
                aggregation_fn=d["aggregation_fns"],  # type: ignore
                fallback=d["fallback"],  # type: ignore
            )
            for d in combination_dict
        ]


class OutcomeGroupSpec(BaseModel):
    # Shared attributes from GroupSpec
    prefix: str = "outc"
    named_dataframes: Sequence[NamedDataframe]
    aggregation_fns: Sequence[Callable]
    fallback: Sequence[Union[Callable, str, float]]

    # Individual attributes
    lookahead_days: List[float]
    incident: Sequence[bool] = Field(
        description="""Whether the outcome is incident or not, i.e. whether you
            can experience it more than once. For example, type 2 diabetes is incident.
            Incident outcomes can be handled in a vectorised way during resolution,
             which is faster than non-incident outcomes.""",
    )

    def create_combinations(self) -> List[OutcomeSpec]:
        """Create all combinations from the group spec."""
        combination_dict = create_feature_combinations_from_dict(
            dictionary=self.__dict__,
        )

        return [
            OutcomeSpec(
                prefix=d["prefix"],  # type: ignore
                base_values_df=d["values_pairs"].df,  # type: ignore
                feature_base_name=d["values_pairs"].base_feature_name,  # type: ignore
                lookahead_days=d["lookahead_days"],  # type: ignore
                aggregation_fn=d["aggregation_fns"],  # type: ignore
                fallback=d["fallback"],  # type: ignore
            )
            for d in combination_dict
        ]


class TextPredictorGroupSpec(BaseModel):
    # Shared attributes from GroupSpec
    lookbehind_days: List[float]
    named_dataframes: Sequence[NamedDataframe]
    aggregation_fns: Sequence[Callable]
    fallback: Sequence[Union[Callable, str, float]]

    embedding_fn_name: str

    prefix: Sequence[str] = "pred"

    # Individual attributes

    embedding_fn: Sequence[Callable] = Field(
        description="""A function used for embedding the text. Should take a
        pandas series of strings and return a pandas dataframe of embeddings.
        Defaults to: None.""",
    )
    embedding_fn_kwargs: Optional[List[dict]] = Field(
        default=None,
        description="""Optional kwargs passed onto the embedding_fn.""",
    )
    aggregation_fn: Sequence[Callable] = Field(
        default=[concatenate],
        description="""A function used for resolving multiple values within the
        interval_days, i.e. how to combine texts within the lookbehind window.
        Defaults to: concatenate. Other possible options are "latest" and
        "earliest".""",
    )

    def create_combinations(self) -> List[TextPredictorSpec]:
        """Create all combinations from the group spec."""
        combination_dict = create_feature_combinations_from_dict(
            dictionary=self.__dict__,
        )

        return [
            TextPredictorSpec(
                prefix=d["prefix"],  # type: ignore
                base_values_df=d["named_dataframes"].df,  # type: ignore
                feature_base_name=d["embedding_fn_name"],  # type: ignore
                lookbehind_days=d["lookbehind_days"],  # type: ignore
                aggregation_fn=d["aggregation_fns"],  # type: ignore
                fallback=d["fallback"],  # type: ignore
                embedding_fn=d["embedding_fn"],  # type: ignore
                embedding_fn_kwargs=d["embedding_fn_kwargs"],  # type: ignore
            )
            for d in combination_dict
        ]


GroupSpec = Union[PredictorGroupSpec, OutcomeGroupSpec, TextPredictorGroupSpec]
