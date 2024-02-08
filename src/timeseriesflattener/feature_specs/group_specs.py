import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import pandas as pd
from timeseriesflattener.aggregation_fns import AggregationFunType
from timeseriesflattener.feature_specs.single_specs import AnySpec, OutcomeSpec, PredictorSpec
from timeseriesflattener.utils.pydantic_basemodel import BaseModel


@dataclass(frozen=True)
class NamedDataframe:
    df: pd.DataFrame
    name: str


class PredictorGroupSpec(BaseModel):
    """A group of predictor specifications.

    Args:
        prefix: The prefix to use for the feature names.
        named_dataframes: A list of dataframes and their names.
        aggregation_fns: How to handle multiple values within the lookahead window.
        fallback: A list of fallback values to use if the aggregation fails.
        lookbehind: The number of days to look behind from the prediction time for outcome values.
        incident: Whether the outcome is incident or not, i.e. whether you can experience it more than once.

    """

    # Shared attributes from GroupSpec
    prefix: str = "pred"
    lookbehind_days: Sequence[Union[float, Tuple[float, float]]]
    named_dataframes: Sequence[NamedDataframe]
    aggregation_fns: Sequence[AggregationFunType]
    fallback: Sequence[Union[int, float, str]]

    def create_combinations(self) -> List[PredictorSpec]:
        """Create all combinations from the group spec."""
        combination_dict = create_feature_combinations_from_dict(dictionary=self.__dict__)

        return [
            PredictorSpec(
                prefix=d["prefix"],  # type: ignore
                timeseries_df=d["named_dataframes"].df,  # type: ignore
                feature_base_name=d["named_dataframes"].name,  # type: ignore
                lookbehind_days=d["lookbehind_days"],  # type: ignore
                aggregation_fn=d["aggregation_fns"],  # type: ignore
                fallback=d["fallback"],  # type: ignore
            )
            for d in combination_dict
        ]


class OutcomeGroupSpec(BaseModel):
    """A group of outcome specifications.

    Args:
        prefix: The prefix to use for the feature names.
        named_dataframes: A list of dataframes and their names.
        aggregation_fns: How to handle multiple values within the lookahead window.
        fallback: A list of fallback values to use if the aggregation fails.
        lookahead_days: The number of days to look ahead from the prediction time for outcome values.
        incident: "Whether the outcome is incident or not, i.e. whether you can experience it more than once.
            For example, type 2 diabetes is incident. Incident outcomes can be handled in a vectorised way
            during resolution, which is faster than non-incident outcomes.

    """

    # Shared attributes from GroupSpec
    prefix: str = "outc"
    named_dataframes: Sequence[NamedDataframe]
    aggregation_fns: Sequence[AggregationFunType]
    fallback: Sequence[Union[int, float, str]]

    # Individual attributes
    lookahead_days: Sequence[Union[float, Tuple[float, float]]]
    incident: Sequence[bool]

    def create_combinations(self) -> List[OutcomeSpec]:
        """Create all combinations from the group spec."""
        combination_dict = create_feature_combinations_from_dict(dictionary=self.__dict__)

        return [
            OutcomeSpec(
                prefix=d["prefix"],  # type: ignore
                timeseries_df=d["named_dataframes"].df,  # type: ignore
                feature_base_name=d["named_dataframes"].name,  # type: ignore
                lookahead_days=d["lookahead_days"],  # type: ignore
                aggregation_fn=d["aggregation_fns"],  # type: ignore
                fallback=d["fallback"],  # type: ignore
                incident=d["incident"],  # type: ignore
            )
            for d in combination_dict
        ]


GroupSpec = Union[PredictorGroupSpec, OutcomeGroupSpec]


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
    dictionary = {k: v if isinstance(v, (list, tuple)) else [v] for k, v in dictionary.items()}
    keys, values = zip(*dictionary.items())

    # Create all combinations of top level elements
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutations_dicts  # type: ignore


def create_specs_from_group(feature_group_spec: GroupSpec, output_class: AnySpec) -> List[AnySpec]:
    """Create a list of specs from a GroupSpec."""
    # Create all combinations of top level elements
    # For each attribute in the FeatureGroupSpec

    feature_group_spec_dict = feature_group_spec.__dict__

    permuted_dicts = create_feature_combinations_from_dict(dictionary=feature_group_spec_dict)

    return [output_class(**d) for d in permuted_dicts]  # type: ignore
