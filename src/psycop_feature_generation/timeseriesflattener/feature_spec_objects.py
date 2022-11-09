"""Templates for feature specifications."""

import itertools
from typing import Callable, Literal, Optional, Sequence, Union

import pandas as pd
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra

from psycop_feature_generation.timeseriesflattener.resolve_multiple_functions import (
    resolve_fns,
)


class BaseModel(PydanticBaseModel):
    """."""

    class Config:
        """An pydantic basemodel, which doesn't allow attributes that are not
        defined in the class."""

        arbitrary_types_allowed = True
        extra = Extra.forbid


class AnySpec(BaseModel):
    """A base class for all feature specifications. Allows for easier type hinting."""


class StaticSpec(AnySpec):
    """Specification for a static feature."""

    values_df: Union[pd.DataFrame, str]


class TemporalSpec(AnySpec):
    """The minimum specification required for all collapsed time series,
    whether looking ahead or behind.

    Mostly used for inheritance below.
    """

    # Input data
    values_df: pd.DataFrame

    # Resolving
    interval_days: Union[int, float]
    resolve_multiple_fn_name: str
    resolve_multiple_fn: Callable = resolve_fns.get_all()["mean"]
    fallback: Union[Callable, int, float, str]

    # Testing
    allowed_nan_value_prop: float = 0.0

    # Lab results
    # Which values to load for medications. Takes "all", "numerical" or "numerical_and_coerce". If "numerical_and_corce", takes inequalities like >=9 and coerces them by a multiplication defined in the loader.
    lab_values_to_load: Optional[
        Literal["all", "numerical", "numerical_and_coerce"]
    ] = None

    # Input col names
    prefix: Optional[str] = None
    id_col_name: str = "dw_ek_borger"
    timestamp_col_name: str = "timestamp"
    source_values_col_name: Optional[str]
    out_col_name_override: Optional[str] = None

    # Output col names
    feature_name: Optional[str] = None

    # Specifications for col_name
    loader_kwargs: Optional[dict] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.resolve_multiple_fn: Callable = resolve_fns.get_all()[
            self.resolve_multiple_fn_name
        ]

        # Convert 'nan' to np.nan object
        if self.fallback == "nan":
            self.fallback = float("nan")

        # Get feature name from the dataframe if it's not specified
        # in the class initiation
        if not self.feature_name:
            self.feature_name = [
                c
                for c in self.values_df.columns
                if c not in [self.timestamp_col_name, self.id_col_name]
            ][0]

    def __eq__(self, other):
        # "combination in list_of_combinations" works for all attributes
        # except for values_df, since the truth value of a dataframe is
        # ambiguous.
        # Instead, use pandas' .equals() method for comparing the dfs,
        # and get the combined truth value.

        # We need to override the __eq__ method.
        other_attributes_equal = all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self.__dict__
            if attr != "values_df"
        )

        dfs_equal = self.values_df.equals(other.values_df)

        return other_attributes_equal and dfs_equal

    def get_col_str(self, col_main_override: Optional[str] = None) -> str:
        """."""
        if self.out_col_name_override:
            return self.out_col_name_override

        col_main = col_main_override if col_main_override else self.feature_name

        col_str = f"{self.prefix}_{col_main}_within_{self.interval_days}_days_{self.resolve_multiple_fn_name}_fallback_{self.fallback}"

        if isinstance(self, OutcomeSpec):
            if self.is_dichotomous():
                col_str += "_dichotomous"

        return col_str


class PredictorSpec(TemporalSpec):
    """Specification for a single predictor, where the df has been resolved"""

    prefix: str = "pred"


class OutcomeSpec(TemporalSpec):
    """Specification for a single predictor, where the df has been resolved"""

    prefix: str = "outc"
    col_main: str = "value"
    incident: bool

    def is_dichotomous(self) -> bool:
        """Check if the outcome is dichotomous."""
        return len(self.values_df["value"].unique()) <= 2


class MinGroupSpec(BaseModel):
    """Minimum specification for a group of features, whether they're looking
    ahead or behind."""

    values_df: Sequence[pd.DataFrame]
    interval_days: Sequence[Union[int, float]]
    resolve_multiple_fn_name: Sequence[str]

    fallback: Sequence[Union[Callable, str]]

    allowed_nan_value_prop: Sequence[float]

    lab_values_to_load: Optional[
        Sequence[Literal["all", "numerical", "numerical_and_coerce"]]
    ]
    # Which values to load for medications. Takes "all", "numerical" or "numerical_and_coerce". If "numerical_and_corce", takes inequalities like >=9 and coerces them by a multiplication defined in the loader.

    source_values_col_name: Optional[Sequence[Optional[str]]] = None

    loader_kwargs: Optional[Sequence[dict]] = None

    def create_combinations(self):
        return create_feature_combinations(self)


class PredictorGroupSpec(MinGroupSpec):
    """Specification for a group of predictors."""


class OutcomeGroupSpec(MinGroupSpec):
    """Specificaiton for a group of outcomes."""

    incident: Sequence[bool]


def create_feature_combinations_from_dict(
    d: dict[str, Union[str, list]],
) -> list[dict[str, Union[str, float, int]]]:
    """Create feature combinations from a dictionary of feature specifications.
    Only unpacks the top level of lists.

    Args:
        d (dict[str]): A dictionary of feature specifications.

    Returns:
        list[dict[str]]: list of all possible combinations of the arguments.
    """

    # Make all elements iterable
    d = {k: v if isinstance(v, list) else [v] for k, v in d.items()}
    keys, values = zip(*d.items())
    # Create all combinations of top level elements
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts


def create_feature_combinations(
    feature_group_spec: MinGroupSpec,
) -> list[TemporalSpec]:
    """Create feature combinations from a FeatureGroupSpec."""

    # Create all combinations of top level elements
    # For each attribute in the FeatureGroupSpec
    feature_group_spec_dict = feature_group_spec.__dict__

    permuted_dicts = create_feature_combinations_from_dict(d=feature_group_spec_dict)

    if isinstance(feature_group_spec, PredictorGroupSpec):
        return [PredictorSpec(**d) for d in permuted_dicts]
    elif isinstance(feature_group_spec, OutcomeGroupSpec):
        return [OutcomeSpec(**d) for d in permuted_dicts]
    else:
        raise ValueError(f"{type(feature_group_spec)} is not a valid GroupSpec.")
