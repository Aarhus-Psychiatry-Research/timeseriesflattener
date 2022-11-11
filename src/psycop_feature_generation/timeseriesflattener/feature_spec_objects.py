"""Templates for feature specifications."""

import itertools
from abc import abstractmethod
from typing import Callable, Iterable, Literal, Optional, Sequence, Union

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
    """A base class for all feature specifications.

    Allows for easier type hinting.
    """

    values_df: pd.DataFrame
    feature_name: str
    prefix: str
    # Used for column name generation, e.g. pred_<feature_name>.

    input_col_name_override: Optional[str] = None
    # An override for the input column name. If None, will attempt
    # to infer it by looking for the only column that doesn't match id_col_name or timestamp_col_name.

    def get_col_str(self) -> str:
        """."""
        col_str = f"{self.prefix}_{self.feature_name}"

        if isinstance(self, OutcomeSpec):
            if self.is_dichotomous():
                col_str += "_dichotomous"

        return col_str


class StaticSpec(AnySpec):
    """Specification for a static feature."""


class TemporalSpec(AnySpec):
    """The minimum specification required for all collapsed time series,
    whether looking ahead or behind.

    Mostly used for inheritance below.
    """

    # Resolving
    interval_days: Union[int, float]
    resolve_multiple_fn_name: str

    resolve_multiple_fn: Callable = resolve_fns.get_all()["mean"]
    # Uses "mean" as a placeholder, is resolved in __init__.
    # If "mean" isn't set, gives a validation error because no Callable is
    # set.

    fallback: Union[Callable, int, float, str]

    # Testing
    allowed_nan_value_prop: float = 0.0

    # Input col names
    prefix: str
    id_col_name: str = "dw_ek_borger"
    timestamp_col_name: str = "timestamp"

    # Output col names
    feature_name: str

    # Specifications for col_name
    loader_kwargs: Optional[dict] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # convert resolve_multiple_str to fn
        self.resolve_multiple_fn = resolve_fns.get_all()[self.resolve_multiple_fn_name]

        # override fallback strings with objects
        if self.fallback == "nan":
            self.fallback = float("nan")

    def get_col_str(self, col_main_override: Optional[str] = None) -> str:
        """."""
        col_str = f"{self.prefix}_{self.feature_name}_within_{self.interval_days}_days_{self.resolve_multiple_fn_name}_fallback_{self.fallback}"

        if isinstance(self, OutcomeSpec):
            if self.is_dichotomous():
                col_str += "_dichotomous"

        return col_str

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


class PredictorSpec(TemporalSpec):
    """Specification for a single predictor, where the df has been resolved."""

    prefix: str = "pred"


class OutcomeSpec(TemporalSpec):
    """Specification for a single predictor, where the df has been resolved."""

    prefix: str = "outc"
    col_main: str = "value"
    incident: bool

    def is_dichotomous(self) -> bool:
        """Check if the outcome is dichotomous."""
        return len(self.values_df["value"].unique()) <= 2


class MinGroupSpec(BaseModel):
    """Minimum specification for a group of features, whether they're looking
    ahead or behind."""

    values_df: list[pd.DataFrame]
    feature_name: str
    input_col_name_override: Optional[str] = None

    interval_days: list[Union[int, float]]
    resolve_multiple_fn_name: list[str]
    fallback: list[Union[Callable, str]]

    allowed_nan_value_prop: list[float] = [0.0]

    loader_kwargs: Optional[list[dict]] = None


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
    d = {k: v if isinstance(v, (list, tuple)) else [v] for k, v in d.items()}
    keys, values = zip(*d.items())

    # Create all combinations of top level elements
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutations_dicts


def create_specs_from_group(
    feature_group_spec: MinGroupSpec,
    output_class: AnySpec,
) -> list[AnySpec]:

    # Create all combinations of top level elements
    # For each attribute in the FeatureGroupSpec
    feature_group_spec_dict = feature_group_spec.__dict__

    permuted_dicts = create_feature_combinations_from_dict(d=feature_group_spec_dict)

    return [output_class(**d) for d in permuted_dicts]  # type: ignore


class PredictorGroupSpec(MinGroupSpec):
    """Specification for a group of predictors."""

    def create_combinations(self):
        return create_specs_from_group(
            feature_group_spec=self,
            output_class=PredictorSpec,
        )


class OutcomeGroupSpec(MinGroupSpec):
    """Specificaiton for a group of outcomes."""

    incident: Sequence[bool]

    def create_combinations(self):
        return create_specs_from_group(
            feature_group_spec=self,
            output_class=OutcomeSpec,
        )
