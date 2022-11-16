"""Templates for feature specifications."""

import itertools
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import pandas as pd
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra

from psycop_feature_generation.timeseriesflattener.resolve_multiple_functions import (
    resolve_multiple_fns,
)
from psycop_feature_generation.utils import data_loaders


class BaseModel(PydanticBaseModel):
    """."""

    class Config:
        """A pydantic basemodel, which doesn't allow attributes that are not
        defined in the class."""

        arbitrary_types_allowed = True
        extra = Extra.forbid


class AnySpec(BaseModel):
    """A base class for all feature specifications.

    Allows for easier type hinting.
    """

    values_loader: Optional[Union[Callable, str]] = None
    # Loader for the df. If Callable, it should return a dataframe. If str,
    # tries to resolve from XYZ registry, then calls the function which should return
    # a dataframe.

    values_df: pd.DataFrame
    # Dataframe with the values.

    feature_name: str
    prefix: str
    # Used for column name generation, e.g. <prefix>_<feature_name>.

    input_col_name_override: Optional[str] = None

    # An override for the input column name. If None, will attempt
    # to infer it by looking for the only column that doesn't match id_col_name or timestamp_col_name.

    def __init__(self, **data):
        self.resolve_values_df(data)

        super().__init__(**data)

    def resolve_values_df(self, data: dict[str, Any]):
        if "values_loader" not in data and "values_df" not in data:
            raise ValueError("Either values_loader or df must be specified.")

        if "values_loader" in data and "values_df" in data:
            raise ValueError("Only one of values_loader or df can be specified.")

        if "values_df" not in data:
            if isinstance(data["values_loader"], str):
                data["feature_name"] = data["values_loader"]
                data["values_loader"] = data_loaders.get(data["values_loader"])

            if callable(data["values_loader"]):
                data["values_df"] = data["values_loader"]()
            else:
                raise ValueError("values_loader could not be resolved to a callable")

    def __eq__(self, other):
        # Trying to run `spec in list_of_specs` works for all attributes except for df,
        # since the truth value of a dataframe is ambiguous. To remedy this, we use pandas'
        # .equals() method for comparing the dfs, and get the combined truth value.

        # We need to override the __eq__ method.
        other_attributes_equal = all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self.__dict__
            if attr != "values_df"
        )

        dfs_equal = self.values_df.equals(other.values_df)

        return other_attributes_equal and dfs_equal

    def get_col_str(self) -> str:
        """Create column name for the output column."""
        col_str = f"{self.prefix}_{self.feature_name}"

        return col_str


class StaticSpec(AnySpec):
    """Specification for a static feature."""


class TemporalSpec(AnySpec):
    """The minimum specification required for all collapsed time series (temporal features),
    whether looking ahead or behind.

    Mostly used for inheritance below.
    """

    interval_days: Union[int, float]
    # How far to look in the given direction (ahead for outcomes, behind for predictors)

    resolve_multiple_fn_name: str
    # Name of resolve multiple fn, resolved from resolve_multiple_functions.py

    resolve_multiple_fn: Callable = resolve_multiple_fns.get_all()["mean"]
    # Uses "mean" as a placeholder, is resolved in __init__.
    # If "mean" isn't set, gives a validation error because no Callable is set.

    fallback: Union[Callable, int, float, str]
    # Which value to use if no values are found within interval_days.

    allowed_nan_value_prop: float = 0.0
    # If NaN is higher than this in the input dataframe during resolution, raise an error.

    id_col_name: str = "dw_ek_borger"
    # Col name for ids in the input dataframe.

    timestamp_col_name: str = "timestamp"
    # Col name for timestamps in the input dataframe.

    loader_kwargs: Optional[dict] = None

    # Optional keyword arguments for the data loader

    def __init__(self, **data):
        super().__init__(**data)

        # TODO: Resolve multiple_fn if it is a string, don't
        # use two different attributes.

        # convert resolve_multiple_str to fn
        self.resolve_multiple_fn = resolve_multiple_fns.get_all()[
            self.resolve_multiple_fn_name
        ]

        # override fallback strings with objects
        if self.fallback == "nan":
            self.fallback = float("nan")

    def get_col_str(self) -> str:
        """Generate the column name for the output column."""
        col_str = f"{self.prefix}_{self.feature_name}_within_{self.interval_days}_days_{self.resolve_multiple_fn_name}_fallback_{self.fallback}"

        return col_str


class PredictorSpec(TemporalSpec):
    """Specification for a single predictor, where the df has been resolved."""

    prefix: str = "pred"


class OutcomeSpec(TemporalSpec):
    """Specification for a single predictor, where the df has been resolved."""

    prefix: str = "outc"

    incident: bool

    # Whether the outcome is incident or not, i.e. whether you can experience it more than once.
    # For example, type 2 diabetes is incident. Incident outcomes cna be handled in a vectorised
    # way during resolution, which is faster than non-incident outcomes.

    def get_col_str(self) -> str:
        col_str = super().get_col_str()

        if self.is_dichotomous():
            col_str += "_dichotomous"

        return col_str

    def is_dichotomous(self) -> bool:
        """Check if the outcome is dichotomous."""
        col_name = (
            "value"
            if not self.input_col_name_override
            else self.input_col_name_override
        )

        return len(self.values_df[col_name].unique()) <= 2


class MinGroupSpec(BaseModel):
    """Minimum specification for a group of features, whether they're looking
    ahead or behind."""

    values_df: list[pd.DataFrame]

    input_col_name_override: Optional[str] = None
    # Override for the column name to use as values in df.

    interval_days: list[Union[int, float]]
    # How far to look in the given direction (ahead for outcomes, behind for predictors)

    resolve_multiple_fn_name: list[str]
    # Name of resolve multiple fn, resolved from resolve_multiple_functions.py

    fallback: list[Union[Callable, str]]
    # Which value to use if no values are found within interval_days.

    allowed_nan_value_prop: list[float] = [0.0]
    # If NaN is higher than this in the input dataframe during resolution, raise an error.

    feature_name: str
    # Name of the output column.


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
    """Create a list of specs from a GroupSpec."""

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
    """Specification for a group of outcomes."""

    incident: Sequence[bool]

    # Whether the outcome is incident or not, i.e. whether you can experience it more than once.
    # For example, type 2 diabetes is incident. Incident outcomes can be handled in a vectorised
    # way during resolution, which is faster than non-incident outcomes.

    def create_combinations(self):
        return create_specs_from_group(
            feature_group_spec=self,
            output_class=OutcomeSpec,
        )
