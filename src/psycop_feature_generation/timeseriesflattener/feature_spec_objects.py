"""Templates for feature specifications."""

import itertools
from collections.abc import Sequence
from functools import cache
from typing import Any, Callable, Optional, Union

import pandas as pd
from frozendict import frozendict  # type: ignore
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra
from wasabi import Printer

from psycop_feature_generation.timeseriesflattener.resolve_multiple_functions import (
    resolve_multiple_fns,
)
from psycop_feature_generation.utils import data_loaders

msg = Printer(timestamp=True)


@cache
def load_df_with_cache(
    loader_fn: Callable, kwargs: dict[str, Any], feature_name: str
) -> pd.DataFrame:
    msg.info(f"{feature_name}: Loading values")
    df = loader_fn(**kwargs)
    msg.good(f"{feature_name}: Loaded values")

    return df


def in_dict_and_not_none(d: dict, key: str) -> bool:
    """Check if a key is in a dictionary and its value is not None."""
    return key in d and d[key] is not None


def resolve_values_df(data: dict[str, Any]):
    if "values_loader" not in data and "values_df" not in data:
        raise ValueError("Either values_loader or df must be specified.")

    if in_dict_and_not_none(d=data, key="values_loader") and in_dict_and_not_none(
        key="values_df", d=data
    ):
        raise ValueError("Only one of values_loader or df can be specified.")

    if "values_df" not in data or data["values_df"] is None:
        if isinstance(data["values_loader"], str):
            data["feature_name"] = data["values_loader"]
            data["values_loader"] = data_loaders.get(data["values_loader"])

        if callable(data["values_loader"]):
            if not "loader_kwargs" in data:
                data["loader_kwargs"] = {}

            data["values_df"] = load_df_with_cache(
                loader_fn=data["values_loader"],
                kwargs=frozendict(data["loader_kwargs"]),
                feature_name=data["feature_name"],
            )

    if not isinstance(data["values_df"], pd.DataFrame):
        raise ValueError("values_df must be or resolve to a pandas DataFrame.")

    return data


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

    values_loader: Optional[Callable] = None
    # Loader for the df. Tries to resolve from the resolve_multiple_nfs registry,
    # then calls the function which should return a dataframe.

    loader_kwargs: Optional[dict[str, Any]] = None
    # Optional kwargs for the values_loader

    values_df: Optional[pd.DataFrame] = None
    # Dataframe with the values.

    feature_name: str
    prefix: str
    # Used for column name generation, e.g. <prefix>_<feature_name>.

    input_col_name_override: Optional[str] = None
    # An override for the input column name. If None, will attempt
    # to infer it by looking for the only column that doesn't match id_col_name or timestamp_col_name.

    output_col_name_override: Optional[str] = None
    # Override the generated col name after flattening the time series.

    def __init__(self, **data):
        data = resolve_values_df(data)

        if in_dict_and_not_none(d=data, key="output_col_name_override"):
            data["prefix"] = ""

        super().__init__(**data)

        # Type-hint the values_df to no longer be optional. Changes the outwards-facing
        # type hint so that mypy doesn't complain.
        self.values_df: pd.DataFrame = self.values_df

    def __eq__(self, other):
        """Trying to run `spec in list_of_specs` works for all attributes except for df,
        since the truth value of a dataframe is ambiguous. To remedy this, we use pandas'
        .equals() method for comparing the dfs, and get the combined truth value.

        We need to override the __eq__ method."""
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

    resolve_multiple_fn: Callable

    key_for_resolve_multiple: str
    # Key used to lookup the resolve_multiple_fn in the resolve_multiple_fns registry.
    # Used for column name generation.

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
        if isinstance(data["resolve_multiple_fn"], str):
            # convert resolve_multiple_str to fn
            data["key_for_resolve_multiple"] = data["resolve_multiple_fn"]

            data["resolve_multiple_fn"] = resolve_multiple_fns.get_all()[
                data["resolve_multiple_fn"]
            ]

        super().__init__(**data)

        self.resolve_multiple_fn = data["resolve_multiple_fn"]

        # override fallback strings with objects
        if self.fallback == "nan":
            self.fallback = float("nan")

    def get_col_str(self) -> str:
        """Generate the column name for the output column."""
        col_str = f"{self.prefix}_{self.feature_name}_within_{self.interval_days}_days_{self.key_for_resolve_multiple}_fallback_{self.fallback}"

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

        return len(self.values_df[col_name].unique()) <= 2  # type: ignore


class MinGroupSpec(BaseModel):
    """Minimum specification for a group of features, whether they're looking
    ahead or behind. Used to generate combinations of features."""

    values_loader: list[str]
    # Loader for the df. Tries to resolve from the resolve_multiple_nfs registry,
    # then calls the function which should return a dataframe.

    values_df: Optional[pd.DataFrame] = None
    # Dataframe with the values.

    input_col_name_override: Optional[str] = None
    # Override for the column name to use as values in df.

    output_col_name_override: Optional[str] = None
    # Override for the column name to use as values in the output df.

    interval_days: list[Union[int, float]]
    # How far to look in the given direction (ahead for outcomes, behind for predictors)

    resolve_multiple_fn: list[str]
    # Name of resolve multiple fn, resolved from resolve_multiple_functions.py

    fallback: list[Union[Callable, str]]
    # Which value to use if no values are found within interval_days.

    allowed_nan_value_prop: list[float] = [0.0]
    # If NaN is higher than this in the input dataframe during resolution, raise an error.

    def __init__(self, **data):
        super().__init__(**data)

        # Check that all passed loaders are valid
        invalid_loaders = list(
            set(self.values_loader) - set(data_loaders.get_all().keys())
        )
        if len(invalid_loaders) != 0:
            nl = "\n"  # New line variable as f-string can't handle backslashes
            raise ValueError(
                f"""Some loader strings could not be resolved in the data_loaders catalogue. Did you make a typo? If you want to add your own loaders to the catalogue, see explosion / catalogue on GitHub for info. 
                {nl*2}Loaders that could not be resolved:"""
                f"""{nl}{nl.join(str(loader) for loader in invalid_loaders)}{nl}{nl}"""
                f"""Available loaders:{nl}{nl.join(str(loader) for loader in data_loaders.get_all().keys())}"""
            )

        if self.output_col_name_override:
            input_col_name = (
                "value"
                if not self.input_col_name_override
                else self.input_col_name_override
            )

            self.values_df.rename(
                columns={input_col_name: self.output_col_name_override},
                inplace=True,
            )


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
