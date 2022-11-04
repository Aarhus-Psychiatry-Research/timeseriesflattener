"""Templates for feature specifications."""

import itertools
from typing import Callable, Literal, Optional, Sequence, Union

import pandas as pd
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra


class BaseModel(PydanticBaseModel):
    """."""

    class Config:
        """An pydantic basemodel, which doesn't allow attributes that are not
        defined in the class."""

        arbitrary_types_allowed = True
        extra = Extra.forbid


class MinSpec(BaseModel):
    """The minimum specification required for all collapsed time series,
    whether looking ahead or behind.

    Mostly used for inheritance below.
    """

    # Input data
    values_df: Union[pd.DataFrame, str]

    # Resolving
    interval_days: Union[int, float]
    resolve_multiple: str
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
    feature_content_for_naming: Optional[str] = None

    # Specifications for col_name
    loader_kwargs: Optional[dict] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set main col name if not specified
        if not self.feature_content_for_naming:
            if not isinstance(self.values_df, pd.DataFrame):
                self.feature_content_for_naming = self.values_df
            else:
                self.feature_content_for_naming = [
                    c
                    for c in self.values_df.columns
                    if c not in [self.timestamp_col_name, self.id_col_name]
                ][0]

        # Convert 'nan' to np.nan object
        if self.fallback == "nan":
            self.fallback = float("nan")

    def get_col_str(self, col_main_override: Optional[str] = None) -> str:
        """."""
        if self.out_col_name_override:
            return self.out_col_name_override

        col_main = (
            col_main_override if col_main_override else self.feature_content_for_naming
        )

        return f"{self.prefix}_{col_main}_within_{self.interval_days}_days_{self.resolve_multiple}_fallback_{self.fallback}"


class PredictorSpec(MinSpec):
    """Specification for a single predictor."""

    prefix: str = "pred"


class OutcomeSpec(MinSpec):
    """Specification for a single outcome."""

    prefix: str = "outc"
    col_main: str = "value"
    incident: bool

    def is_dichotomous(self) -> bool:
        """Check if the outcome is dichotomous."""
        if isinstance(self.values_df, pd.DataFrame):
            return len(self.values_df["value"].unique()) <= 2

        raise TypeError(
            "values_df must be a pandas DataFrame to check if dichotomous. Resolve before checking.",
        )


class FlattenInDirectionSpec(MinSpec):
    """Specification for a single feature."""

    def __init__(
        self,
        direction: Literal["ahead", "behind"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.direction = direction
        self.construct()

    def construct(self):
        """Construct the specification."""
        if self.direction == "ahead":
            return OutcomeSpec(**self.dict())
        elif self.direction == "behind":
            return PredictorSpec(**self.dict())
        else:
            raise ValueError("direction must be either 'forward' or 'backward'.")


class MinGroupSpec(BaseModel):
    """Minimum specification for a group of features, whether they're looking
    ahead or behind."""

    values_df: Sequence[Union[pd.DataFrame, str]]
    interval_days: Sequence[Union[int, float]]
    resolve_multiple: Sequence[str]

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
) -> list[MinSpec]:
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
