"""Feature specifications where the values are not resolved yet."""

from typing import Literal, Optional, Sequence

import pandas as pd

from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    BaseModel,
    MinGroupSpec,
    OutcomeGroupSpec,
    OutcomeSpec,
    PredictorGroupSpec,
    PredictorSpec,
    StaticSpec,
    TemporalSpec,
    create_specs_from_group,
)
from psycop_feature_generation.timeseriesflattener.resolve_multiple_functions import (
    resolve_fns,
)


class UnresolvedAnySpec(BaseModel):
    values_lookup_name: str
    input_col_name_override: Optional[str]
    output_col_name_override: Optional[str]

    def resolve_spec(
        self,
        str2df: dict[str, pd.DataFrame],
    ) -> TemporalSpec:
        """Resolve the values_df."""
        str2resolve_multiple = resolve_fns.get_all()

        kwargs_dict = self.dict()

        # Remove the attributes that are not allowed in the outcome specs,
        # or which will be duplicates because they are manually specified in
        # the return statement.
        for redundant_key in (
            "values_df",
            "resolve_multiple_fn",
            "lab_values_to_load",
            "values_lookup_name",
        ):
            if redundant_key in kwargs_dict:
                kwargs_dict.pop(redundant_key)

        if isinstance(self, UnresolvedPredictorSpec):
            resolve_to_class = PredictorSpec
        elif isinstance(self, UnresolvedOutcomeSpec):
            resolve_to_class = OutcomeSpec
        elif isinstance(self, UnresolvedStaticSpec):
            resolve_to_class = StaticSpec

            return resolve_to_class(
                values_df=str2df[self.values_lookup_name], **kwargs_dict
            )

        return resolve_to_class(
            values_df=str2df[self.values_lookup_name],
            resolve_multiple_fn=str2resolve_multiple[self.resolve_multiple_fn_name],
            **kwargs_dict,
        )


class UnresolvedGroupSpec(MinGroupSpec):
    values_lookup_name: Sequence[str]
    values_df: Optional[Sequence[pd.DataFrame]]
    feature_name: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.feature_name:
            self.feature_name = self.values_lookup_name


class UnresolvedTemporalSpec(UnresolvedAnySpec, TemporalSpec):
    resolve_multiple_fn_name: str
    values_df: Optional[pd.DataFrame] = None

    # Override methods used in init of TemporalSpec
    def resolve_multiple_str_to_fn(self):
        pass

    def override_fallback_strings_with_objects(self):
        pass

    def infer_feature_name_from_df(self):
        pass


class UnresolvedPredictorSpec(UnresolvedTemporalSpec):
    """Specification for a single predictor."""

    prefix: str = "pred"


class UnresolvedLabPredictorSpec(UnresolvedPredictorSpec):
    """Specification for a single medication predictor, where the df has been resolved."""

    # Lab results
    # Which values to load for medications. Takes "all", "numerical" or "numerical_and_coerce". If "numerical_and_corce", takes inequalities like >=9 and coerces them by a multiplication defined in the loader.
    lab_values_to_load: Literal[
        "all", "numerical", "numerical_and_coerce"
    ] = "numerical_and_coerce"


class UnresolvedPredictorGroupSpec(UnresolvedGroupSpec, PredictorGroupSpec):
    """Specification for a group of predictors, where the df has not been
    resolved."""

    def create_combinations(self):
        return create_specs_from_group(
            feature_group_spec=self,
            output_class=UnresolvedPredictorSpec,
        )


class UnresolvedLabPredictorGroupSpec(UnresolvedPredictorGroupSpec):
    """Specification for a group of predictors, where the df has not been
    resolved."""

    # Lab results
    # Which values to load for medications. Takes "all", "numerical" or "numerical_and_coerce". If "numerical_and_corce", takes inequalities like >=9 and coerces them by a multiplication defined in the loader.
    lab_values_to_load: Sequence[
        Literal["all", "numerical", "numerical_and_coerce"]
    ] = ["numerical_and_coerce"]

    def create_combinations(self):
        return create_specs_from_group(
            feature_group_spec=self,
            output_class=UnresolvedLabPredictorSpec,
        )


class UnresolvedOutcomeSpec(UnresolvedTemporalSpec):
    """Specification for a single outcome."""

    prefix: str = "outc"
    col_main: str = "value"
    incident: bool


class UnresolvedOutcomeGroupSpec(UnresolvedGroupSpec, OutcomeGroupSpec):
    """Specification for a group of predictors, where the df has not been
    resolved."""

    def create_combinations(self):
        return create_specs_from_group(
            feature_group_spec=self,
            output_class=UnresolvedOutcomeSpec,
        )


class UnresolvedStaticSpec(UnresolvedAnySpec):
    """Specification for a static feature, where the df has not been
    resolved."""
