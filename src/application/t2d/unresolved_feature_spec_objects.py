"""Feature specifications where the values are not resolved (i.e. converted from string representation to dataframe/callable) yet."""

from collections.abc import Sequence
from typing import Literal, Optional

import pandas as pd
from pydantic import Field

from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    AnySpec,
    BaseModel,
    MinGroupSpec,
    OutcomeGroupSpec,
    OutcomeSpec,
    PredictorSpec,
    TemporalSpec,
    create_specs_from_group,
)
from psycop_feature_generation.timeseriesflattener.resolve_multiple_functions import (
    resolve_multiple_fns,
)


class UnresolvedAnySpec(BaseModel):
    """Class containing attributes and logic that any and all feature specifications share. Unresolved, meaning that there's no dataframe with values yet."""

    values_lookup_name: str
    # A string which can be resolved to a function which returns a dataframe with the values.

    input_col_name_override: Optional[str]
    # Which column in the dataframe to use as raw values.

    output_col_name_override: Optional[str]
    # Override the name of the output column when resolving a dataframe.

    def resolve_spec(
        self,
        str2df: dict[str, pd.DataFrame],
    ) -> TemporalSpec:
        """Resolve the values_df."""
        str2resolve_multiple = resolve_multiple_fns.get_all()

        kwargs_dict = self.dict()

        # Infer feature_name from values_lookup_name
        kwargs_dict["feature_name"] = kwargs_dict["values_lookup_name"]

        # Remove the attributes that are not allowed in the resolve_to_class,
        # or which are inferred in the return statement.

        # This implementation is super brittle - whenever a new key is added to
        # any class that is resolved, but which isn't added to feature_spec_objects,
        # it breaks. Alternative ideas are very welcome.
        # We can get around it by allowing extras (e.g. attributes which aren't specified) in the feature_spec_objects,
        # but that leaves us open to typos.
        for redundant_key in (
            "values_df",
            "resolve_multiple_fn",
            "lab_values_to_load",
            "values_lookup_name",
            "output_col_name_override",
        ):
            if redundant_key in kwargs_dict:
                kwargs_dict.pop(redundant_key)

        # For classes where the resolve_multiple_fn has already been inferred
        if isinstance(self, UnresolvedPredictorSpec):
            resolve_to_class = PredictorSpec
        elif isinstance(self, UnresolvedOutcomeSpec):
            resolve_to_class = OutcomeSpec
        elif isinstance(self, UnresolvedStaticSpec):
            resolve_to_class = AnySpec

            if self.output_col_name_override:
                kwargs_dict["feature_name"] = self.output_col_name_override
                kwargs_dict["prefix"] = ""

            return resolve_to_class(
                values_df=str2df[self.values_lookup_name], **kwargs_dict
            )

        # And for those where inference is still needed
        return resolve_to_class(
            values_df=str2df[self.values_lookup_name],
            resolve_multiple_fn=str2resolve_multiple[self.resolve_multiple_fn_name],
            **kwargs_dict,
        )


class UnresolvedGroupSpec(MinGroupSpec):
    """Class for things that all feature specification groups share. Unresolved, meaning that there's no dataframe with values yet."""

    values_lookup_name: Sequence[str]

    # Override defaults from MinGroupSpec that don't make sense for an
    # unresolved group spec.
    values_df: Optional[Sequence[pd.DataFrame]]
    feature_name: Optional[list[str]]


class UnresolvedTemporalSpec(UnresolvedAnySpec, TemporalSpec):
    """Class for things that all temporal feature specification groups share. Unresolved, meaning that there's no dataframe with values yet."""

    resolve_multiple_fn_name: str
    # Name of the resolve_multiple function, will be resolved to a function in resolve_multiple_functions.py.

    # Override the requirement of a feature_name from MinGroupSpec,
    # not needed for unresolved groups since it can be inferred from
    # values_lookup_name
    values_df: Optional[pd.DataFrame] = None
    feature_name: Optional[str] = None


class UnresolvedPredictorSpec(UnresolvedTemporalSpec):
    """Specification for a single predictor."""

    prefix: str = "pred"
    # Prefix for the output column name.


class UnresolvedLabPredictorSpec(UnresolvedPredictorSpec):
    """Specification for a single medication predictor, where the df has been
    resolved."""

    # Lab results
    lab_values_to_load: Literal[
        "all",
        "numerical",
        "numerical_and_coerce",
    ] = "numerical_and_coerce"
    # Which values to load for medications. Takes "all", "numerical" or "numerical_and_coerce". If "numerical_and_corce", takes inequalities like >=9 and coerces them by a multiplication defined in the loader.


class UnresolvedPredictorGroupSpec(UnresolvedGroupSpec, MinGroupSpec):
    """Specification for a group of predictors, where the df has not been
    resolved."""

    prefix: str = "pred"

    def create_combinations(self):
        return create_specs_from_group(
            feature_group_spec=self,
            output_class=UnresolvedPredictorSpec,
        )


class UnresolvedLabPredictorGroupSpec(UnresolvedPredictorGroupSpec):
    """Specification for a group of predictors, where the df has not been
    resolved."""

    # Lab results
    lab_values_to_load: Sequence[
        Literal["all", "numerical", "numerical_and_coerce"]
    ] = ["numerical_and_coerce"]
    #  # Which values to load for medications. Takes "all", "numerical" or "numerical_and_coerce". If "numerical_and_coerce", takes inequalities like >=9 and coerces them by a multiplication defined in the loader.

    def create_combinations(self):
        return create_specs_from_group(
            feature_group_spec=self,
            output_class=UnresolvedLabPredictorSpec,
        )


class UnresolvedOutcomeSpec(UnresolvedTemporalSpec):
    """Specification for a single outcome."""

    prefix: str = "outc"
    incident: bool
    # Whether the outcome is incident or not (i.e. whether you can get the same outcome more than once).
    # Incident outcomes can be handled in a vectorized manner when resolving, making it much faster.


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

    prefix: str = "pred"
