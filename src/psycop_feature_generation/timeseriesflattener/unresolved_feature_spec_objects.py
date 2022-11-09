"""Ffeature specifications where the values are not resolved yet."""

from typing import Callable, Optional, Sequence

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
        new_cls: TemporalSpec

        if isinstance(self, UnresolvedOutcomeSpec):
            new_cls = OutcomeSpec
        elif isinstance(self, UnresolvedPredictorSpec):
            new_cls = PredictorSpec
        else:
            raise ValueError("Unknown class type.")

        str2resolve_multiple = resolve_fns.get_all()

        return new_cls(
            values_df=str2df[self.values_lookup_name],
            resolve_multiple_fn=str2resolve_multiple[self.resolve_multiple_fn_name],
            **self.dict()
        )


class UnresolvedGroupSpec(MinGroupSpec):
    values_lookup_name: Sequence[str]
    values_df: Optional[Sequence[pd.DataFrame]]


class UnresolvedPredictorGroupSpec(UnresolvedGroupSpec, PredictorGroupSpec):
    """Specification for a group of predictors, where the df has not been resolved."""

    def create_combinations(self):
        return create_specs_from_group(
            feature_group_spec=self, output_class=UnresolvedPredictorSpec
        )


class UnresolvedOutcomeGroupSpec(UnresolvedGroupSpec, OutcomeGroupSpec):
    """Specification for a group of predictors, where the df has not been resolved."""

    def create_combinations(self):
        return create_specs_from_group(
            feature_group_spec=self, output_class=UnresolvedOutcomeSpec
        )


class UnresolvedStaticSpec(UnresolvedAnySpec):
    """Specification for a static feature, where the df has not been resolved."""


class UnresolvedTemporalSpec(UnresolvedAnySpec):
    resolve_multiple_fn_name: str


class UnresolvedPredictorSpec(UnresolvedTemporalSpec):
    """Specification for a single predictor."""

    prefix: str = "pred"


class UnresolvedOutcomeSpec(UnresolvedTemporalSpec):
    """Specification for a single outcome."""

    prefix: str = "outc"
    col_main: str = "value"
    incident: bool
