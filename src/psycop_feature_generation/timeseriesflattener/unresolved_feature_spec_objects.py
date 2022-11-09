from typing import Callable

import pandas as pd

from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    OutcomeSpec,
    PredictorSpec,
    TemporalSpec,
)


class UnresolvedSpec(TemporalSpec):
    values_lookup_name: str
    resolve_multiple_fn_name: str

    def resolve_spec(
        self, str2df: dict[str, pd.DataFrame], str2resolve_multiple: dict[str, Callable]
    ) -> TemporalSpec:
        """Resolve the values_df."""
        new_cls: TemporalSpec

        if isinstance(self, UnresolvedOutcomeSpec):
            new_cls = OutcomeSpec
        elif isinstance(self, UnresolvedPredictorSpec):
            new_cls = PredictorSpec
        else:
            raise ValueError("Unknown class type.")

        return new_cls(
            values_df=str2df[self.values_lookup_name],
            resolve_multiple_fn=str2resolve_multiple[self.resolve_multiple_fn_name],
            **self.dict()
        )


class UnresolvedPredictorSpec(UnresolvedSpec):
    """Specification for a single predictor."""

    prefix: str = "pred"


class UnresolvedOutcomeSpec(UnresolvedSpec):
    """Specification for a single outcome."""

    prefix: str = "outc"
    col_main: str = "value"
    incident: bool
