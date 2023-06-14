from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence, Union

import pandas as pd
from pydantic import Field
from timeseriesflattener.feature_specs.single_specs import (
    AGGREGATION_FN_DEFINITION,
    FALLBACK_DEFINITION,
)
from timeseriesflattener.utils.pydantic_basemodel import BaseModel


@dataclass(frozen=True)
class Inputdf:
    df: pd.DataFrame
    base_feature_name: str


VALUES_PAIRS_DEF = Field(
    description="""Dataframe and its feature name.""",
)


class GroupSpec(BaseModel, ABC):
    class Doc:
        short_description = """Minimum specification for a group of features,
        whether they're looking ahead or behind.

        Used to generate combinations of features."""

    prefix: str = "pred"
    values_pairs: Sequence[Inputdf] = VALUES_PAIRS_DEF
    aggregation_fns: Sequence[Callable] = AGGREGATION_FN_DEFINITION
    fallback: Sequence[Union[Callable, str, float]] = FALLBACK_DEFINITION

    @abstractmethod
    def create_combinations(self):
        ...
