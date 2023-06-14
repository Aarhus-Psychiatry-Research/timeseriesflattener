from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence, Union

import pandas as pd
from pydantic import Field
from timeseriesflattener.feature_specs.single_specs import (
    AGGREGATION_FN_DEFINITION,
)
from timeseriesflattener.utils.pydantic_basemodel import BaseModel


@dataclass(frozen=True)
class NamedDataframe:
    df: pd.DataFrame
    name: str


VALUES_PAIRS_DEF = Field(
    description="""Dataframe and its feature name.""",
)


class GroupSpec(BaseModel, ABC):
    prefix: str = "pred"
    named_dataframes: Sequence[NamedDataframe] = VALUES_PAIRS_DEF
    aggregation_fns: Sequence[Callable] = AGGREGATION_FN_DEFINITION
    fallback: Sequence[Union[int, float, str]]

    @abstractmethod
    def create_combinations(self):
        ...
