from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING, Sequence

from .._frame_validator import _validate_col_name_columns_exist
from .meta import LookDistances, ValueFrame, ValueType, _lookdistance_to_normalised_lookperiod

if TYPE_CHECKING:
    import polars as pl

    from ..aggregators import Aggregator


@dataclass
class PredictorSpec:
    value_frame: ValueFrame
    lookbehind_distances: InitVar[LookDistances]
    aggregators: Sequence["Aggregator"]
    fallback: ValueType
    column_prefix: str = "pred"

    def __post_init__(self, lookbehind_distances: LookDistances):
        self.normalised_lookperiod = [
            _lookdistance_to_normalised_lookperiod(lookdistance=lookdistance, direction="behind")
            for lookdistance in lookbehind_distances
        ]
        _validate_col_name_columns_exist(obj=self)

    @property
    def df(self) -> "pl.LazyFrame":
        return self.value_frame.df
