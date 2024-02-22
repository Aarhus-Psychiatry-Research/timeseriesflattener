from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .._frame_validator import _validate_col_name_columns_exist
from .meta import ValueFrame, ValueType

if TYPE_CHECKING:
    import polars as pl

    from .timestamp_frame import TimestampValueFrame


@dataclass
class TimeDeltaSpec:
    init_frame: "TimestampValueFrame"
    fallback: ValueType
    output_name: str
    column_prefix: str = "pred"
    time_format: Literal["seconds", "minutes", "hours", "days", "years"] = "days"

    def __post_init__(self):
        _validate_col_name_columns_exist(obj=self)
        max_values_per_id = (
            self.init_frame.collect()
            .get_column(self.init_frame.entity_id_col_name)
            .unique_counts()
            .max()
        )
        if max_values_per_id > 1:  # type: ignore
            raise ValueError(
                f"Expected only one value per {self.init_frame.entity_id_col_name} in the TimestampValueFrame, but found up to {max_values_per_id}."
            )
        self.value_frame = ValueFrame(
            init_df=self.init_frame.df.rename(
                {self.init_frame.value_timestamp_col_name: self.output_name}
            ),
            value_col_name=self.output_name,
            entity_id_col_name=self.init_frame.entity_id_col_name,
            value_timestamp_col_name=self.output_name,
        )

    @property
    def df(self) -> "pl.LazyFrame":
        return self.init_frame.df
