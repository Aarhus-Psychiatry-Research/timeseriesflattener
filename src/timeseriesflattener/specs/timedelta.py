from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from timeseriesflattener.validators import validate_col_name_columns_exist

from .value import ValueFrame

if TYPE_CHECKING:
    import polars as pl

    from .timestamp import TimestampValueFrame


@dataclass
class TimeDeltaSpec:
    init_frame: TimestampValueFrame
    fallback: int | float | str | None
    output_name: str
    column_prefix: str = "pred"
    time_format: Literal["seconds", "minutes", "hours", "days", "years"] = "days"
    """Specification for a time delta feature for an entity, i.e. the time between a prediction timestamp and a value timestamp (e.g. a birthdate).
    Useful for e.g. calculating age or the time since a certain event. Joins on the entity_id column.

    init_frame must contain columns:
        entity_id_col_name: The name of the column containing the entity ids. Must be a string, and the column's values must be strings which are unique.
        value_timestamp_col_name: The name of the column containing the timestamps of when the event occurs. Must be a string, and the column's values must be datetimes.

    output_name: the desired name of the feature column.
    time_format:
        """

    def __post_init__(self):
        validate_col_name_columns_exist(obj=self)
        max_values_per_id = (
            self.init_frame.df.get_column(self.init_frame.entity_id_col_name).unique_counts().max()
        )
        if max_values_per_id > 1:  # type: ignore
            raise ValueError(
                f"Expected only one value per {self.init_frame.entity_id_col_name} in the TimestampValueFrame, but found up to {max_values_per_id}."
            )
        self.value_frame = ValueFrame(
            init_df=self.init_frame.df.rename(
                {self.init_frame.value_timestamp_col_name: self.output_name}
            ),
            entity_id_col_name=self.init_frame.entity_id_col_name,
            value_timestamp_col_name=self.output_name,
        )
        self.value_frame.value_col_names = [self.output_name]

    @property
    def df(self) -> pl.DataFrame:
        return self.value_frame.df

    @staticmethod
    def from_primitives(
        df: pl.DataFrame,
        entity_id_col_name: str,
        output_name: str,
        value_timestamp_col_name: str = "timestamp",
        column_prefix: str = "pred",
        fallback: int | float | str | None = 0,
    ) -> TimeDeltaSpec:
        """Create a TimeDeltaSpec from primitives."""
        return TimeDeltaSpec(
            init_frame=TimestampValueFrame(
                init_df=df,
                value_timestamp_col_name=value_timestamp_col_name,
                entity_id_col_name=entity_id_col_name,
            ),
            fallback=fallback,
            output_name=output_name,
            column_prefix=column_prefix,
        )
