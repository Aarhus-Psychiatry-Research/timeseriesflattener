from dataclasses import dataclass
from typing import Tuple, Union

import pandas as pd
from timeseriesflattener.aggregation_fns import AggregationFunType
from timeseriesflattener.utils.pydantic_basemodel import BaseModel


@dataclass(frozen=True)
class LookPeriod:
    min_days: float
    max_days: float

    def __post_init__(self):
        if self.min_days > self.max_days:
            raise ValueError(
                f"Invalid LookPeriod. The min_days ({self.min_days}) must be smaller than the max_days {self.max_days}."
            )


@dataclass(frozen=True)
class CoercedFloats:
    lookperiod: LookPeriod
    fallback: Union[float, int]


def can_be_coerced_losslessly_to_int(value: float) -> bool:
    try:
        int_version = int(value)
        return (int_version - value) == 0
    except ValueError:
        return False


def coerce_floats(lookperiod: LookPeriod, fallback: float) -> CoercedFloats:
    min_days = (
        lookperiod.min_days
        if not can_be_coerced_losslessly_to_int(lookperiod.min_days)
        else int(lookperiod.min_days)
    )
    max_days = (
        lookperiod.max_days
        if not can_be_coerced_losslessly_to_int(lookperiod.max_days)
        else int(lookperiod.max_days)
    )

    coerced_lookperiod = LookPeriod(min_days=min_days, max_days=max_days)

    fallback = fallback if not can_be_coerced_losslessly_to_int(fallback) else int(fallback)

    return CoercedFloats(lookperiod=coerced_lookperiod, fallback=fallback)


class StaticSpec(BaseModel):
    """Specification for a static feature.

    Args:
        timeseries_df: Dataframe with the values. Should contain columns:
            entity_id (int, float, str): ID of the entity each time series belongs to
            value (int, float, str): The values in the timeseries.
            timestamp (datetime): Timestamps
            NOTE: Column names can be overridden when initialising TimeSeriesFlattener.
        feature_base_name: The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_baase_name>_<metadata>.
        prefix: The prefix used for column name generation, e.g.
            <prefix>_<feature_name>_<metadata>. Defaults to "pred".
    """

    timeseries_df: pd.DataFrame
    feature_base_name: str
    prefix: str = "pred"

    def get_output_col_name(self) -> str:
        return f"{self.prefix}_{self.feature_base_name}"


def get_temporal_col_name(
    prefix: str,
    feature_base_name: str,
    lookperiod: LookPeriod,
    aggregation_fn: AggregationFunType,
    fallback: Union[float, int],
) -> str:
    """Get the column name for the temporal feature."""
    coerced = coerce_floats(lookperiod=lookperiod, fallback=fallback)
    lookperiod_str = (
        f"{coerced.lookperiod.max_days!s}"
        if coerced.lookperiod.min_days == 0
        else f"{coerced.lookperiod.min_days!s}_to_{coerced.lookperiod.max_days!s}"
    )
    col_str = f"{prefix}_{feature_base_name}_within_{lookperiod_str}_days_{aggregation_fn.__name__}_fallback_{coerced.fallback}"
    return col_str


class OutcomeSpec(BaseModel):
    """Specification for an outcome feature.

    Args:
        timeseries_df: Dataframe with the values. Should contain columns:
            entity_id (int, float, str): ID of the entity each time series belongs to
            value (int, float, str): The values in the timeseries.
            timestamp (datetime): Timestamps
            NOTE: Column names can be overridden when initialising TimeSeriesFlattener.
        feature_base_name: The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_baase_name>_<metadata>.
        lookahead_days: In which interval from the prediction time to look for outcome values.
            Can be tuple of two floats specifying (min_days, max_days) or float | int which will resolve to (0, value).
        aggregation_fn: How to aggregate multiple values within lookahead days. Should take a grouped dataframe as input and return a single value.
        fallback: Value to return if no values is found within window.
        incident: Whether the outcome is incident or not. E.g. type 2 diabetes is incident because you can only experience it once.
            Incident outcomes can be handled in a vectorised way during resolution, which is faster than non-incident outcomes.
            Requires that each entity only occurs once in the timeseries_df.
        prefix: The prefix used for column name generation, e.g.
            <prefix>_<feature_name>_<metadata>. Defaults to "pred".
    """

    timeseries_df: pd.DataFrame
    feature_base_name: str
    lookahead_days: Union[float, Tuple[float, float]]
    aggregation_fn: AggregationFunType
    fallback: Union[float, int]
    incident: bool
    prefix: str = "outc"

    @property
    def lookahead_period(self) -> LookPeriod:
        if isinstance(self.lookahead_days, (float, int)):
            return LookPeriod(min_days=0, max_days=self.lookahead_days)
        return LookPeriod(min_days=self.lookahead_days[0], max_days=self.lookahead_days[1])

    def get_output_col_name(self) -> str:
        """Get the column name for the output column."""
        col_str = get_temporal_col_name(
            prefix=self.prefix,
            feature_base_name=self.feature_base_name,
            lookperiod=self.lookahead_period,
            aggregation_fn=self.aggregation_fn,
            fallback=self.fallback,
        )

        if self.is_dichotomous():
            col_str += "_dichotomous"

        return col_str

    def is_dichotomous(self) -> bool:
        """Check if the outcome is dichotomous."""
        return len(self.timeseries_df["value"].unique()) <= 2


class PredictorSpec(BaseModel):
    """Specification for predictor feature.

    Args:
        timeseries_df: Dataframe with the values. Should contain columns:
            entity_id (int, float, str): ID of the entity each time series belongs to
            value (int, float, str): The values in the timeseries.
            timestamp (datetime): Timestamps
            NOTE: Column names can be overridden when initialising TimeSeriesFlattener.
        feature_base_name: The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_baase_name>_<metadata>.
        lookbehind_days: In which interval from the prediction time to look for predictor values.
            Can be tuple of two floats specifying (min_days, max_days) or float | int which will resolve to (0, value).
        aggregation_fn: How to aggregate multiple values within lookbehind days. Should take a grouped dataframe as input and return a single value.
        fallback: Value to return if no values is found within window.
        prefix: The prefix used for column name generation, e.g.
            <prefix>_<feature_name>_<metadata>. Defaults to "pred".
    """

    timeseries_df: pd.DataFrame
    feature_base_name: str
    aggregation_fn: AggregationFunType
    fallback: Union[float, int]
    lookbehind_days: Union[float, Tuple[float, float]]
    prefix: str = "pred"

    @property
    def lookbehind_period(self) -> LookPeriod:
        if isinstance(self.lookbehind_days, (float, int)):
            return LookPeriod(min_days=0, max_days=self.lookbehind_days)
        return LookPeriod(min_days=self.lookbehind_days[0], max_days=self.lookbehind_days[1])

    def get_output_col_name(self) -> str:
        """Generate the column name for the output column."""
        return get_temporal_col_name(
            prefix=self.prefix,
            feature_base_name=self.feature_base_name,
            lookperiod=self.lookbehind_period,
            aggregation_fn=self.aggregation_fn,
            fallback=self.fallback,
        )


TemporalSpec = Union[PredictorSpec, OutcomeSpec]
AnySpec = Union[StaticSpec, TemporalSpec]
