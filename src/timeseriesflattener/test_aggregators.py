from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
import polars as pl
import pytest
from timeseriesflattener.testing.utils_for_testing import str_to_pl_df

from ._intermediary_frames import TimeMaskedFrame
from .aggregators import (
    CountAggregator,
    EarliestAggregator,
    HasValuesAggregator,
    LatestAggregator,
    MaxAggregator,
    MeanAggregator,
    MinAggregator,
    SlopeAggregator,
    SumAggregator,
    VarianceAggregator,
)
from .spec_processors.temporal import _aggregate_masked_frame
from .test_flattener import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .aggregators import Aggregator


@dataclass(frozen=True)
class ComplexAggregatorExample:
    aggregator: Aggregator
    input_frame: pl.LazyFrame
    expected_output: pl.DataFrame


@dataclass(frozen=True)
class SingleVarAggregatorExample:
    aggregator: Aggregator
    input_values: Sequence[float | None]
    expected_output_values: Sequence[float]
    fallback_str: str = "nan"

    @property
    def input_frame(self) -> pl.LazyFrame:
        return pl.LazyFrame(
            {
                "prediction_time_uuid": [1] * len(self.input_values),
                "value": self.input_values,
                "timestamp": ["2021-01-01"] * len(self.input_values),
            }
        )

    @property
    def expected_output(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "prediction_time_uuid": [1],
                f"value_{self.aggregator.name}_fallback_{self.fallback_str}": self.expected_output_values,
            }
        )


AggregatorExampleType = Union[ComplexAggregatorExample, SingleVarAggregatorExample]

# TODO: Write integration tests with Earliest and Latest aggregators, since they will depend on sorting


@pytest.mark.parametrize(
    ("example"),
    [
        SingleVarAggregatorExample(
            aggregator=MinAggregator(), input_values=[1, 2], expected_output_values=[1]
        ),
        SingleVarAggregatorExample(
            aggregator=MaxAggregator(), input_values=[1, 2], expected_output_values=[2]
        ),
        SingleVarAggregatorExample(
            aggregator=MeanAggregator(), input_values=[1, 2], expected_output_values=[1.5]
        ),
        SingleVarAggregatorExample(
            aggregator=CountAggregator(), input_values=[1, 2], expected_output_values=[2]
        ),
        SingleVarAggregatorExample(
            aggregator=SumAggregator(), input_values=[1, 2], expected_output_values=[3]
        ),
        SingleVarAggregatorExample(
            aggregator=VarianceAggregator(), input_values=[1, 2], expected_output_values=[0.5]
        ),
        SingleVarAggregatorExample(
            aggregator=HasValuesAggregator(), input_values=[1, 2], expected_output_values=[1], fallback_str="False"
        ),
        SingleVarAggregatorExample(
            aggregator=HasValuesAggregator(),
            input_values=[None],  # type: ignore
            expected_output_values=[0],
            fallback_str="False",
        ),
        ComplexAggregatorExample(
            aggregator=SlopeAggregator(timestamp_col_name="timestamp"),
            input_frame=str_to_pl_df(
                """prediction_time_uuid,timestamp,value
1,2013-01-01,1
1,2013-01-02,3
"""
            ).lazy(),
            expected_output=str_to_pl_df(
                """prediction_time_uuid,value_slope_fallback_nan
1,2.0,
"""
            ),
        ),
        ComplexAggregatorExample(
            aggregator=EarliestAggregator(timestamp_col_name="timestamp"),
            input_frame=str_to_pl_df(
                """prediction_time_uuid,timestamp,value
1,2013-01-01,1, # Kept, first value in 1
1,2013-01-02,2, # Dropped, second value in 1
2,2013-01-04,3, # Dropped, second value in 2
2,2013-01-03,4, # Kept, first value in 2"""
            ).lazy(),
            expected_output=str_to_pl_df(
                """prediction_time_uuid,value_earliest_fallback_nan
1,1,
2,4,"""
            ),
        ),
        ComplexAggregatorExample(
            aggregator=LatestAggregator(timestamp_col_name="timestamp"),
            input_frame=str_to_pl_df(
                """prediction_time_uuid,timestamp,value
1,2013-01-01,1, # Dropped, first value in 1
1,2013-01-02,2, # Kept, second value in 1
2,2013-01-04,3, # Kept, second value in 2
2,2013-01-03,4, # Dropped, first value in 2"""
            ).lazy(),
            expected_output=str_to_pl_df(
                """prediction_time_uuid,value_latest_fallback_nan
1,2,
2,3,"""
            ),
        ),
    ],
    ids=lambda example: example.aggregator.__class__.__name__,
)
def test_aggregator(example: AggregatorExampleType):
    result = _aggregate_masked_frame(
        masked_frame=TimeMaskedFrame(
            init_df=example.input_frame,
            value_col_names=["value"],
            prediction_time_uuid_col_name="prediction_time_uuid",
            timestamp_col_name="timestamp",
        ),
        aggregators=[example.aggregator],
        fallback=np.nan if example.aggregator.name != "bool" else False,
    )

    assert_frame_equal(result.collect(), example.expected_output)