from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl
import pytest
from timeseriesflattener.testing.utils_for_testing import str_to_pl_df

from ._process_spec import _aggregate_masked_frame
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
from .feature_specs import Aggregator, TimeMaskedFrame
from .test_flattener import assert_frame_equal


@dataclass(frozen=True)
class ComplexAggregatorExample:
    aggregator: Aggregator
    input: pl.LazyFrame
    expected_output: pl.DataFrame


@dataclass(frozen=True)
class SingleVarAggregatorExample:
    aggregator: Aggregator
    input_values: Sequence[float | None]
    expected_output_values: Sequence[float]

    @property
    def input(self) -> pl.LazyFrame:
        return pl.LazyFrame(
            {"pred_time_uuid": [1] * len(self.input_values), "value": self.input_values}
        )

    @property
    def expected_output(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "pred_time_uuid": [1],
                f"value_{self.aggregator.name}_fallback_nan": self.expected_output_values,
            }
        )


AggregatorExampleType = ComplexAggregatorExample | SingleVarAggregatorExample

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
            aggregator=HasValuesAggregator(), input_values=[1, 2], expected_output_values=[1]
        ),
        SingleVarAggregatorExample(
            aggregator=HasValuesAggregator(),
            input_values=[None],  # type: ignore
            expected_output_values=[0],
        ),
        ComplexAggregatorExample(
            aggregator=SlopeAggregator(timestamp_col_name="timestamp"),
            input=str_to_pl_df(
                """pred_time_uuid,timestamp,value
1,2013-01-01,1
1,2013-01-02,3
"""
            ).lazy(),
            expected_output=str_to_pl_df(
                """pred_time_uuid,value_slope_fallback_nan
1,2.0,
"""
            ),
        ),
        ComplexAggregatorExample(
            aggregator=EarliestAggregator(timestamp_col_name="timestamp"),
            input=str_to_pl_df(
                """pred_time_uuid,timestamp,value
1,2013-01-01,1, # Kept, first value in 1
1,2013-01-02,2, # Dropped, second value in 1
2,2013-01-04,3, # Dropped, second value in 2
2,2013-01-03,4, # Kept, first value in 2"""
            ).lazy(),
            expected_output=str_to_pl_df(
                """pred_time_uuid,value_earliest_fallback_nan
1,1,
2,4,"""
            ),
        ),
        ComplexAggregatorExample(
            aggregator=LatestAggregator(timestamp_col_name="timestamp"),
            input=str_to_pl_df(
                """pred_time_uuid,timestamp,value
1,2013-01-01,1, # Dropped, first value in 1
1,2013-01-02,2, # Kept, second value in 1
2,2013-01-04,3, # Kept, second value in 2
2,2013-01-03,4, # Dropped, first value in 2"""
            ).lazy(),
            expected_output=str_to_pl_df(
                """pred_time_uuid,value_latest_fallback_nan
1,2,
2,3,"""
            ),
        ),
    ],
    ids=lambda example: example.aggregator.__class__.__name__,
)
def test_aggregator(example: AggregatorExampleType):
    result = _aggregate_masked_frame(
        sliced_frame=TimeMaskedFrame(
            init_df=example.input,
            value_col_name="value",
            pred_time_uuid_col_name="pred_time_uuid",
            timestamp_col_name="timestamp",
        ),
        aggregators=[example.aggregator],
        fallback=np.nan,
    )

    assert_frame_equal(result.collect(), example.expected_output)
