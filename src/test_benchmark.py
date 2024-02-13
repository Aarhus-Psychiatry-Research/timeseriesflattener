import datetime as dt
import random
from dataclasses import dataclass
from typing import Literal, Sequence

import joblib
import numpy as np
import polars as pl
import pytest
from iterpy.iter import Iter
from timeseriesflattenerv2.aggregators import MaxAggregator, MeanAggregator
from timeseriesflattenerv2.feature_specs import (
    Aggregator,
    LookDistance,
    PredictionTimeFrame,
    PredictorSpec,
    ValueFrame,
)
from timeseriesflattenerv2.flattener import Flattener

from .timeseriesflattener.aggregation_fns import AggregationFunType


def _generate_valueframe(n_obseravations: int, feature_name: str) -> ValueFrame:
    return ValueFrame(
        init_df=pl.LazyFrame(
            {
                "entity_id": list(range(n_obseravations)),
                feature_name: [random.random() for _ in range(n_obseravations)],
                "timestamp": [
                    dt.datetime.now() + dt.timedelta(days=random.randint(i, i + 10))
                    for i in range(n_obseravations)
                ],
            }
        ),
        value_col_name=feature_name,
    )


@dataclass(frozen=True)
class BenchmarkDataset:
    pred_time_frame: PredictionTimeFrame
    predictor_specs: Sequence[PredictorSpec]


cache = joblib.Memory(".benchmark_cache")


@cache.cache()
def _generate_benchmark_dataset(
    n_pred_times: int,
    n_features: int,
    n_observations_per_pred_time: int,
    aggregations: Sequence[Literal["max", "mean"]],
    lookbehinds: Sequence[LookDistance],
) -> BenchmarkDataset:
    pred_time_df = PredictionTimeFrame(
        init_df=pl.LazyFrame(
            {
                "entity_id": list(range(n_pred_times)),
                "pred_timestamp": [
                    dt.datetime.now() + dt.timedelta(days=random.randint(i, i + 10))
                    for i in range(n_pred_times)
                ],
            }
        )
    )

    aggregations_to_aggregators = {"max": MaxAggregator(), "mean": MeanAggregator()}
    aggregators: Sequence[Aggregator] = (
        Iter(aggregations).map(aggregations_to_aggregators.get).to_list()
    )  # type: ignore

    predictor_specs = [
        PredictorSpec(
            value_frame=_generate_valueframe(
                n_observations_per_pred_time * n_pred_times, f"feature_{i}"
            ),
            lookbehind_distances=lookbehinds,
            aggregators=aggregators,
            fallback=np.nan,
        )
        for i in range(n_features)
    ]

    return BenchmarkDataset(pred_time_frame=pred_time_df, predictor_specs=predictor_specs)


def _v2_aggregator_to_v1(agg: Aggregator) -> AggregationFunType:
    if isinstance(agg, MaxAggregator):
        return maximum
    if isinstance(agg, MeanAggregator):
        return minimum
    raise ValueError(f"Unknown aggregator {agg}")


def _v2_pred_spec_to_v1(pred_spec: PredictorSpec) -> Sequence[V1PSpec]:
    return V1PGSpec(
        lookbehind_days=[d.days for d in pred_spec.lookbehind_distances],
        named_dataframes=[
            NamedDataframe(
                df=pred_spec.value_frame.collect().to_pandas(),
                name=pred_spec.value_frame.value_col_name,
            )
        ],
        aggregation_fns=[_v2_aggregator_to_v1(agg) for agg in pred_spec.aggregators],
        fallback=[np.nan],
    ).create_combinations()


@dataclass(frozen=True)
class TestExample:
    n_pred_times: int = 24_000
    n_features: int = 1
    n_observations_per_pred_time: int = 10
    n_lookbehinds: int = 1
    aggregations: Sequence[Literal["max", "mean"]] = ("max",)

    def get_test_label(self) -> str:
        # Get all parameters and their defaults
        params_with_defaults = TestExample.__dataclass_fields__.keys()
        # Get all parameters that are not at their default value
        non_default_params = {
            k: v
            for k, v in self.__dict__.items()
            if k in params_with_defaults and v != TestExample.__dataclass_fields__[k].default
        }

        # If all non-default parameters are numbers, we can divide them by their default value
        if all(isinstance(v, (int, float)) for v in non_default_params.values()):
            non_default_params_str = (
                "_".join(
                    f"{k}={v / TestExample.__dataclass_fields__[k].default}"
                    for k, v in non_default_params.items()
                )
                + "x"
            )
        else:
            non_default_params_str = "_".join(f"{k}={v}" for k, v in non_default_params.items())
        return f"{non_default_params_str}"


@pytest.mark.parametrize(
    ("example"),
    [
        TestExample(),
        TestExample(n_pred_times=48_000),
        TestExample(n_features=2),
        TestExample(n_lookbehinds=2),
        TestExample(n_lookbehinds=4),
        TestExample(n_lookbehinds=8),
        TestExample(aggregations=["max", "mean"]),
    ],
    ids=lambda e: e.get_test_label(),
)
def test_bench(
    example: TestExample,
    benchmark,  # noqa: ANN001
):
    dataset = _generate_benchmark_dataset(
        n_pred_times=example.n_pred_times,
        n_features=example.n_features,
        n_observations_per_pred_time=example.n_observations_per_pred_time,
        aggregations=example.aggregations,
        lookbehinds=[dt.timedelta(days=i) for i in range(example.n_lookbehinds)],
    )

    flattener = Flattener(predictiontime_frame=dataset.pred_time_frame, compute_lazily=False)

    @benchmark
    def flatten():
        flattener.aggregate_timeseries(specs=dataset.predictor_specs)
