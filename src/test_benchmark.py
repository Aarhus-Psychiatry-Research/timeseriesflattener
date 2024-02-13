import datetime as dt
import random
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import polars as pl
import pytest
from iterpy.iter import Iter
from timeseriesflattener.feature_specs.group_specs import NamedDataframe
from timeseriesflattener.feature_specs.group_specs import PredictorGroupSpec as V1PGSpec
from timeseriesflattener.feature_specs.single_specs import PredictorSpec as V1PSpec
from timeseriesflattenerv2.aggregators import MaxAggregator, MeanAggregator
from timeseriesflattenerv2.feature_specs import (
    Aggregator,
    LookDistance,
    PredictionTimeFrame,
    PredictorSpec,
    ValueFrame,
)
from timeseriesflattenerv2.flattener import Flattener

from .timeseriesflattener.aggregation_fns import AggregationFunType, maximum, minimum


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


@pytest.mark.parametrize(("n_pred_times"), [1, 10, 100], ids=lambda i: f"preds={i}")
@pytest.mark.parametrize(("n_features"), [1, 10, 100], ids=lambda i: f"feats={i}")
@pytest.mark.parametrize(
    ("n_observations_per_pred_time"), [2, 4, 8], ids=lambda i: f"obs_per_pred={i}"
)
def test_benchmark(
    n_pred_times: int,
    n_features: int,
    n_observations_per_pred_time: int,
    benchmark,  # noqa: ANN001
):
    dataset = _generate_benchmark_dataset(
        n_pred_times=n_pred_times,
        n_features=n_features,
        n_observations_per_pred_time=n_observations_per_pred_time,
        aggregations=["max", "mean"],
        lookbehinds=[dt.timedelta(days=i) for i in range(1, 10)],
    )

    flattener = Flattener(predictiontime_frame=dataset.pred_time_frame, lazy=True)

    @benchmark
    def flatten():
        flattener.aggregate_timeseries(specs=dataset.predictor_specs)


if __name__ == "__main__":
    value = _generate_benchmark_dataset(
        n_pred_times=100,
        n_features=10,
        n_observations_per_pred_time=100,
        aggregations=["max", "mean"],
        lookbehinds=[dt.timedelta(days=i) for i in range(1, 10)],
    )
