import datetime as dt
import random
from dataclasses import dataclass
from typing import Literal, Sequence

import polars as pl
import pytest
from iterpy.iter import Iter
from timeseriesflattenerv2.aggregators import MaxAggregator, MeanAggregator
from timeseriesflattenerv2.feature_specs import (
    AggregatedFrame,
    Aggregator,
    LookDistance,
    PredictionTimeFrame,
    PredictorSpec,
    ValueFrame,
)
from timeseriesflattenerv2.flattener import Flattener


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
            fallback=None,
        )
        for i in range(n_features)
    ]

    return BenchmarkDataset(pred_time_frame=pred_time_df, predictor_specs=predictor_specs)


@pytest.mark.parametrize(("n_pred_times"), [1, 10, 100], ids=lambda i: f"preds={i}")
@pytest.mark.parametrize(("n_features"), [1, 10, 100], ids=lambda i: f"feats={i}")
@pytest.mark.parametrize(
    ("n_observations_per_pred_time"), [1, 10, 100], ids=lambda i: f"obs_per_pred={i}"
)
def test_benchmark(n_pred_times: int, n_features: int, n_observations_per_pred_time: int):
    dataset = _generate_benchmark_dataset(
        n_pred_times=n_pred_times,
        n_features=n_features,
        n_observations_per_pred_time=n_observations_per_pred_time,
        aggregations=["max", "mean"],
        lookbehinds=[dt.timedelta(days=i) for i in range(1, 10)],
    )

    @pytest.mark.benchmark()
    def flatten():
        flattener = Flattener(
            predictiontime_frame=dataset.pred_time_frame, lazy=True
        ).aggregate_timeseries(specs=dataset.predictor_specs)
        assert isinstance(flattener, AggregatedFrame)


if __name__ == "__main__":
    value = _generate_benchmark_dataset(
        n_pred_times=100,
        n_features=10,
        n_observations_per_pred_time=100,
        aggregations=["max", "mean"],
        lookbehinds=[dt.timedelta(days=i) for i in range(1, 10)],
    )
