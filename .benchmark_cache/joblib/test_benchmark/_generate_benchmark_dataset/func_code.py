# first line: 47
@cache.cache()
def _generate_benchmark_dataset(
    n_pred_times: int,
    n_features: int,
    n_observations_per_pred_time: int,
    aggregations: Sequence[Literal["max", "mean"]],
    lookbehinds: Sequence[LookDistance | tuple[LookDistance, LookDistance]],
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
