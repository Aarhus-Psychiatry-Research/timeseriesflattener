from dataclasses import dataclass
from typing import Sequence

import polars as pl
from iterpy.iter import Iter
from rich.progress import track

from timeseriesflattenerv2.horizontally_concatenate_dfs import horizontally_concatenate_dfs
from timeseriesflattenerv2.process_spec import process_spec

from .feature_specs import AggregatedFrame, PredictionTimeFrame, ValueSpecification


@dataclass(frozen=True)
class SpecError(Exception):
    description: str


def _specs_are_without_conflicts(specs: Sequence[ValueSpecification]) -> Iter[SpecError]:
    conflicting_value_col_names = (
        Iter(specs)
        .map(lambda s: s.value_frame.value_col_name)
        .groupby(lambda value_col_name: value_col_name)
        .filter(lambda val: len(val[1]) > 1)
        .map(
            lambda error: SpecError(
                description=f"{error[0]} occurs in {len(error[1])} specs. All input value column names must be unique to avoid conflicts in the output."
            )
        )
    )

    return conflicting_value_col_names


@dataclass
class Flattener:
    predictiontime_frame: PredictionTimeFrame
    lazy: bool = True

    def aggregate_timeseries(self, specs: Sequence[ValueSpecification]) -> AggregatedFrame:
        # Check for conflicts in the specs
        conflicting_specs = _specs_are_without_conflicts(specs)

        if conflicting_specs.count() > 0:
            raise SpecError(
                "Conflicting specs."
                + "".join(
                    conflicting_specs.map(lambda error: f"  \n - {error.description}").to_list()
                )
            )

        if not self.lazy:
            self.predictiontime_frame.df = self.predictiontime_frame.collect()  # type: ignore
            for spec in specs:
                spec.value_frame.df = spec.value_frame.collect()  # type: ignore

        # Process and collect the specs. One-by-one, to get feedback on progress.
        dfs: Sequence[pl.LazyFrame] = []
        for spec in track(specs, description="Processing specs..."):
            print(f"Processing spec: {spec.value_frame.value_col_name}")
            processed_spec = process_spec(predictiontime_frame=self.predictiontime_frame, spec=spec)
            if isinstance(processed_spec.df, pl.LazyFrame):
                dfs.append(processed_spec.collect().lazy())
            else:
                dfs.append(processed_spec.df)

        return AggregatedFrame(
            df=horizontally_concatenate_dfs(
                dfs, pred_time_uuid_col_name=self.predictiontime_frame.pred_time_uuid_col_name
            ),
            pred_time_uuid_col_name=self.predictiontime_frame.pred_time_uuid_col_name,
            timestamp_col_name=self.predictiontime_frame.timestamp_col_name,
        )
