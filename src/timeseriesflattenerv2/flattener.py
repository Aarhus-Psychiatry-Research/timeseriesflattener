from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Sequence

import polars as pl
import tqdm
from iterpy.iter import Iter
from rich.progress import track

from timeseriesflattenerv2._horisontally_concat import horizontally_concatenate_dfs
from timeseriesflattenerv2._process_spec import process_spec

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


@dataclass(frozen=True)
class MissingColumnNameError(Exception):
    description: str


@dataclass(frozen=True)
class SpecRequirementPair:
    required_columns: Sequence[str]
    spec: ValueSpecification

    def missing_columns(self) -> Iter[str]:
        return Iter(self.required_columns).filter(
            lambda col_name: col_name not in self.spec.value_frame.df.columns
        )


def _specs_contain_required_columns(
    specs: Sequence[ValueSpecification], predictiontime_frame: PredictionTimeFrame
) -> Iter[MissingColumnNameError]:
    missing_col_names = (
        Iter(specs)
        .map(
            lambda s: SpecRequirementPair(
                required_columns=predictiontime_frame.required_columns(), spec=s
            )
        )
        .filter(lambda pair: pair.missing_columns().count() > 0)
        .flatten()
        .map(
            lambda pair: MissingColumnNameError(
                description=f"{pair.missing_columns().to_list()} is missing in the {pair.spec.value_frame.value_col_name} specification."
            )
        )
    )

    return missing_col_names


@dataclass
class Flattener:
    predictiontime_frame: PredictionTimeFrame
    compute_lazily: bool = False
    n_workers: int | None = None
    drop_pred_times_with_insufficient_lookahead_distance: bool = False
    drop_pred_times_with_insufficient_lookbehind_distance: bool = False

    def aggregate_timeseries(self, specs: Sequence[ValueSpecification]) -> AggregatedFrame:
        if self.compute_lazily:
            print(
                "We have encountered performance issues on Windows when using lazy evaluation. If you encounter performance issues, try setting lazy=False."
            )

        # Check for conflicts in the specs
        conflicting_specs = _specs_are_without_conflicts(specs)
        underspecified_specs = _specs_contain_required_columns(
            specs=specs, predictiontime_frame=self.predictiontime_frame
        )
        errors = Iter([conflicting_specs, underspecified_specs]).flatten()

        if errors.count() > 0:
            raise SpecError(
                "Conflicting specs."
                + "".join(errors.map(lambda error: f"  \n - {error.description}").to_list())
            )

        if not self.compute_lazily:
            self.predictiontime_frame.df = self.predictiontime_frame.collect()  # type: ignore
            for spec in specs:
                spec.value_frame.df = spec.value_frame.collect()  # type: ignore

        # Process and collect the specs. One-by-one, to get feedback on progress.
        dfs: Sequence[pl.LazyFrame] = []
        if self.n_workers is None:
            for spec in track(specs, description="Processing specs..."):
                print(f"Processing spec: {spec.value_frame.value_col_name}")
                processed_spec = process_spec(
                    predictiontime_frame=self.predictiontime_frame, spec=spec
                )

                if isinstance(processed_spec.df, pl.LazyFrame):
                    dfs.append(processed_spec.collect().lazy())
                else:
                    dfs.append(processed_spec.df)
        else:
            print(
                "Processing specs with multiprocessing. Note that this multiplies memory pressure by the number of workers. If you run out of memory, try reducing the number of workers, or relying exclusively on Polars paralellisation or setting it to None."
            )
            with Pool(self.n_workers) as pool:
                value_frames = tqdm.tqdm(
                    pool.imap(
                        func=partial(process_spec, predictiontime_frame=self.predictiontime_frame),
                        iterable=specs,
                    )
                )
                dfs = [value_frame.df for value_frame in value_frames]

        return AggregatedFrame(
            df=horizontally_concatenate_dfs(
                dfs, pred_time_uuid_col_name=self.predictiontime_frame.pred_time_uuid_col_name
            ),
            pred_time_uuid_col_name=self.predictiontime_frame.pred_time_uuid_col_name,
            timestamp_col_name=self.predictiontime_frame.timestamp_col_name,
        )
