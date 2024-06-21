from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Union

import polars as pl
import tqdm
from iterpy.iter import Iter
from rich.progress import track

from timeseriesflattener.utils import horizontally_concatenate_dfs
from timeseriesflattener.processors import process_spec

from .intermediary import AggregatedFrame
from .specs.outcome import BooleanOutcomeSpec, OutcomeSpec
from .specs.temporal import PredictorSpec
from .specs.static import StaticSpec
from .specs.timedelta import TimeDeltaSpec
from .processors import ValueSpecification

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import TypeAlias

    from .specs.prediction_times import PredictionTimeFrame


@dataclass(frozen=True)
class SpecError(Exception):
    description: str


def _get_spec_conflicts(specs: Sequence[ValueSpecification]) -> list[SpecError]:
    conflicting_value_col_names = (
        Iter(specs)
        .map(lambda s: s.value_frame.value_col_names)
        .flatten()
        .groupby(lambda value_col_name: value_col_name)
        .filter(lambda val: len(val[1]) > 1)
        .map(
            lambda error: SpecError(
                description=f"The value column '{error[0]}' is specified in {len(error[1])} specs. All value column names must be unique to avoid conflicts in the output."
            )
        )
    )

    return conflicting_value_col_names.to_list()


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
) -> list[MissingColumnNameError]:
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
                description=f"{pair.missing_columns().to_list()} is missing in the {pair.spec.value_frame.value_col_names} specification."
            )
        )
    )

    return missing_col_names.to_list()


@dataclass
class Flattener:
    predictiontime_frame: PredictionTimeFrame
    n_workers: int | None = None
    """Flatten multiple irregular time series to a static feature set.
    
    Args:
        predictiontime_frame: A frame that contains the prediction times.
        n_workers: The number of workers to use for multiprocessing. 
            If None, multiprocessing will be handled entirely by polars, otherwise, 
            multiple processes will be used with joblib. 
            Multiprocessing adds some performance at the cost of memory pressure.
            Note that we already attempted multi-threaded processing with Polars, but the query
            optimiser took an infinite amount of time to optimise the query, 
            so we removed it after commit 73772874802940b6b1e17c110b9c06aa4dd5f8fb.
        """

    def _process_specs(
        self, specs: Sequence[ValueSpecification], step_size: dt.timedelta | None = None
    ) -> Sequence[pl.DataFrame]:
        # Process and collect the specs. One-by-one, to get feedback on progress.
        dfs: Sequence[pl.DataFrame] = []
        if self.n_workers is None:
            for spec in track(specs, description="Processing specs..."):
                print(f"Processing spec: {spec.value_frame.value_col_names}")
                processed_spec = process_spec(
                    predictiontime_frame=self.predictiontime_frame, spec=spec, step_size=step_size
                )
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

        return dfs

    def aggregate_timeseries(
        self, specs: Sequence[ValueSpecification], step_size: dt.timedelta | None = None
    ) -> AggregatedFrame:
        """Perform the aggregation/flattening.

        Args:
            specs: The specifications for the features to be created.
            step_size: The chunk size for prediction time processing.
                If None, will process all prediction times in one go.
                If not None, will process prediction times in chunks of step_size.
                Smaller chunk sizes will reduce memory pressure, but increase processing time.
        """

        # Check for errors in specs
        errors = _get_spec_conflicts(specs) + _specs_contain_required_columns(
            specs=specs, predictiontime_frame=self.predictiontime_frame
        )

        if len(errors) > 0:
            raise SpecError(
                "Conflicting specs."
                + "".join(Iter(errors).map(lambda error: f"  \n - {error.description}").to_list())
            )

        dfs = self._process_specs(specs=specs, step_size=step_size)

        feature_dfs = horizontally_concatenate_dfs(
            dfs,
            prediction_time_uuid_col_name=self.predictiontime_frame.prediction_time_uuid_col_name,
        )

        return AggregatedFrame(
            df=horizontally_concatenate_dfs(
                [self.predictiontime_frame.df, feature_dfs],
                prediction_time_uuid_col_name=self.predictiontime_frame.prediction_time_uuid_col_name,
            ),
            entity_id_col_name=self.predictiontime_frame.entity_id_col_name,
            prediction_time_uuid_col_name=self.predictiontime_frame.prediction_time_uuid_col_name,
            timestamp_col_name=self.predictiontime_frame.timestamp_col_name,
        )
