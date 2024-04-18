from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

from iterpy.iter import Iter
from timeseriesflattener.v1.aggregation_fns import (
    AggregationFunType,
    boolean,
    change_per_day,
    count,
    earliest,
    latest,
    maximum,
    mean,
    minimum,
    summed,
    variance,
)
from timeseriesflattener.v1.feature_specs.group_specs import NamedDataframe, V1PGSProtocol

import timeseriesflattener.feature_specs.predictor as v2_specs

from ..aggregators import (
    Aggregator,
    CountAggregator,
    EarliestAggregator,
    HasValuesAggregator,
    MaxAggregator,
    MeanAggregator,
    MinAggregator,
    SlopeAggregator,
    SumAggregator,
    VarianceAggregator,
)
from .meta import ValueFrame

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd


@dataclass
class PredictorGroupSpec(V1PGSProtocol):
    lookbehind_days: Sequence[float | tuple[float, float]]
    named_dataframes: Sequence[NamedDataframe]
    aggregation_fns: Sequence[AggregationFunType]
    fallback: Sequence[int | float | str]
    prefix: str = "pred"
    entity_id_col_name_out: str = "entity_id"

    def _infer_entity_id_col_name(self, df: pd.DataFrame) -> str:
        return next(c for c in df.columns if "entity" in c.lower() or "borger" in c.lower())

    def create_combinations(self) -> Sequence[v2_specs.PredictorSpec]:
        if isinstance(self.lookbehind_days[0], tuple):
            lookbehind_days = [
                (dt.timedelta(days=day[0]), dt.timedelta(days=day[1]))  # type: ignore
                for day in self.lookbehind_days
            ]
        elif isinstance(self.lookbehind_days[0], (float, int)):
            lookbehind_days = [dt.timedelta(days=day) for day in self.lookbehind_days]  # type: ignore
        else:
            raise ValueError(f"Unknown lookbehind_days type: {self.lookbehind_days}")

        aggregatorname2aggregator: Mapping[AggregationFunType, Aggregator] = {
            latest: EarliestAggregator("timestamp"),
            earliest: EarliestAggregator("timestamp"),
            minimum: MinAggregator(),
            maximum: MaxAggregator(),
            mean: MeanAggregator(),
            summed: SumAggregator(),
            count: CountAggregator(),
            variance: VarianceAggregator(),
            boolean: HasValuesAggregator(),
            change_per_day: SlopeAggregator("timestamp"),
        }

        aggregators = Iter(self.aggregation_fns).map(lambda agg: aggregatorname2aggregator[agg])

        items: list[v2_specs.PredictorSpec] = []
        for fallback in self.fallback:
            local_items = (
                Iter(self.named_dataframes)
                .map(
                    lambda ndf: v2_specs.PredictorSpec(
                        value_frame=ValueFrame(
                            init_df=ndf.df.rename(
                                {
                                    "value": ndf.name,
                                    self._infer_entity_id_col_name(
                                        ndf.df
                                    ): self.entity_id_col_name_out,
                                },
                                axis=1,
                            ),
                            entity_id_col_name=self.entity_id_col_name_out,
                        ),
                        lookbehind_distances=lookbehind_days,
                        fallback=fallback,  # noqa: B023
                        aggregators=aggregators.to_list(),
                    )
                )
                .to_list()
            )
            items.extend(local_items)

        return items
