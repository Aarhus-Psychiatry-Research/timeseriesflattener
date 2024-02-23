from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from timeseriesflattener.aggregators import MeanAggregator

from .meta import ValueFrame
from .outcome import OutcomeSpec
from .predictor import PredictorSpec
from .timedelta import TimeDeltaSpec
from .timestamp_frame import TimestampValueFrame

MockValueFrame = ValueFrame(
    init_df=pl.LazyFrame({"value": [1], "timestamp": ["2021-01-01"], "entity_id": [1]})
)


def test_predictor_spec_post_init():
    lookdistance_start = dt.timedelta(days=1)
    lookdistance_end = dt.timedelta(days=10)

    predictor_spec = PredictorSpec(
        value_frame=MockValueFrame,
        lookbehind_distances=[(lookdistance_start, lookdistance_end)],
        aggregators=[MeanAggregator()],
        fallback=0,
    )

    assert predictor_spec.normalised_lookperiod[0].first == -lookdistance_end
    assert predictor_spec.normalised_lookperiod[0].last == -lookdistance_start


def test_outcome_spec_post_init():
    lookdistance_start = dt.timedelta(days=1)
    lookdistance_end = dt.timedelta(days=10)

    outcome_spec = OutcomeSpec(
        value_frame=MockValueFrame,
        lookahead_distances=[(lookdistance_start, lookdistance_end)],
        aggregators=[MeanAggregator()],
        fallback=0,
    )

    assert outcome_spec.normalised_lookperiod[0].first == lookdistance_start
    assert outcome_spec.normalised_lookperiod[0].last == lookdistance_end


def test_timedelta_spec_error_if_non_unique_ids():
    with pytest.raises(ValueError, match=".*Expected only one value.*"):
        TimeDeltaSpec(
            init_frame=TimestampValueFrame(
                init_df=pl.LazyFrame(
                    {"timestamp": ["2021-01-01", "2021-01-02"], "entity_id": [1, 1]}
                ),
                value_timestamp_col_name="timestamp",
            ),
            fallback=0,
            output_name="timedelta",
        )
