from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from timeseriesflattener.aggregators import MeanAggregator

from .value import ValueFrame
from .outcome import BooleanOutcomeSpec, OutcomeSpec
from .temporal import PredictorSpec
from .timedelta import TimeDeltaSpec
from .timestamp import TimestampValueFrame
from .static import StaticSpec

MockDF = pl.DataFrame({"value": [1], "timestamp": ["2021-01-01"], "entity_id": [1]})
MockValueFrame = ValueFrame(init_df=MockDF)


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
                init_df=pl.DataFrame(
                    {"timestamp": ["2021-01-01", "2021-01-02"], "entity_id": [1, 1]}
                ),
                value_timestamp_col_name="timestamp",
            ),
            fallback=0,
            output_name="timedelta",
        )


def test_predictor_from_primitives():
    predictor_spec = PredictorSpec.from_primitives(
        df=MockDF,
        entity_id_col_name="entity_id",
        value_timestamp_col_name="timestamp",
        lookbehind_days=[10],
        aggregators=["mean"],
        fallback=0,
    )

    assert predictor_spec.normalised_lookperiod[0].first == dt.timedelta(days=-10)
    assert predictor_spec.normalised_lookperiod[0].last == dt.timedelta(days=0)


def test_static_spec_from_primitives():
    static_spec = StaticSpec.from_primitives(
        df=MockDF, entity_id_col_name="entity_id", column_prefix="static", fallback=0
    )

    assert static_spec.value_frame.df.shape == MockDF.shape
    assert static_spec.value_frame.df.columns == MockDF.columns


def test_boolean_outcome_spec_correct_dates():
    boolean_outcome_spec = BooleanOutcomeSpec.from_primitives(
        df=MockDF,
        entity_id_col_name="entity_id",
        lookahead_days=[10],
        aggregators=["mean"],
        column_prefix="boolean_outcome",
    )

    assert boolean_outcome_spec.normalised_lookperiod[0].first == dt.timedelta(days=0)
    assert boolean_outcome_spec.normalised_lookperiod[0].last == dt.timedelta(days=10)


def test_boolean_outcome_spec_from_primitives():
    boolean_outcome_spec = BooleanOutcomeSpec.from_primitives(
        df=MockDF,
        entity_id_col_name="entity_id",
        lookahead_days=[10],
        aggregators=["mean"],
        column_prefix="boolean_outcome",
    )

    assert boolean_outcome_spec.normalised_lookperiod[0].first == dt.timedelta(days=0)
    assert boolean_outcome_spec.normalised_lookperiod[0].last == dt.timedelta(days=10)
