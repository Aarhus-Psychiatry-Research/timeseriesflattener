import datetime as dt

import polars as pl

from timeseriesflattenerv2.aggregators import MeanAggregator

from .feature_specs import OutcomeSpec, PredictorSpec, ValueFrame

MockValueFrame = ValueFrame(
    init_df=pl.LazyFrame({"value": [1], "timestamp": ["2021-01-01"], "entity_id": [1]}),
    value_col_name="value",
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
