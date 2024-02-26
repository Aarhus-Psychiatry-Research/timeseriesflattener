from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

# Load a dataframe with times you wish to make a prediction
prediction_times_df = pl.DataFrame(
    {"id": [1, 1, 2], "date": ["2020-01-01", "2020-02-01", "2020-02-01"]}
)
# Load a dataframe with raw values you wish to aggregate as predictors
predictor_df = pl.DataFrame(
    {
        "id": [1, 1, 1, 2],
        "date": ["2020-01-15", "2019-12-10", "2019-12-15", "2020-01-02"],
        "value": [1, 2, 3, 4],
    }
)
# Load a dataframe specifying when the outcome occurs
outcome_df = pl.DataFrame({"id": [1], "date": ["2020-03-01"], "value": [1]})

# Specify how to aggregate the predictors and define the outcome
from timeseriesflattener import (
    MaxAggregator,
    MinAggregator,
    OutcomeSpec,
    PredictionTimeFrame,
    PredictorSpec,
    ValueFrame,
)

predictor_spec = PredictorSpec(
    value_frame=ValueFrame(
        init_df=predictor_df.lazy(), entity_id_col_name="id", value_timestamp_col_name="date"
    ),
    lookbehind_distances=[dt.timedelta(days=1)],
    aggregators=[MaxAggregator(), MinAggregator()],
    fallback=np.nan,
    column_prefix="pred",
)

outcome_spec = OutcomeSpec(
    value_frame=ValueFrame(
        init_df=outcome_df.lazy(), entity_id_col_name="id", value_timestamp_col_name="date"
    ),
    lookahead_distances=[dt.timedelta(days=1)],
    aggregators=[MaxAggregator(), MinAggregator()],
    fallback=np.nan,
    column_prefix="outc",
)

# Instantiate TimeseriesFlattener and add the specifications
from timeseriesflattener import Flattener

result = Flattener(
    predictiontime_frame=PredictionTimeFrame(
        init_df=prediction_times_df.lazy(), entity_id_col_name="id", timestamp_col_name="date"
    )
).aggregate_timeseries(specs=[predictor_spec, outcome_spec])
result.collect()
