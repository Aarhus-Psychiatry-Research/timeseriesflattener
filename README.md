<a href="https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener"><img src="https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/blob/main/docs/_static/icon.png?raw=true" width="200" align="right"/></a>

# Timeseriesflattener

[![github actions docs](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/actions/workflows/documentation.yml/badge.svg)](https://aarhus-psychiatry-research.github.io/timeseriesflattener/)
[![github actions pytest](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/actions/workflows/main_test_and_release.yml/badge.svg)](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/actions)
[![python versions](https://img.shields.io/pypi/pyversions/timeseriesflattener)](https://pypi.org/project/timeseriesflattener/)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)

[![PyPI version](https://badge.fury.io/py/timeseriesflattener.svg)](https://pypi.org/project/timeseriesflattener/)
[![status](https://joss.theoj.org/papers/3bbea8745668d1aa40ff796c6fd3db87/status.svg)](https://joss.theoj.org/papers/3bbea8745668d1aa40ff796c6fd3db87)

Time series from e.g. electronic health records often have a large number of variables, are sampled at irregular intervals and tend to have a large number of missing values. Before this type of data can be used for prediction modelling with machine learning methods such as logistic regression or XGBoost, the data needs to be reshaped.  

In essence, the time series need to be *flattened* so that each prediction time is represented by a set of predictor values and an outcome value. These predictor values can be constructed by aggregating the preceding values in the time series within a certain time window. 

`timeseriesflattener` aims to simplify this process by providing an easy-to-use and fully-specified pipeline for flattening complex time series.  

## 🔧 Installation
To get started using timeseriesflattener simply install it using pip by running the following line in your terminal:

```
pip install timeseriesflattener
```

## ⚡ Quick start

```py
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
        "predictor_value": [1, 2, 3, 4],
    }
)
# Load a dataframe specifying when the outcome occurs
outcome_df = pl.DataFrame({"id": [1], "date": ["2020-03-01"], "outcome_value": [1]})

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

```
Output:

|      |   id | date                | prediction_time_uuid  | pred_test_feature_within_30_days_mean_fallback_nan | outc_test_outcome_within_31_days_maximum_fallback_0_dichotomous |
| ---: | ---: | :------------------ | :-------------------- | -------------------------------------------------: | --------------------------------------------------------------: |
|    0 |    1 | 2020-01-01 00:00:00 | 1-2020-01-01-00-00-00 |                                                2.5 |                                                               0 |
|    1 |    1 | 2020-02-01 00:00:00 | 1-2020-02-01-00-00-00 |                                                  1 |                                                               1 |
|    2 |    2 | 2020-02-01 00:00:00 | 2-2020-02-01-00-00-00 |                                                  4 |                                                               0 |


## 📖 Documentation

| Documentation          |                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------- |
| 🎓 **[Tutorial]**       | Simple and advanced tutorials to get you started using `timeseriesflattener`           |
| 🎛 **[General docs]** | The detailed reference for timeseriesflattener's API. |
| 🙋 **[FAQ]**            | Frequently asked question                                                              |
| 🗺️ **[Roadmap]**        | Kanban board for the roadmap for the project                                           |

[Tutorial]: https://aarhus-psychiatry-research.github.io/timeseriesflattener/tutorials.html
[General docs]: https://Aarhus-Psychiatry-Research.github.io/timeseriesflattener/
[FAQ]: https://Aarhus-Psychiatry-Research.github.io/timeseriesflattener/faq.html
[Roadmap]: https://github.com/orgs/Aarhus-Psychiatry-Research/projects/11/views/1

## 💬 Where to ask questions

| Type                           |                        |
| ------------------------------ | ---------------------- |
| 🚨 **Bug Reports**              | [GitHub Issue Tracker] |
| 🎁 **Feature Requests & Ideas** | [GitHub Issue Tracker] |
| 👩‍💻 **Usage Questions**          | [GitHub Discussions]   |
| 🗯 **General Discussion**       | [GitHub Discussions]   |

[github issue tracker]: https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/issues
[github discussions]: https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/discussions


## 🎓 Projects
PSYCOP projects which use `timeseriesflattener`. Note that some of these projects have yet to be published and are thus private.

| Project                 | Publications |                                                                                                                                                                                                                                       |
| ----------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Type 2 Diabetes](https://github.com/Aarhus-Psychiatry-Research/psycop-common/tree/main/psycop/projects/t2d)**   |              | Prediction of type 2 diabetes among patients with visits to psychiatric hospital departments                                                                                                                                          |
