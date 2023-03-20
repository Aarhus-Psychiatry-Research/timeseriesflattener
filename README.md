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
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # Load a dataframe with times you wish to make a prediction
    prediction_times_df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "date": ["2020-01-01", "2020-02-01", "2020-02-01"],
        },
    )
    # Load a dataframe with raw values you wish to aggregate as predictors
    predictor_df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2],
            "date": [
                "2020-01-15",
                "2019-12-10",
                "2019-12-15",
                "2020-01-02",
            ],
            "value": [1, 2, 3, 4],
        },
    )
    # Load a dataframe specifying when the outcome occurs
    outcome_df = pd.DataFrame({"id": [1], "date": ["2020-03-01"], "value": [1]})

    # Specify how to aggregate the predictors and define the outcome
    from timeseriesflattener.feature_spec_objects import OutcomeSpec, PredictorSpec
    from timeseriesflattener.resolve_multiple_functions import maximum, mean

    predictor_spec = PredictorSpec(
        values_df=predictor_df,
        lookbehind_days=30,
        fallback=np.nan,
        entity_id_col_name="id",
        resolve_multiple_fn=mean,
        feature_name="test_feature",
    )
    outcome_spec = OutcomeSpec(
        values_df=outcome_df,
        lookahead_days=31,
        fallback=0,
        entity_id_col_name="id",
        resolve_multiple_fn=maximum,
        feature_name="test_outcome",
        incident=False,
    )

    # Instantiate TimeseriesFlattener and add the specifications
    from timeseriesflattener import TimeseriesFlattener

    ts_flattener = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        entity_id_col_name="id",
        timestamp_col_name="date",
        n_workers=1,
        drop_pred_times_with_insufficient_look_distance=False,
    )
    ts_flattener.add_spec([predictor_spec, outcome_spec])
    df = ts_flattener.get_df()
    df
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
| **[Type 2 Diabetes]**   |              | Prediction of type 2 diabetes among patients with visits to psychiatric hospital departments                                                                                                                                          |
| **[Cancer]**            |              | Prediction of Cancer among patients with visits to psychiatric hospital departments                                                                                                                                                   |
| **[COPD]**              |              | Prediction of Chronic obstructive pulmonary disease (COPD) among patients with visits to psychiatric hospital departments                                                                                                             |
| **[Forced admissions]** |              | Prediction of forced admissions of patients to the psychiatric hospital departments. Encompasses two separate projects: 1. Prediciting at time of discharge for inpatient admissions. 2. Predicting day before outpatient admissions. |
| **[Coercion]**          |              | Prediction of coercion among patients admittied to the hospital psychiatric department. Encompasses predicting mechanical restraint, sedative medication and manual restraint 48 hours before coercion occurs.                        |


[Type 2 diabetes]: https://github.com/Aarhus-Psychiatry-Research/psycop-t2d
[Cancer]: https://github.com/Aarhus-Psychiatry-Research/psycop-cancer
[COPD]: https://github.com/Aarhus-Psychiatry-Research/psycop-copd
[Forced admissions]: https://github.com/Aarhus-Psychiatry-Research/psycop-forced-admissions
[Coercion]: https://github.com/Aarhus-Psychiatry-Research/pyscop-coercion
