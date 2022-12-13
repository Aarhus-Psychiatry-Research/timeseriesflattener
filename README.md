<a href="https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener"><img src="https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/blob/main/docs/_static/icon.png?raw=true" width="220" align="right"/></a>

# Timeseriesflattener

[![github actions docs](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/actions/workflows/documentation.yml/badge.svg)](https://aarhus-psychiatry-research.github.io/timeseriesflattener/)
[![github actions pytest](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/actions/workflows/main_test_and_release.yml/badge.svg)](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/actions)
![python versions](https://img.shields.io/badge/Python-%3E=3.9-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)

[![PyPI version](https://badge.fury.io/py/timeseriesflattener.svg)](https://pypi.org/project/timeseriesflattener/)

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

|    |   id | date                | prediction_time_uuid   |   pred_test_feature_within_30_days_mean_fallback_nan |   outc_test_outcome_within_31_days_maximum_fallback_0_dichotomous |
|---:|-----:|:--------------------|:-----------------------|-----------------------------------------------------:|------------------------------------------------------------------:|
|  0 |    1 | 2020-01-01 00:00:00 | 1-2020-01-01-00-00-00  |                                                  2.5 |                                                                 0 |
|  1 |    1 | 2020-02-01 00:00:00 | 1-2020-02-01-00-00-00  |                                                  1   |                                                                 1 |
|  2 |    2 | 2020-02-01 00:00:00 | 2-2020-02-01-00-00-00  |                                                  4   |                                                                 0 |


## 🗺️ Roadmap
Roadmap is tracked on our [kanban board](https://github.com/orgs/Aarhus-Psychiatry-Research/projects/11/views/1).

## 🤖 Functionality
`timeseriesflattener` includes features required for converting any number of (irregular) time series into a single dataframe with a row for each desired prediction time and columns for each constructed feature. Raw values are aggregated by an ID column, which allows for e.g. aggregating values for each patient independently.

When constructing feature sets from time series in general, or medical time series in particular, there are several choices one needs to make. 

1. When to issue predictions (*prediction time*). E.g. at every physical visit, every morning, or another clinically meaningful time.
2. How far back/ahead from the prediction times to look for raw values (*lookbehind/lookahead*). 
3. Which method to use for aggregation if multiple values exist in the lookbehind.
4. Which value to use if there are no data points in the lookbehind.

![Terminology: A: *Lookbehind* determines how far back in time to look for values for predictors, whereas *lookahead* determines how far into the future to look for outcome values. A *prediction time* indicates at which point the model issues a prediction, and is used as a reference for the *lookbehind* and *lookahead*.  B: Labels for prediction times are true negatives if the outcome never occurs, or if the outcome happens outside the lookahead window. Labels are only true positives if the outcome occurs inside the lookahead window. C) Values within the *lookbehind* window are aggregated using a specified function, for example the mean as shown in this example, or max/min etc. D) Prediction times are dropped if the *lookbehind* extends further back in time than the start of the dataset or if the *lookahead* extends further than the end of the dataset. This behaviour is optional](https://user-images.githubusercontent.com/23191638/207274283-1207e2ce-86c7-4ee8-82a5-d81617c8bb77.png)

The above figure graphically represents the terminology used in the package. **A:** *Lookbehind* determines how far back in time to look for values for predictors, whereas *lookahead* determines how far into the future to look for outcome values. A *prediction time* indicates at which point the model issues a prediction, and is used as a reference for the *lookbehind* and *lookahead*.  **B:** Labels for prediction times are true negatives if the outcome never occurs, or if the outcome happens outside the lookahead window. Labels are only true positives if the outcome occurs inside the lookahead window. **C)** Values within the *lookbehind* window are aggregated using a specified function, for example the mean as shown in this example, or max/min etc. **D)** Prediction times are dropped if the *lookbehind* extends further back in time than the start of the dataset or if the *lookahead* extends further than the end of the dataset. This behaviour is optional.

Multiple lookbehind windows and aggregation functions can be specified for each feature to obtain a rich representation of the data. See the [tutorials](placeholder) for example use cases.


## 📖 Documentation

| Documentation          |                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| 🎛 **[API References]** | The detailed reference for timeseriesflattener's API. Including function documentation |
| 🙋 **[FAQ]**            | Frequently asked question                                                                    |

[api references]: https://Aarhus-Psychiatry-Research.github.io/timeseriesflattener/
[FAQ]: https://Aarhus-Psychiatry-Research.github.io/timeseriesflattener/faq.html

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
| **[Forced admissions]** |              | Prediction of forced admissions of patients to the psychiatric hospital departments. Encompasses two seperate projects: 1. Prediciting at time of discharge for inpatient admissions. 2. Predicting day before outpatient admissions. |
| **[Coercion]**          |              | Prediction of coercion among patients admittied to the hospital psychiatric department. Encompasses predicting mechanical restraint, sedative medication and manual restraint 48 hours before coercion occurs.                        |


[Type 2 diabetes]: https://github.com/Aarhus-Psychiatry-Research/psycop-t2d
[Cancer]: https://github.com/Aarhus-Psychiatry-Research/psycop-cancer
[COPD]: https://github.com/Aarhus-Psychiatry-Research/psycop-copd
[Forced admissions]: https://github.com/Aarhus-Psychiatry-Research/psycop-forced-admissions
[Coercion]: https://github.com/Aarhus-Psychiatry-Research/pyscop-coercion
