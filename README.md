<a href="https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener"><img src="https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/blob/main/docs/_static/icon.png?raw=true" width="220" align="right"/></a>

# Time-series Flattener

![python versions](https://img.shields.io/badge/Python-%3E=3.10-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![github actions pytest](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/actions/workflows/main_test_and_release.yml/badge.svg)](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/actions)
[![PyPI version](https://badge.fury.io/py/timeseriesflattener.svg)](https://pypi.org/project/timeseriesflattener/)

Time series from e.g. electronic health records often have a large number of variables, are sampled at irregular intervals and tend to have a large number of missing values. Before this type of data can be used for prediction modelling with machine learning methods such as logistic regression or XGBoost, the data needs to be reshaped. In essence, the time series need to be *flattened* so that each prediction time is represented by a set of predictor values and an outcome value. These predictor values can be constructed by aggregating the preceding values in the time series within a certain time window. This process lays the foundation for further analyses and requires handling a number of tasks such as 1) how to deal with missing values, 2) which value to use if none fall within the prediction window, and 3) how to handle predictors that attempt to look further back than the start of the dataset.  

`timeseriesflattener` aims to simplify this process by providing an easy-to-use and fully-specified pipeline for flattening complex time series. `timeseriesflattener` implements all the functionality required for aggregating features in specific time windows, grouped by e.g. patient IDs, in a computationally efficient manner. 

## Functionality
`timeseriesflattener` includes features required for converting any number of (irregular) time series into a single dataframe with a row for each desired prediction time and columns for each constructed feature. Raw values are aggregated by an ID column, which allows for e.g. aggregating values for each patient independently.

When constructing feature sets from time series in general, or medical time series in particular, there are several choices one needs to make. 

1. When to issue predictions (*prediction time*). E.g. at every physical visit, every morning, or another clinically meaningful time.
2. How far back/ahead from the prediction times to look for raw values (*lookbehind/lookahead*). 
3. Which method to use for aggregation if multiple values exist in the lookbehind.
4. Which value to use if there are no data points in the lookbehind.

![Terminology: A: *Lookbehind* determines how far back in time to look for values for predictors, whereas *lookahead* determines how far into the future to look for outcome values. A *prediction time* indicates at which point the model issues a prediction, and is used as a reference for the *lookbehind* and *lookahead*.  B: Labels for prediction times are true negatives if the outcome never occurs, or if the outcome happens outside the lookahead window. Labels are only true positives if the outcome occurs inside the lookahead window. C) Values within the *lookbehind* window are aggregated using a specified function, for example the mean as shown in this example, or max/min etc. D) Prediction times are dropped if the *lookbehind* extends further back in time than the start of the dataset or if the *lookahead* extends further than the end of the dataset. This behaviour is optional](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/tree/main/docs/_static/terminology_figure.png)

The above figure shows graphically represents the terminology used in the package. **A:** *Lookbehind* determines how far back in time to look for values for predictors, whereas *lookahead* determines how far into the future to look for outcome values. A *prediction time* indicates at which point the model issues a prediction, and is used as a reference for the *lookbehind* and *lookahead*.  **B:** Labels for prediction times are true negatives if the outcome never occurs, or if the outcome happens outside the lookahead window. Labels are only true positives if the outcome occurs inside the lookahead window. **C)** Values within the *lookbehind* window are aggregated using a specified function, for example the mean as shown in this example, or max/min etc. **D)** Prediction times are dropped if the *lookbehind* extends further back in time than the start of the dataset or if the *lookahead* extends further than the end of the dataset. This behaviour is optional

Multiple lookbehind windows and aggregation functions can be specified for each feature to obtain a rich representation of the data. See the [tutorials](placeholder) for example use cases.

## Roadmap
Roadmap is tracked on our [kanban board](https://github.com/orgs/Aarhus-Psychiatry-Research/projects/11/views/1).

## 🔧 Installation
To get started using timeseriesflattener simply install it using pip by running the following line in your terminal:

```
pip install timeseriesflattener
```

## Quick start

```py
import pandas as pd

# Load a dataframe with times you wish to make a prediction


```

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
