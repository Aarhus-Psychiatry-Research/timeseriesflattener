<a href="https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener"><img src="https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/blob/main/docs/_static/icon.png?raw=true" width="220" align="right"/></a>

# Time-series Flattener

![python versions](https://img.shields.io/badge/Python-%3E=3.10-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![github actions pytest](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/actions/workflows/main_test_and_release.yml/badge.svg)](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/actions)
[![PyPI version](https://badge.fury.io/py/timeseriesflattener.svg)](https://pypi.org/project/timeseriesflattener/)

Time series from e.g. electronic health records often have a large number of variables, are sampled at irregular intervals, and tend to have a large number of missing values. Before this type of data can be used for training using traditional machine learning methods, the data needs to be reshaped. In essense, the time series need to be flattened to only contain a single value for prediction, which is an aggregate of the preceeding values in a certain time window. This process is fraught with methodological pitfalls which can compromise the validity of succeeding analyses.

`timeseriesflattener` aims to simplify this process, by providing an easy-to-use and fully-specified pipeline for flattening complex time series. `timeseriesflattener` implements all the functionality required for aggregating features in specific time windows, grouped by e.g. patient IDs, in a computationally efficient manner.

The package is currently used for feature extraction from electronic health records in the [PSYCOP projects](https://www.cambridge.org/core/journals/acta-neuropsychiatrica/article/psychiatric-clinical-outcome-prediction-psycop-cohort-leveraging-the-potential-of-electronic-health-records-in-the-treatment-of-mental-disorders/73CDCC5B36FF1347E6419EC7B80DEC48).

## 🔧 Installation
To get started using timeseriesflattener simply install it using pip by running the following line in your terminal:

```
pip install timeseriesflattener
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
| **[Coersion]**          |              | Prediction of coercion among patients admittied to the hospital psychiatric department. Encompasses predicting mechanical restraint, sedative medication and manual restraint 48 hours before coercion occurs.                        |


[Type 2 diabetes]: https://github.com/Aarhus-Psychiatry-Research/psycop-t2d
[Cancer]: https://github.com/Aarhus-Psychiatry-Research/psycop-cancer
[COPD]: https://github.com/Aarhus-Psychiatry-Research/psycop-copd
[Forced admissions]: https://github.com/Aarhus-Psychiatry-Research/psycop-forced-admissions
[Coersion]: https://github.com/Aarhus-Psychiatry-Research/pyscop-coercion
