# Installation
## For development
`pip install . -e`
The `-e` flag marks the install as editable, "overwriting" the package as you edit the source files.

## For use
`pip install git+https://github.com/Aarhus-Psychiatry-Research/timeseries-flattener.git`

## Purpose
To train baseline models (logistic regression, elastic net, SVM, XGBoost/random forest etc.), we need to represent the longitudinal data in a tabular, flattened way. 

In essence, we need to generate a training example for each prediction time, where that example contains "latest_blood_pressure" (float), "X_diagnosis_within_n_hours" (boolean) etc.

To generate this, I propose the time-series flattener class (`TimeSeriesFlattener`). It builds a dataset like described above.

## TimeSeriesFlattener
```
class FlattenedTimeSeries:
  Attributes:
    prediction_df (dataframe): Cols: dw_ek_borger, prediction_time, (value if relevant).

  Methods:
    add_outcome
        outcome_df (dataframe): Cols: dw_ek_borger, datotid, (value if relevant).
        lookahead_window (float): How far ahead to look for an outcome. If none found, use fallback.
        resolve_multiple (str): How to handle more than one record within the lookbehind. Suggestions: earliest, latest, mean_of_records, max, min.
        fallback (list): How to handle lack of a record within the lookbehind. Suggestions: latest, mean_of_patient, mean_of_population, hardcode (qualified guess)
        name (str): What to name the column
    
    add_predictor
        predictor (dataframe): Cols: dw_ek_borger, datotid, (value if relevant).
        lookback_window (float): How far back to look for a predictor. If none found, use fallback.
        resolve_multiple (str): How to handle more than one record within the lookbehind. Suggestions: earliest, latest, mean_of_records, max, min.
        fallback (list): How to handle lack of a record within the lookbehind. Suggestions: latest, mean_of_patient, mean_of_population, hardcode (qualified guess)
        name (str): What to name the column
```

Inspiration-code can be found in previous commits.

#### Example
```python
import FlattenedTimeSeries

dataset = FlattenedTimeSeries(prediction_df = prediction_times)

dataset.add_outcome(
    outcome=type_2_diabetes,
    lookahead_window=730,
    resolve_multiple="max",
    fallback=[0],
    name="t2d",
)

dataset.add_predictor(
    predictor=hba1c,
    lookback_window=365,
    resolve_multiple="max",
    fallback=["latest", 40],
    name="hba1c",
)
```

Dataset now looks like this:

| dw_ek_borger | datetime_prediction | outc_t2d_within_next_730_days | pred_max_hba1c_within_prev_365_days |
|--------------|---------------------|-------------------------------|-------------------------------------|
| 1            | yyyy-mm-dd hh:mm:ss | 0                             | 48                                  |
| 2            | yyyy-mm-dd hh:mm:ss | 0                             | 40                                  |
| 3            | yyyy-mm-dd hh:mm:ss | 1                             | 44                                  |


For binary outcomes, `add_predictor` with `fallback = [0]` would take a df with only the times where the event occurred, and then generate 0's for the rest. 

I propose we create the above functionality on a just-in-time basis, building the features as we need them.

## Static features
Since we'll all add age, gender etc., might want some higher-level functions for this type of static info. That said, the amount of code for implementing it is very small.