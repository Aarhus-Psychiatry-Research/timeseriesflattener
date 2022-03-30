![python versions](https://img.shields.io/badge/Python-%3E=3.7-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/martbern/d6c40a5b5a3169c079e8b8f778b8e517/raw/badge-psycop-ml-utils-pytest-coverage.json)

# Installation
## For development
`pip install . -e`

The `-e` flag marks the install as editable, "overwriting" the package as you edit the source files.

Recommended to also add black as a pre-commit hook:
`pre-commit install`

## For use
`pip install git+https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git`

# Usage
## Loading data from SQL

Currently only contains one function to load a view from SQL, `sql_load`

```py 
from loaders import sql_load

view = "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret_2011]"
sql = "SELECT * FROM [fct]." + view
df = sql_load(sql, chunksize = None)
```

## Flattening time series
To train baseline models (logistic regression, elastic net, SVM, XGBoost/random forest etc.), we need to represent the longitudinal data in a tabular, flattened way. 

In essence, we need to generate a training example for each prediction time, where that example contains "latest_blood_pressure" (float), "X_diagnosis_within_n_hours" (boolean) etc.

To generate this, I propose the time-series flattener class (`TimeSeriesFlattener`). It builds a dataset like described above.

### TimeSeriesFlattener
```python
class FlattenedDataset:
    def __init__():
        """Class containing a time-series flattened.

        Args:
            prediction_times_df (DataFrame): Dataframe with prediction times.
            prediction_timestamp_colname (str, optional): Colname for timestamps. Defaults to "timestamp".
            id_colname (str, optional): Colname for patients ids. Defaults to "dw_ek_borger".
        """

    def add_outcome():
        """Adds an outcome-column to the dataset

        Args:
            outcome_df (DataFrame): Cols: dw_ek_borger, datotid, value if relevant.
            lookahead_days (float): How far ahead to look for an outcome in days. If none found, use fallback.
            resolve_multiple (str): What to do with more than one value within the lookahead.
                Suggestions: earliest, latest, mean, max, min.
            fallback (List[str]): What to do if no value within the lookahead.
                Suggestions: latest, mean_of_patient, mean_of_population, hardcode (qualified guess)
            timestamp_colname (str): Column name for timestamps
            values_colname (str): Colname for outcome values in outcome_df
            id_colname (str): Column name for citizen id
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
                Defaults to using values_colname.
        """

    def add_predictor():
        """Adds a predictor-column to the dataset

        Args:
            predictor_df (DataFrame): Cols: dw_ek_borger, datotid, value if relevant.
            lookahead_days (float): How far ahead to look for an outcome in days. If none found, use fallback.
            resolve_multiple (str): What to do with more than one value within the lookahead.
                Suggestions: earliest, latest, mean, max, min.
            fallback (List[str]): What to do if no value within the lookahead.
                Suggestions: latest, mean_of_patient, mean_of_population, hardcode (qualified guess)
            outcome_colname (str): What to name the column
            id_colname (str): Column name for citizen id
            timestamp_colname (str): Column name for timestamps
        """
```

Inspiration-code can be found in previous commits.

#### Example
- [ ] Update examples as API matures

```python
import FlattenedDataset

dataset = FlattenedDataset(prediction_times_df = prediction_times, prediction_timestamp_colname = "timestamp", id_colname = "dw_ek_borger")

dataset.add_outcome(
    outcome_df=type_2_diabetes_df,
    lookahead_days=730,
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