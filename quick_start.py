import pandas as pd
import numpy as np

if __name__ == "__main__":

    # Load a dataframe with times you wish to make a prediction
    prediction_times_df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "date": ["2020-01-01", "2020-02-01", "2020-02-01", "2020-03-01"],
        }
    )
    # Load a dataframe with raw values you wish to aggregate as predictors
    predictor_df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "date": [
                "2020-01-15",
                "2019-12-10",
                "2019-12-15",
                "2020-01-13",
                "2020-02-02",
                "2020-02-03",
            ],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )
    # Load a dataframe specifying when the outcome occurs
    outcome_df = pd.DataFrame({"id": [1], "date": ["2020-03-01"], "value": [1]})

    # Specify how to aggregate the predictors and define the outcome
    from timeseriesflattener.feature_spec_objects import PredictorSpec, OutcomeSpec
    from timeseriesflattener.resolve_multiple_functions import mean, maximum

    predictor_spec = PredictorSpec(
        values_df=predictor_df,
        lookbehind_days=15,
        fallback=np.nan,
        id_col_name="id",
        timestamp_col_name="date",
        resolve_multiple_fn=mean,
        feature_name="test_feature",
    )
    outcome_spec = OutcomeSpec(
        values_df=outcome_df,
        lookahead_days=31,
        fallback=0,
        id_col_name="id",
        timestamp_col_name="date",
        resolve_multiple_fn=maximum,
        feature_name="test_outcome",
        incident=False,
    )

    # Instantiate TimeseriesFlattener and add the specifications
    from timeseriesflattener import TimeseriesFlattener

    ts_flattener = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        id_col_name="id",
        timestamp_col_name="date",
        n_workers=1,
        drop_pred_times_with_insufficient_look_distance=True,
    )
    ts_flattener.add_spec([predictor_spec, outcome_spec])
    ts_flattener.compute()
    ts_flattener.get_df()

    ## Add markdown table of output (when bugs are fixed)
