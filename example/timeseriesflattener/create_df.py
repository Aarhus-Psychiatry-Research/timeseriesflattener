import time

from wasabi import msg

from load_data import LoadData
from timeseriesflattener.create_feature_combinations import create_feature_combinations
from timeseriesflattener.flattened_dataset import FlattenedDataset

if __name__ == "__main__":
    PREDICTOR_LIST = create_feature_combinations(
        [
            {
                "predictor_df": "hba1c_vals",
                "lookbehind_days": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "resolve_multiple": "latest",
                "fallback": 35,
                "source_values_col_name": "val",
                "new_col_name": "hba1c",
            }
        ]
    )

    print(PREDICTOR_LIST)

    prediction_times = LoadData.physical_visits_to_psychiatry()
    # event_times = LoadData.event_times()
    hba1c_vals = LoadData.hba1c()

    msg.info("Initialising flattened dataset")
    flattened_df = FlattenedDataset(prediction_times_df=prediction_times, n_workers=32)

    # Predictors
    msg.info("Adding predictors")
    start_time = time.time()

    predictor_dfs = {"hba1c_vals": hba1c_vals}

    flattened_df.add_predictors_from_list_of_argument_dictionaries(
        predictor_list=PREDICTOR_LIST,
        predictor_dfs_dict=predictor_dfs,
    )

    end_time = time.time()
    msg.good(f"Finished adding predictors, took {end_time - start_time}")
