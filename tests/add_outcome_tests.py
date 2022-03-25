from timeseriesflattener.flattened_dataset import *
import pandas as pd

# Currently only "integration tests" (correct term?)
# Will expand with proper unit tests if we agree on the API


def test_event_after_prediction_time():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2022-01-01 00:00:01, 1
                        """
    run_tests_from_df_strings(prediction_times_str, event_times_str, [1])


def test_event_before_prediction():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2021-12-30 23:59:59, 1
                        """
    run_tests_from_df_strings(prediction_times_str, event_times_str, [0])


def test_multiple_citizens():
    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            1,2022-01-02 00:00:00
                            5,2025-01-02 00:00:00
                            5,2025-08-05 00:00:00
                            """
    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2021-12-31 00:00:01, 1
                        1,2023-01-02 00:00:00, 1
                        5,2025-01-03 00:00:00, 1
                        5,2022-01-05 00:00:01, 1
                        """
    run_tests_from_df_strings(prediction_times_str, event_times_str, [1, 0, 1, 0])


def str_to_df(str):
    from io import StringIO

    df = pd.read_table(StringIO(str), sep=",", index_col=False)

    convert_cols_with_matching_colnames_to_datetime(df, "timestamp")

    # Drop "Unnamed" cols
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


def convert_cols_with_matching_colnames_to_datetime(df: DataFrame, colname_substr: str):
    """Convert columns that contain colname_substr in their name to datetimes

    Args:
        df (DataFrame)
        colname_substr (str): Substring to match on.
    """
    df.loc[:, df.columns.str.contains(colname_substr)] = df.loc[
        :, df.columns.str.contains(colname_substr)
    ].apply(pd.to_datetime)


def run_tests_from_df_strings(
    prediction_times: str,
    event_times: str,
    expected_values: List,  # Generalise and expand with more arguments as we expand functionality
):
    """Run tests from string representations of dataframes.

    Args:
        prediction_times (str): A string-representation of prediction-times
        event_times (str): A string-representation of event_times
        expected_values (List): The expected values for each prediction time

    Example:
    > prediction_times_str =  '''dw_ek_borger,timestamp,
    >                         1,2021-12-31 00:00:00
    >                         '''
    >
    > event_times_str =  '''dw_ek_borger,timestamp,val,
    >                    1,2022-01-01 00:00:01, 1
    >                    '''
    >
    > run_tests_from_strings(prediction_times_str, event_times_str, [1])
    """
    df_prediction_times = str_to_df(prediction_times)
    df_event_times = str_to_df(event_times)

    dataset = FlattenedDataset(
        prediction_times_df=df_prediction_times, prediction_time_colname="timestamp"
    )

    dataset.add_outcome(
        outcome_df=df_event_times,
        lookahead_days=2,
        resolve_multiple="max",
        fallback=0,
        values_colname="val",
        timestamp_colname="timestamp",
    )

    expected_values = pd.DataFrame({"val_within_2_days": expected_values})

    pd.testing.assert_series_equal(
        dataset.df["val_within_2_days"], expected_values["val_within_2_days"]
    )
