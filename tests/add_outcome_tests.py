from turtle import st
from timeseriesflattener.flattened_dataset import *
import pandas as pd


def test_event_after_prediction_time():
    prediction_times_str = """dw_ek_borger,datotid_start,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,datotid_event,val,
                        1,2022-01-01 00:00:01, 1
                        """
    run_tests_from_strings(prediction_times_str, event_times_str, [1])


def test_event_before_prediction():
    prediction_times_str = """dw_ek_borger,datotid_start,
                            1,2021-12-31 00:00:00
                            """
    event_times_str = """dw_ek_borger,datotid_event,val,
                        1,2021-12-30 23:59:59, 1
                        """
    run_tests_from_strings(prediction_times_str, event_times_str, [0])


def test_multiple_citizens():
    prediction_times_str = """dw_ek_borger,datotid_start,
                            1,2021-12-31 00:00:00
                            1,2022-01-02 00:00:00
                            5,2025-01-02 00:00:00
                            5,2025-08-05 00:00:00
                            """
    event_times_str = """dw_ek_borger,datotid_event,val,
                        1,2021-12-31 00:00:01, 1
                        1,2023-01-02 00:00:00, 1
                        5,2025-01-03 00:00:00, 1
                        5,2022-01-05 00:00:01, 1
                        """
    run_tests_from_strings(prediction_times_str, event_times_str, [1, 0, 1, 0])


def str_to_df(str):
    from io import StringIO
    import pandas as pd

    df = pd.read_table(StringIO(str), sep=",", index_col=False)
    df.loc[:, df.columns.str.contains("datotid")] = df.loc[
        :, df.columns.str.contains("datotid")
    ].apply(pd.to_datetime)

    return df.loc[:, ~df.columns.str.contains("^Unnamed")]  # Drop "Unnamed" cols


def run_tests_from_strings(
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
    > prediction_times_str =  '''dw_ek_borger,datotid_start,
    >                         1,2021-12-31 00:00:00
    >                         '''
    >
    > event_times_str =  '''dw_ek_borger,datotid_event,val,
    >                    1,2022-01-01 00:00:01, 1
    >                    '''
    >
    > run_tests_from_strings(prediction_times_str, event_times_str, [1])
    """
    df_prediction_times = str_to_df(prediction_times)
    df_event_times = str_to_df(event_times)

    dataset = FlattenedDataset(
        prediction_times_df=df_prediction_times, prediction_time_colname="datotid_start"
    )

    dataset.add_outcome(
        outcome_df=df_event_times,
        lookahead_days=2,
        resolve_multiple="max",
        fallback=0,
        outcome_colname="val",
        timestamp_colname="datotid_event",
    )

    expected_values = pd.DataFrame({"val_within_2_days": expected_values})

    pd.testing.assert_series_equal(
        dataset.df["val_within_2_days"], expected_values["val_within_2_days"]
    )
