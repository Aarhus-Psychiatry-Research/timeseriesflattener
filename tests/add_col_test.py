import os
print(os.getcwd())

from src.timeseriesflattener.flattened_time_series import add_col_event_within_months
import pandas as pd

def str_to_df(str):
    from io import StringIO
    import pandas as pd

    df = pd.read_table(StringIO(str), sep=",", index_col=False)
    df.loc[:, df.columns.str.contains("datotid")] = df.loc[
        :, df.columns.str.contains("datotid")
    ].apply(pd.to_datetime)
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]  # Drop "Unnamed" cols


prediction_times_str = """dw_ek_borger,datotid_start,
1,2022-01-01 00:00:00
1,2022-01-02 00:00:00
5,2025-01-05 00:00:00
5,2025-08-05 00:00:00
5,2030-01-02 00:00:00"""

df_prediction_times = str_to_df(prediction_times_str)

event_times_str = """dw_ek_borger,datotid_event,
1,2021-12-31 00:00:00
1,2023-01-02 00:00:00
5,2025-01-01 00:00:00
5,2025-01-02 00:00:00
"""

df_event_times = str_to_df(event_times_str)


def test_add_col_event():
    test_df = add_col_event_within_months(
        df_prediction_times, df_event_times, 180, "event"
    )

    expected_vals = pd.DataFrame(
        {"event_within_180_days": [True, True, True, False, False]}
    )

    pd.testing.assert_series_equal(
        test_df["event_within_180_days"], expected_vals["event_within_180_days"]
    )
