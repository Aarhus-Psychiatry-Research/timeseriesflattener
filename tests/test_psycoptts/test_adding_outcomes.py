from pandas import DataFrame
import pandas as pd
from pandas.testing import assert_frame_equal

from psycoptts.add_outcomes import add_outcome_from_df
from testing_utils import *


def test_adding_first_outcome():
    all_patients_str = """dw_ek_borger
                        1,
                        2,
                        3,
                        4,
                        """

    outcome_str = """dw_ek_borger
                        1,
                        2,
                        """

    all_patients_df = str_to_df(all_patients_str)
    outcome_df = str_to_df(outcome_str)
    out_df = add_outcome_from_df(all_patients_df, outcome_df, "outcome_name")

    expected_out_str = """dw_ek_borger,outcome_name,
                        1, 1.0, 
                        2, 1.0, 
                        3, 0.0, 
                        4, 0.0, 
                        """
    expected_values = str_to_df(expected_out_str)

    assert_frame_equal(out_df, expected_values)


def test_adding_second_outcome():
    all_patients_str = """dw_ek_borger
                        1,
                        2,
                        3,
                        4,
                        """

    outcome_1_str = """dw_ek_borger
                        1,
                        2,
                        """

    outcome_2_str = """dw_ek_borger
                        3,
                        4,
                        """

    all_patients_df = str_to_df(all_patients_str)
    outcome1_df = str_to_df(outcome_1_str)
    outcome2_df = str_to_df(outcome_2_str)

    out_df = add_outcome_from_df(all_patients_df, outcome1_df, "outcome1_name")
    out_df = add_outcome_from_df(out_df, outcome2_df, "outcome2_name")

    expected_out_str = """dw_ek_borger,outcome1_name,outcome2_name
                        1, 1.0, 0.0,
                        2, 1.0, 0.0,
                        3, 0.0, 1.0,
                        4, 0.0, 1.0
                        """
    expected_values = str_to_df(expected_out_str)

    assert_frame_equal(out_df, expected_values)
