"""Test that feature spec objects work as intended."""
import numpy as np
import pandas as pd
import pytest

from timeseriesflattener.feature_spec_objects import (
    AnySpec,
    PredictorGroupSpec,
    check_that_col_names_in_kwargs_exist_in_df,
)
from timeseriesflattener.resolve_multiple_functions import maximum
from timeseriesflattener.testing.load_synth_data import (  # pylint: disable=unused-import; noqa
    load_synth_predictor_float,
)
from timeseriesflattener.testing.utils_for_testing import long_df_with_multiple_values
from timeseriesflattener.utils import data_loaders, split_df_and_register_to_dict


def test_anyspec_init():
    """Test that AnySpec initialises correctly."""
    values_loader_name = "synth_predictor_float"

    spec = AnySpec(
        values_loader=values_loader_name,
        prefix="test",
    )

    assert isinstance(spec.values_df, pd.DataFrame)
    assert spec.feature_name == values_loader_name


def test_loader_kwargs():
    """Test that loader kwargs are passed correctly."""
    spec = AnySpec(
        values_loader="synth_predictor_float",
        prefix="test",
        loader_kwargs={"n_rows": 10},
    )

    assert len(spec.values_df) == 10


def test_invalid_multiple_data_args():
    """Test that error is raised if multiple data args are passed."""

    with pytest.raises(ValueError, match=r".*nly one of.*"):
        AnySpec(
            values_loader="synth_predictor_float",
            values_name="synth_data",
            prefix="test",
        )


def test_anyspec_incorrect_values_loader_str():
    """Raise error if values loader is not a key in registry."""
    with pytest.raises(ValueError, match=r".*in registry.*"):
        AnySpec(values_loader="I don't exist", prefix="test")


def test_that_col_names_in_kwargs_exist_in_df():
    """Raise error if col name specified which is not in df."""
    # Create a sample dataframe
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

    # Test valid column names
    data = {"col_name_1": "A", "col_name_2": "B", "values_df": df}
    check_that_col_names_in_kwargs_exist_in_df(data=data, df=df)

    # Test invalid column names
    data = {"col_name_1": "A", "col_name_2": "D", "values_df": df}
    with pytest.raises(ValueError, match="D is not in df"):
        check_that_col_names_in_kwargs_exist_in_df(data=data, df=df)


def test_create_combinations_while_resolving_from_registry(
    long_df_with_multiple_values: pd.DataFrame,
):
    """Test that split_df_and_register_to_dict resolves correctly when multiple dataframes are fetched."""

    split_df_and_register_to_dict(df=long_df_with_multiple_values)

    group_spec = PredictorGroupSpec(
        values_name=[
            "value_name_1",
            "value_name_2",
        ],
        resolve_multiple_fn=["mean"],
        lookbehind_days=[30],
        fallback=[0],
    ).create_combinations()

    assert len(group_spec) == 2


def test_skip_all_if_no_need_to_process():
    """Test that no combinations are created if no need to process."""
    assert (
        len(
            PredictorGroupSpec(
                values_loader=["synth_predictor_float"],
                input_col_name_override="value",
                lookbehind_days=[1],
                resolve_multiple_fn=["max"],
                fallback=[0],
                allowed_nan_value_prop=[0.5],
            ).create_combinations(),
        )
        == 1
    )


def test_skip_one_if_no_need_to_process():
    """Test that one combination is skipped if no need to process."""
    created_combinations = PredictorGroupSpec(
        values_loader=["synth_predictor_float"],
        input_col_name_override="value",
        lookbehind_days=[1, 2],
        resolve_multiple_fn=["max", "min"],
        fallback=[0],
        allowed_nan_value_prop=[0],
    ).create_combinations()

    assert len(created_combinations) == 4


def test_resolve_multiple_fn_to_str():
    """Test that resolve_multiple_fn is converted to str correctly."""
    pred_spec_batch = PredictorGroupSpec(
        values_loader=["synth_predictor_float"],
        lookbehind_days=[365, 730],
        fallback=[np.nan],
        resolve_multiple_fn=[maximum],
    ).create_combinations()

    assert "maximum" in pred_spec_batch[0].get_col_str()
