"""Test that feature spec objects work as intended."""

from typing import List

import numpy as np
import pandas as pd
import pytest
from timeseriesflattener.feature_spec_objects import (
    BaseModel,
    OutcomeGroupSpec,
    OutcomeSpec,
    PredictorGroupSpec,
    PredictorSpec,
    TemporalSpec,
    TextPredictorSpec,
    _AnySpec,
    check_that_col_names_in_kwargs_exist_in_df,
    generate_docstring_from_attributes,
)
from timeseriesflattener.resolve_multiple_functions import maximum
from timeseriesflattener.testing.load_synth_data import (  # pylint: disable=unused-import; noqa
    load_synth_predictor_float,
    synth_predictor_binary,
)
from timeseriesflattener.utils import split_df_and_register_to_dict


def test_anyspec_init():
    """Test that AnySpec initialises correctly."""
    values_loader_name = "synth_predictor_float"

    spec = _AnySpec(
        values_loader=values_loader_name,
        prefix="test",
    )

    assert isinstance(spec.values_df, pd.DataFrame)
    assert spec.feature_name == values_loader_name


def test_loader_kwargs():
    """Test that loader kwargs are passed correctly."""
    spec = _AnySpec(
        values_loader="synth_predictor_float",
        prefix="test",
        loader_kwargs={"n_rows": 10},
    )

    assert len(spec.values_df) == 10


def test_invalid_multiple_data_args():
    """Test that error is raised if multiple data args are passed."""

    with pytest.raises(ValueError, match=r".*nly one of.*"):
        _AnySpec(
            values_loader="synth_predictor_float",
            values_name="synth_data",
            prefix="test",
        )


def test_anyspec_incorrect_values_loader_str():
    """Raise error if values loader is not a key in registry."""
    with pytest.raises(ValueError, match=r".*in registry.*"):
        _AnySpec(values_loader="I don't exist", prefix="test")


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


def get_lines_with_diff(text1: str, text2: str) -> List[str]:
    """Find all lines in text1 which are different from text2."""
    # Remove whitespace and periods
    text_1 = text1.replace(" ", "").replace(".", "")
    text_2 = text2.replace(" ", "").replace(".", "")

    lines1 = text_1.splitlines()
    lines2 = text_2.splitlines()
    return [line for line in lines1 if line not in lines2]


@pytest.mark.parametrize(
    "spec",
    [
        _AnySpec,
        TemporalSpec,
        PredictorSpec,
        PredictorGroupSpec,
        TextPredictorSpec,
        OutcomeSpec,
        OutcomeGroupSpec,
    ],
)
def test_feature_spec_docstrings(spec: BaseModel):
    """Test that docstring is generated correctly."""
    current_docstring = spec.__doc__
    generated_docstring = generate_docstring_from_attributes(cls=spec)
    # strip docstrings of newlines and whitespace to allow room for formatting
    current_docstring_no_whitespace = (
        current_docstring.replace(" ", "")
        .replace(
            "\n",
            "",
        )
        .replace(".", "")
    )
    generated_docstring_no_whitespace = (
        generated_docstring.replace(" ", "")
        .replace(
            "\n",
            "",
        )
        .replace(".", "")
    )

    lines_with_diff = get_lines_with_diff(
        text1=current_docstring,
        text2=generated_docstring,
    )

    if current_docstring_no_whitespace != generated_docstring_no_whitespace:
        raise AssertionError(
            f"""{spec} docstring is not updated correctly.

        Docstrings are automatically generated from the attributes of the class using
        the `timeseriesflattener.feature_spec_objects.generate_docstring_from_attributes` function.

        To modify docstrings or field descriptions, please modify the `short_description`
        field in the `Doc` class of the relevant spec object or the `description` field of the
        relevant attribute.

        If you have modified the `Doc` class or the attributes of the spec,
        copy-paste the generated docstring below into the docstring of the class.

        Got: \n\n{current_docstring}.

        Expected: \n\n{generated_docstring}

        Differences are in lines: \n\n{lines_with_diff}
        """,
        )


def test_predictorgroupspec_combinations_loader_kwargs():
    """Test that loader kwargs are used correctly in PredictorGroupSpec combinations."""

    binary_100_rows = synth_predictor_binary(n_rows=100)
    float_100_rows = load_synth_predictor_float(n_rows=100)

    spec = PredictorGroupSpec(
        values_loader=("synth_predictor_binary", "synth_predictor_float"),
        loader_kwargs=[{"n_rows": 100}],
        prefix="test_",
        resolve_multiple_fn=["bool"],
        fallback=[0],
        lookbehind_days=[10],
    )

    combinations = spec.create_combinations()

    pd.testing.assert_frame_equal(binary_100_rows, combinations[0].values_df)
    pd.testing.assert_frame_equal(float_100_rows, combinations[1].values_df)
