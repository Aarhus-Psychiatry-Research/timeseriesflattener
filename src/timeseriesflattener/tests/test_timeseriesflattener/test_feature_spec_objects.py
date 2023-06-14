"""Test that feature spec objects work as intended."""


from typing import List

import numpy as np
import pytest

from timeseriesflattener.aggregation_functions import maximum
from timeseriesflattener.feature_specs.base_group_spec import NamedDataframe
from timeseriesflattener.feature_specs.group_specs import (
    OutcomeGroupSpec,
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
    OutcomeSpec,
    PredictorSpec,
    TemporalSpec,
    TextPredictorSpec,
)
from timeseriesflattener.feature_specs.utils.generate_docstring_from_attributes import (
    generate_docstring_from_attributes,
)
from timeseriesflattener.utils.pydantic_basemodel import BaseModel


def test_skip_all_if_no_need_to_process(empty_named_df: NamedDataframe):
    """Test that no combinations are created if no need to process."""
    assert (
        len(
            PredictorGroupSpec(
                named_dataframes=[empty_named_df],
                lookbehind_days=[1],
                aggregation_fns=maximum,
                fallback=[0],
            ).create_combinations(),
        )
        == 1
    )


def test_skip_one_if_no_need_to_process(empty_named_df: NamedDataframe):
    """Test that one combination is skipped if no need to process."""
    created_combinations = PredictorGroupSpec(
        named_dataframes=[empty_named_df],
        lookbehind_days=[1, 2],
        aggregation_fns=maximum,
        fallback=[0],
    ).create_combinations()

    assert len(created_combinations) == 4


def test_resolve_multiple_fn_to_str(empty_named_df: NamedDataframe):
    """Test that resolve_multiple_fn is converted to str correctly."""
    pred_spec_batch = PredictorGroupSpec(
        named_dataframes=[empty_named_df],
        lookbehind_days=[365, 730],
        fallback=[np.nan],
        aggregation_fns=maximum,
    ).create_combinations()

    assert "maximum" in pred_spec_batch[0].get_output_col_name()


def test_lookbehind_days_handles_floats(empty_named_df: NamedDataframe):
    """Test that lookbheind days does not coerce floats into ints."""
    pred_spec_batch = PredictorGroupSpec(
        named_dataframes=[empty_named_df],
        lookbehind_days=[2, 0.5],
        fallback=[np.nan],
        aggregation_fns=maximum,
    ).create_combinations()

    assert pred_spec_batch[1].lookbehind_days == 0.5


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
        AnySpec,
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
    current_docstring: str = spec.__doc__  # type: ignore
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
