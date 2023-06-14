import numpy as np
from timeseriesflattener.feature_specs.base_group_spec import NamedDataframe
from timeseriesflattener.feature_specs.group_specs import PredictorGroupSpec
from timeseriesflattener.feature_specs.single_specs import PredictorSpec
from timeseriesflattener.aggregation_functions import mean
from timeseriesflattener.testing.utils_for_testing import str_to_df


def test_combinations_from_group():
    test_df = str_to_df("""col""")

    test_class = PredictorGroupSpec(
        named_dataframes=[NamedDataframe(df=test_df, name="col")],
        prefix="pred",
        lookbehind_days=[1, 2, 3],
        aggregation_fns=[mean],
        fallback=[np.NaN],
    )

    output = test_class.create_combinations()

    first_spec = output[0]
    expected_spec = PredictorSpec(
        base_values_df=test_df,
        prefix="pred",
        feature_base_name="col",
        aggregation_fn=mean,
        fallback="nan",
        lookbehind_days=1.0,
    )

    assert first_spec.prefix == expected_spec.prefix
    assert first_spec.feature_base_name == expected_spec.feature_base_name
    assert first_spec.aggregation_fn == expected_spec.aggregation_fn
    assert first_spec.fallback == expected_spec.fallback
    assert first_spec.lookbehind_days == expected_spec.lookbehind_days
