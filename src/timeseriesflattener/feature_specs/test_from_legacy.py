from __future__ import annotations

import pandas as pd
from timeseriesflattener.v1.aggregation_fns import (
    boolean,
    change_per_day,
    count,
    earliest,
    latest,
    maximum,
    mean,
    minimum,
    summed,
    variance,
)
from timeseriesflattener.v1.feature_specs.group_specs import NamedDataframe

from .from_legacy import PredictorGroupSpec
from .predictor import PredictorSpec


def test_create_predictorspec_from_legacy():
    legacy_spec = PredictorGroupSpec(
        lookbehind_days=((1, 2), (3, 4)),  # type: ignore # Hates literals
        named_dataframes=[
            NamedDataframe(
                df=pd.DataFrame({"timestamp": ["2013-01-01"], "dw_ek_borger": "1", "value": 1}),
                name="test",
            ),
            NamedDataframe(
                df=pd.DataFrame({"timestamp": ["2013-01-01"], "dw_ek_borger": "2", "value": 2}),
                name="test2",
            ),
        ],
        aggregation_fns=[
            latest,
            earliest,
            maximum,
            minimum,
            mean,
            summed,
            count,
            variance,
            boolean,
            change_per_day,
        ],
        fallback=[0],
    )

    new_specs = legacy_spec.create_combinations()

    assert isinstance(new_specs[0], PredictorSpec)
    assert len(new_specs) == 2
