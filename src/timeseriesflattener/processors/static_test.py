from __future__ import annotations

from timeseriesflattener.testing.utils_for_testing import str_to_pl_df

from ..feature_specs.prediction_times import PredictionTimeFrame
from ..feature_specs.static import StaticFrame, StaticSpec
from ..process_spec import process_spec
from ..test_flattener import assert_frame_equal


def test_process_static_spec():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp
        1,2021-01-01
        1,2021-01-01
        2,2021-01-01"""
    )

    value_frame = str_to_pl_df(
        """entity_id,value
        1,a
        2,b"""
    )

    result = process_spec(
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame),
        spec=StaticSpec(
            value_frame=StaticFrame(init_df=value_frame), column_prefix="pred", fallback=0
        ),
    )

    expected = str_to_pl_df(
        """prediction_time_uuid,pred_value_fallback_0
1-2021-01-01 00:00:00.000000,a
1-2021-01-01 00:00:00.000000,a
2-2021-01-01 00:00:00.000000,b
       """
    )

    assert_frame_equal(result.collect(), expected)


def test_process_static_spec_multiple_values():
    pred_frame = str_to_pl_df(
        """entity_id,pred_timestamp
        1,2021-01-01
        1,2021-01-01
        2,2021-01-01"""
    )
    value_frame = str_to_pl_df(
        """entity_id,value_1,value_2
        1,a,b
        2,c,d"""
    )
    result = process_spec(
        predictiontime_frame=PredictionTimeFrame(init_df=pred_frame),
        spec=StaticSpec(
            value_frame=StaticFrame(init_df=value_frame), column_prefix="pred", fallback=0
        ),
    )
    expected = str_to_pl_df(
        """prediction_time_uuid,pred_value_1_fallback_0,pred_value_2_fallback_0
1-2021-01-01 00:00:00.000000,a,b
1-2021-01-01 00:00:00.000000,a,b
2-2021-01-01 00:00:00.000000,c,d
       """
    )
    assert_frame_equal(result.collect(), expected)
