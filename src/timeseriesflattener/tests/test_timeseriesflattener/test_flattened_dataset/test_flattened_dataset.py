"""Larger tests for the `flattened_dataset.py` module."""
import numpy as np
import pandas as pd
import pytest

from timeseriesflattener.aggregation_fns import latest, mean
from timeseriesflattener.feature_specs.single_specs import OutcomeSpec, PredictorSpec, StaticSpec
from timeseriesflattener.flattened_dataset import TimeseriesFlattener


def test_add_spec(synth_prediction_times: pd.DataFrame, synth_outcome: pd.DataFrame):
    # Create an instance of the class that contains the `add_spec` method
    dataset = TimeseriesFlattener(
        prediction_times_df=synth_prediction_times,
        drop_pred_times_with_insufficient_look_distance=False,
    )

    # Create sample specs
    outcome_spec = OutcomeSpec(
        timeseries_df=synth_outcome,
        feature_base_name="outcome",
        lookahead_days=1,
        aggregation_fn=mean,
        fallback=0,
        incident=False,
    )
    predictor_spec = PredictorSpec(
        timeseries_df=synth_outcome,
        feature_base_name="predictor",
        lookbehind_days=1,
        aggregation_fn=mean,
        fallback=np.nan,
    )
    static_spec = StaticSpec(timeseries_df=synth_outcome, feature_base_name="static", prefix="pred")

    # Test adding a single spec
    dataset.add_spec(outcome_spec)
    assert dataset.unprocessed_specs.outcome_specs == [outcome_spec]

    # Test adding multiple specs
    dataset.add_spec([predictor_spec, static_spec])
    assert dataset.unprocessed_specs.predictor_specs == [predictor_spec]
    assert dataset.unprocessed_specs.static_specs == [static_spec]

    # Test adding an invalid spec type
    with pytest.raises(ValueError):  # noqa
        dataset.add_spec("invalid spec")  # type: ignore


def test_compute_specs(synth_prediction_times: pd.DataFrame, synth_outcome: pd.DataFrame):
    # Create an instance of the class that contains the `add_spec` method
    dataset = TimeseriesFlattener(
        prediction_times_df=synth_prediction_times,
        drop_pred_times_with_insufficient_look_distance=False,
    )

    # Create sample specs
    outcome_spec = OutcomeSpec(
        timeseries_df=synth_outcome,
        feature_base_name="outcome",
        lookahead_days=1,
        aggregation_fn=mean,
        fallback=0,
        incident=False,
    )
    predictor_spec = PredictorSpec(
        timeseries_df=synth_outcome,
        feature_base_name="predictor",
        lookbehind_days=1,
        aggregation_fn=mean,
        fallback=np.nan,
    )
    static_spec = StaticSpec(
        timeseries_df=synth_outcome[["value", "entity_id"]],
        feature_base_name="static",
        prefix="pred",
    )

    # Test adding a single spec
    dataset.add_spec([outcome_spec, predictor_spec, static_spec])

    df = dataset.get_df()

    assert isinstance(df, pd.DataFrame)


def test_drop_pred_time_if_insufficient_look_distance():
    # Create a sample DataFrame with some test data
    # Uses datetime to also test that using another column name works
    pred_time_df = pd.DataFrame(
        {
            "entity_id": [1, 1, 1, 1],
            "datetime": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        }
    )

    ts_flattener = TimeseriesFlattener(
        prediction_times_df=pred_time_df,
        drop_pred_times_with_insufficient_look_distance=True,
        timestamp_col_name="datetime",
    )

    pred_val_df = pd.DataFrame({"entity_id": [1], "datetime": ["2022-01-01"], "value": [1]})

    # Create a sample set of specs
    predictor_spec = PredictorSpec(
        timeseries_df=pred_val_df,
        lookbehind_days=1,
        aggregation_fn=latest,
        fallback=np.nan,
        feature_base_name="test_feature",
    )

    out_val_df = pd.DataFrame({"entity_id": [1], "datetime": ["2022-01-05"], "value": [4]})

    outcome_spec = OutcomeSpec(
        timeseries_df=out_val_df,
        lookahead_days=2,
        aggregation_fn=latest,
        fallback=np.nan,
        feature_base_name="test_feature",
        incident=False,
    )

    ts_flattener.add_spec(spec=[predictor_spec, outcome_spec])

    out_df = ts_flattener.get_df()

    # Assert that the correct rows were dropped from the DataFrame
    expected_df = pd.DataFrame({"datetime": ["2022-01-02", "2022-01-03"]})
    # Convert to datetime to avoid a warning
    expected_df = expected_df.astype({"datetime": "datetime64[ns]"})
    pd.testing.assert_series_equal(out_df["datetime"], expected_df["datetime"])


def test_double_compute_doesn_not_duplicate_columns():
    # Load a dataframe with times you wish to make a prediction
    prediction_times_df = pd.DataFrame(
        {
            "entity_id": [1, 1, 2, 2],
            "date": ["2020-01-01", "2020-02-01", "2020-02-01", "2020-03-01"],
        }
    )
    # Load a dataframe with raw values you wish to aggregate as predictors
    predictor_df = pd.DataFrame(
        {
            "entity_id": [1, 1, 1, 1, 2, 2, 2],
            "date": [
                "2020-01-15",
                "2019-12-10",
                "2019-12-15",
                "2019-10-20",
                "2020-01-13",
                "2020-02-02",
                "2020-03-16",
            ],
            "value": [1, 2, 3, 4, 4, 5, 6],
        }
    )

    predictor_spec = PredictorSpec(
        timeseries_df=predictor_df,
        lookbehind_days=15,
        fallback=np.nan,
        aggregation_fn=mean,
        feature_base_name="test_feature",
    )

    ts_flattener = TimeseriesFlattener(
        prediction_times_df=prediction_times_df,
        entity_id_col_name="entity_id",
        timestamp_col_name="date",
        n_workers=1,
        drop_pred_times_with_insufficient_look_distance=True,
    )
    ts_flattener.add_spec([predictor_spec])
    df = ts_flattener.get_df()
    df = ts_flattener.get_df()

    assert df.shape[0] == 4
