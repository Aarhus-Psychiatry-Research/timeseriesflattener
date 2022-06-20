from pathlib import Path

import pandas as pd
import pytest
from psycopmlutils.model_performance import ModelPerformance
from sklearn.model_selection import PredefinedSplit


@pytest.fixture(scope="function")
def multiclass_df():
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "scores": [
                # id 1
                [0.8, 0.1, 0.05, 0.05],
                [0.4, 0.7, 0.1, 0.1],
                # id 2
                [0.1, 0.05, 0.8, 0.05],
                [0.1, 0.7, 0.1, 0.1],
                # id 3
                [0.1, 0.1, 0.7, 0.1],
                [0.2, 0.5, 0.2, 0.1],
                # id 4
                [0.1, 0.1, 0.2, 0.6],
                [0.1, 0.2, 0.1, 0.6],
            ],
            "label": ["ASD", "ASD", "DEPR", "DEPR", "TD", "TD", "SCHZ", "SCHZ"],
            "model_name": ["test"] * 8,
        }
    )


@pytest.fixture(scope="function")
def binary_df():
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "scores": [[0.8, 0.2], [0.5, 0.5], [0.6, 0.4], [0.9, 0.1]],
            "label": ["TD", "TD", "DEPR", "DEPR"],
            "optional_grouping1": ["grouping1"] * 4,
            "optional_grouping2": ["grouping2"] * 4,
        }
    )


@pytest.fixture(scope="function")
def binary_float_df():
    return pd.DataFrame({"scores": [0.6, 0.2, 0.8], "label": [1, 0, 0]})


@pytest.fixture(scope="function")
def multiclass_score_mapping():
    return {0: "ASD", 1: "DEPR", 2: "TD", 3: "SCHZ"}


@pytest.fixture(scope="function")
def binary_score_mapping():
    return {0: "TD", 1: "DEPR"}


def test_multiclass_transform_from_dataframe(multiclass_df, multiclass_score_mapping):

    res = ModelPerformance.performance_metrics_from_df(
        multiclass_df,
        id_col_name="id",
        metadata_col_names="model_name",
        prediction_col_name="scores",
        label_col_name="label",
        id2label=multiclass_score_mapping,
    )

    assert len(res["model_name"].unique()) == 1
    assert len(res["level"].unique()) == 2
    assert res.shape[0] == 40  # (3 metrics per class (4) + 7 overall) * 2


def test_binary_transform_from_dataframe(binary_df, binary_score_mapping):

    res = ModelPerformance.performance_metrics_from_df(
        df=binary_df,
        id_col_name="id",
        metadata_col_names="all",
        prediction_col_name="scores",
        label_col_name="label",
        id2label=binary_score_mapping,
    )

    assert (
        res[
            (res["class"] == "TD")
            & (res["level"] == "id")
            & (res["score_type"] == "recall")
        ]["value"].tolist()[0]
        == 1.0
    )


def test_binary_transform_from_dataframe_with_float(
    binary_float_df, binary_score_mapping
):

    res = ModelPerformance.performance_metrics_from_df(
        df=binary_float_df,
        metadata_col_names="all",
        prediction_col_name="scores",
        label_col_name="label",
    )

    assert res[res["score_type"] == "acc"]["value"].values[0] == pytest.approx(0.666667)


def test_binary_transform_from_dataframe_with_float_wide(binary_float_df):
    res = ModelPerformance.performance_metrics_from_df(
        binary_float_df,
        to_wide=True,
        prediction_col_name="scores",
        label_col_name="label",
    )
    assert res["acc-overall"][0] == pytest.approx(0.666667)
    assert res.shape[0] == 1


def test_transform_folder():
    folder = Path("tests") / "test_model_performance" / "test_data"
    metadata_cols = ["model_name", "split", "type", "binary"]

    dfs = []
    for diagnosis in ["DEPR", "ASD", "SCHZ", "multiclass"]:
        if diagnosis != "multiclass":
            score_mapping = {0: diagnosis, 1: "TD"}
        else:
            score_mapping = {0: "TD", 1: "DEPR", 2: "ASD", 3: "SCHZ"}

        df = ModelPerformance.performance_metrics_from_folder(
            folder,
            pattern=f"*{diagnosis}*.jsonl",
            prediction_col_name="scores",
            id2label=score_mapping,
            metadata_col_names=metadata_cols,
            label_col_name="label",
        )

        dfs.append(df)

    dfs = pd.concat(dfs)
    assert len(dfs.columns) == 9
    assert len(dfs["model_name"]) > 1
