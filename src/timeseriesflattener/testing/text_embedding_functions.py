"""Functions for embedding text data into a vector space. These functions are used
in the tests. You can use them for inspiration to create your own embedding functions."""
import pickle as pkl
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from timeseriesflattener.utils import PROJECT_ROOT


def _load_bow_model() -> CountVectorizer:
    """Loads the bag-of-words model from a pickle file"""
    filename = PROJECT_ROOT / "tests" / "test_data" / "models" / "synth_bow_model.pkl"

    with Path(filename).open("rb") as f:
        return pkl.load(f)


def _load_pca_model() -> PCA:
    """Loads the PCA model from a pickle file"""
    filename = PROJECT_ROOT / "tests" / "test_data" / "models" / "synth_pca_model.pkl"

    with Path(filename).open("rb") as f:
        return pkl.load(f)


def bow_test_embedding(text_series: Series) -> DataFrame:
    """Embeds the text data using a bag-of-words model"""
    model = _load_bow_model()
    return pd.DataFrame(
        model.transform(text_series).toarray(),
        columns=model.get_feature_names_out(),
    )


def pca_test_embedding(text_series: Series) -> DataFrame:
    """Embeds the text data using a PCA model"""
    model = _load_pca_model()
    return pd.DataFrame(model.transform(text_series), columns=["pca_1", "pca_2"])
