"""Script for creating a bag-of-words model and a PCA model on the synthetic data.
Used for testing purposes."""
import pickle as pkl
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from timeseriesflattener.testing.load_synth_data import load_synth_text


def train_bow_model(corpus: Sequence[str]) -> CountVectorizer:
    """
    Trains a bag-of-words model on the synthetic data"

    Args:
        corpus (Sequence[str]): The corpus to train on
    """
    model = CountVectorizer(lowercase=True, max_features=10)
    model.fit(corpus)
    return model


def load_synth_txt_data() -> List[str]:
    """
    Loads the synthetic text data and returns the text
    """
    df = load_synth_text()
    return df["text"].dropna().tolist()


def train_pca_model(embedding: np.ndarray) -> PCA:
    """
    Trains a PCA model on the synthetic data

    Args:
        embedding: The embedding to train on
    """
    model = PCA(n_components=2)
    model.fit(embedding)
    return model


def save_model_to_test_dir(
    model: Any,
    filename: str,
):  # pylint: disable=missing-type-doc
    """
    Saves the model to a pickle file

    Args:
        model: The model to save
        filename: The filename to save the model to
    """
    project_root = Path(__file__).resolve().parents[3]
    filename = project_root / "tests" / "test_data" / "models" / filename  # type: ignore

    with Path(filename).open("wb") as f:
        pkl.dump(model, f)


if __name__ == "__main__":
    corpus = load_synth_txt_data()
    bow_model = train_bow_model(corpus)
    embedding = bow_model.transform(corpus)
    pca_model = train_pca_model(embedding.toarray())

    save_model_to_test_dir(bow_model, "synth_bow_model.pkl")
    save_model_to_test_dir(pca_model, "synth_pca_model.pkl")
