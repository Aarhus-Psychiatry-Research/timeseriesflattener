import pickle as pkl
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from timeseriesflattener.testing.load_synth_data import load_synth_text


def train_bow_model(corpus: Sequence[str]):
    """
    Trains a bag-of-words model on the synthetic data"

    Args:
        corpus (Sequence[str]): The corpus to train on
    """
    model = CountVectorizer(lowercase=True, max_features=10)
    model.fit(corpus)
    return model


def load_synth_txt_data():
    """
    Loads the synthetic text data and returns the text
    """
    df = load_synth_text()
    return df["text"].dropna().tolist()


def save_bow_model(model, filename: str):
    """
    Saves the model to a pickle file

    Args:
        model: The model to save
        filename: The filename to save the model to
    """
    project_root = Path(__file__).resolve().parents[3]
    filename = project_root / "tests" / "test_data" / "models" / filename

    with open(filename, "wb") as f:
        pkl.dump(model, f)


if __name__ == "__main__":
    corpus = load_synth_txt_data()
    model = train_bow_model(corpus)
    save_bow_model(model, "synth_bow_model.pkl")
