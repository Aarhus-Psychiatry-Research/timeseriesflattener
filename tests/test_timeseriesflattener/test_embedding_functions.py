"""Tests for the text embedding functions"""
import pandas as pd
import pytest
from timeseriesflattener.testing.text_embedding_functions import bow_test_embedding
from timeseriesflattener.text_embedding_functions import (
    huggingface_embedding,
    sentence_transformers_embedding,
)


def test_embedding_fn(synth_text_data: pd.DataFrame):
    """Test that synth embedding function works as expected"""
    df = synth_text_data.dropna(subset="text")
    embedding_df = bow_test_embedding(df["text"])
    assert embedding_df.shape == (df.shape[0], 10)


@pytest.mark.huggingface()
def test_huggingface_embedding(synth_text_data: pd.DataFrame):
    """Test that the huggingface embedding function works as expected"""
    df = synth_text_data.dropna(subset="text")
    df = df.head(5)
    embedding_df = huggingface_embedding(
        df["text"],
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    assert embedding_df.shape == (df.shape[0], 384)


@pytest.mark.huggingface()
def test_sentence_transformer_embedding(synth_text_data: pd.DataFrame):
    """Test that the sentence-transformer embedding function works as expected"""
    df = synth_text_data.dropna(subset="text")
    df = df.head(5)
    embedding_df = sentence_transformers_embedding(
        df["text"],
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    assert embedding_df.shape == (df.shape[0], 384)
