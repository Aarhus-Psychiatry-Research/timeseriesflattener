import pytest

from timeseriesflattener.testing.text_embedding_functions import bow_test_embedding
from timeseriesflattener.testing.utils_for_testing import synth_text_data


def test_embedding_fn(synth_text_data):
    """Test that the embedding function works as expected"""
    df = synth_text_data.dropna(subset="text")
    embedding_fn = bow_test_embedding
    embedding_df = embedding_fn(df["text"])
    assert embedding_df.shape == (df.shape[0], 10)
