"""Tests for generating huggingface embeddings."""

from pathlib import Path

_test_hf_embeddings = [i for i in range(384)]


TEST_HF_EMBEDDINGS = [
    "embedding-" + str(dimension) for dimension in _test_hf_embeddings
]
