"""Tests for generating tf-idf vocabulary."""

from pathlib import Path

import dill as pkl

_test_tf_idf_vocab = [
    "a",
    "and",
    "for",
    "in",
    "of",
    "or",
    "patient",
    "that",
    "to",
    "was",
]
TEST_TFIDF_VOCAB = ["tfidf-" + word for word in _test_tf_idf_vocab]

if __name__ == "__main__":

    p = Path("tests") / "test_data" / "test_tfidf"

    with open(p / "tfidf_10.pkl", "rb") as f:
        tfidf = pkl.load(f)

    print(tfidf.get_feature_names())
