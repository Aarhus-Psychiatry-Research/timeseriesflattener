"""Example of how to get tfidf vocab."""

from psycop_feature_generation.utils import FEATURIZERS_PATH

# pylint: disable=missing-function-docstring


def get_tfidf_vocab(
    n_features: int,
) -> list[str]:  # pylint: disable=missing-function-docstring
    with open(  # pylint: disable=unspecified-encoding
        FEATURIZERS_PATH / f"tfidf_{str(n_features)}.txt",
        "r",
    ) as f:
        return f.read().splitlines()


TFIDF_VOCAB = {n: get_tfidf_vocab(n) for n in [100, 500, 1000]}
