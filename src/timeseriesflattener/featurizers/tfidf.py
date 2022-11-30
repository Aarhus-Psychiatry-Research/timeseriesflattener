"""Train a TF-IDF featurizer on train set of all clinical notes."""
from pathlib import Path

import dill as pkl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wasabi import Printer

from psycop_feature_generation.loaders.raw.load_ids import load_ids
from psycop_feature_generation.loaders.raw.load_text import load_all_notes
from psycop_feature_generation.utils import FEATURIZERS_PATH, PROJECT_ROOT


def create_tfidf_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    max_df: float = 0.95,
    min_df: float = 0.01,
    max_features: int = 100,
) -> TfidfVectorizer:
    """Creates a TF-IDF vectorizer with a whitespace tokenizer.

    Args:
        ngram_range (Tuple[int, int], optional): How many ngrams to make. Defaults to (1, 2).
        max_df (float, optional): Removes words occuring in max_df proportion of documents.
            Defaults to 0.95.
        min_df (float, optional): Removes words occuring in less than min_df proportion
            of documents. Defaults to 0.01.
        max_features (int, optional): How many features to create. Defaults to 100.

    Returns:
        TfidfVectorizer: Sklearn TF-IDF vectorizer
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        tokenizer=lambda x: x.split(" "),
        lowercase=True,
        max_df=max_df,  # remove very common words
        min_df=min_df,  # remove very rare words
        max_features=max_features,
    )


if __name__ == "__main__":

    SYNTHETIC = False
    OVERTACI = True

    msg = Printer(timestamp=True)

    # Train TF-IDF model on all notes
    if OVERTACI:
        if not FEATURIZERS_PATH.exists():
            FEATURIZERS_PATH.mkdir()

        text = load_all_notes(  # pylint: disable=undefined-variable
            featurizer=None,
            n_rows=None,
            featurizer_kwargs=None,
        )

        # Subset only train set
        train_ids = load_ids(split="train")
        train_ids = train_ids["dw_ek_borger"].unique()
        text = text[text["dw_ek_borger"].isin(train_ids)]
        text = text["text"].tolist()

        for n_features in [100, 500, 1000]:
            msg.info(f"Fitting tf-idf with {n_features} features..")
            vectorizer = create_tfidf_vectorizer(max_features=n_features)
            vectorizer.fit(text)

            with open(FEATURIZERS_PATH / f"tfidf_{n_features}.pkl", "wb") as f:
                pkl.dump(vectorizer, f)

            vocab = ["tfidf-" + word for word in vectorizer.get_feature_names()]
            with open(  # type: ignore # pylint: disable=unspecified-encoding
                FEATURIZERS_PATH / f"tfidf_{n_features}_vocab.txt",
                "w",
            ) as f:
                f.write("\n".join(vocab))  # type: ignore

    # train TF-IDF on synthetic data
    if SYNTHETIC:
        test_path = PROJECT_ROOT / "tests" / "test_data"
        save_dir = test_path / "test_tfidf"
        if not save_dir.exists():
            save_dir.mkdir()

        text = pd.read_csv(test_path / "synth_txt_data.csv")
        text = text["text"].dropna().tolist()

        msg.info("Fitting tf-idf with 10 features..")
        vectorizer = create_tfidf_vectorizer(max_features=10)
        vectorizer.fit(text)

        with open(save_dir / "tfidf_10.pkl", "wb") as f:
            pkl.dump(vectorizer, f)
