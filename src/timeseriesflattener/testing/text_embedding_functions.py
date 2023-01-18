"""Functions for embedding text data into a vector space. These functions are used
in the tests. You can use them for inspiration to create your own embedding functions."""
import pickle as pkl

import catalogue
import pandas as pd
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import CountVectorizer

from timeseriesflattener.utils import PROJECT_ROOT

text_embedding_fns = catalogue.create("timeseriesflattener", "text_embedding_functions")


def _load_bow_model() -> CountVectorizer:
    """Loads the bag-of-words model from a pickle file"""
    filename = PROJECT_ROOT / "tests" / "test_data" / "models" / "synth_bow_model.pkl"

    with open(filename, "rb") as f:
        return pkl.load(f)


@text_embedding_fns.register("test_bow")
def bow_embedding(text_series: Series) -> DataFrame:
    """Embeds the text data using a bag-of-words model"""
    model = _load_bow_model()
    return pd.DataFrame(model.transform(text_series).toarray(), columns=model.get_feature_names())
