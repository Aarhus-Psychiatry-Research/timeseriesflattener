"""Functions for embedding text data"""
from collections.abc import Callable
from typing import Optional

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.base import TransformerMixin
from transformers import pipeline


def embed_text_column(
    df: pd.DataFrame,
    text_col_name: str,
    embedding_fn: Callable,
    embedding_fn_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Embeds text values using the embedding_fn and optionally reduces the
    dimensionality using dim_reduction_fn. Stores the embedding as a multi-
    index column called 'value'.

    Args:
        df (pd.DataFrame): Dataframe with text column to be embedded.
        text_col_name (str): Name of the text column to be embedded.
        embedding_fn (Callable): Function that takes a pd.Series of text and
            returns a pd.DataFrame of embeddings.
        embedding_fn_kwargs (Optional[dict], optional): Keyword arguments for
            embedding_fn. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with the text column replaced by the embedding.
    """
    if embedding_fn_kwargs:
        embedding = embedding_fn(df[text_col_name], **embedding_fn_kwargs)
    else:
        embedding = embedding_fn(df[text_col_name])

    df = df.drop(text_col_name, axis=1)
    # make multiindex with embedding as 'value'
    df = pd.concat([df, embedding], axis=1, keys=["df", "value"])
    return df


def huggingface_embedding(text_series: pd.Series, model_name: str) -> pd.DataFrame:
    """Embeds the text data using a huggingface model. To use this in timeseriesflattener,
    supply the model_name as an embedding_fn_kwargs argument to TextPredictorSpec.
    For example:
    `embedding_fn_kwargs={"model_name": "bert-base-uncased"}`

    Args:
        text_series (pd.Series): Series of text to be embedded.
        model_name (str): Name of the huggingface model to use.
    """
    extractor = pipeline(model=model_name, task="feature-extraction")
    embeddings = extractor(text_series.to_list(), return_tensors=True)
    embeddings = [torch.mean(embedding, dim=1).squeeze() for embedding in embeddings]
    return pd.DataFrame(embeddings).astype(float)


def sentence_transformers_embedding(
    text_series: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """Embeds the text data using a sentence-transformers model. To use this in
    timeseriesflattener, supply the model_name as an embedding_fn_kwargs argument
    to TextPredictorSpec. For example:
    `embedding_fn_kwargs={"model_name": "paraphrase-multilingual-MiniLM-L12-v2"}`

    Args:
        text_series (pd.Series): Series of text to be embedded.
        model_name (str): Name of the sentence-transformers model to use.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_series.to_list())
    return pd.DataFrame(embeddings)


def sklearn_embedding(
    text_series: pd.Series,
    model: TransformerMixin,
) -> pd.DataFrame:
    """Embeds text data using a sklearn model. The model should be trained
    before using this function and have a `get_feature_names` attribute.
    To use this in timeseriesflattener, supply the model as an embedding_fn_kwargs
    argument to TextPredictorSpec. For example:
    `embedding_fn_kwargs={"model": tf_idf_model}`

    Args:
        text_series (pd.Series): Series of text to be embedded.
        model (TransformerMixin): Trained sklearn model
    """
    return pd.DataFrame(
        model.transform(text_series).toarray(),
        columns=model.get_feature_names(),
    )
