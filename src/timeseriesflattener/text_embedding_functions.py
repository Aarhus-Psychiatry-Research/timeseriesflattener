from collections.abc import Callable
from typing import Optional

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def embed_text_column(
    df: pd.DataFrame,
    text_col_name: str,
    embedding_fn: Callable,
) -> pd.DataFrame:
    """Embeds text values using the embedding_fn and optionally reduces the
    dimensionality using dim_reduction_fn. Stores the embedding as a multi-
    index column called 'value'.

    Args:
        df (pd.DataFrame): Dataframe with text column to be embedded.
        text_col_name (str): Name of the text column to be embedded.
        embedding_fn (Callable): Function that takes a pd.Series of text and
            returns a pd.DataFrame of embeddings.
        
    Returns:
        pd.DataFrame: Dataframe with the text column replaced by the embedding.
    """
    embedding = embedding_fn(df[text_col_name])

    df = df.drop(text_col_name, axis=1)
    # make multiindex with embedding as 'value'
    df = pd.concat([df, embedding], axis=1, keys=["df", "value"])
    return df


def huggingface_embedding(text_series: pd.Series, model_name: str) -> pd.DataFrame:
    """Embeds the text data using a huggingface model. To use this
    in timeseriesflattener, you need to write a wrapper with your desired model name."""
    extractor = pipeline(model=model_name, task="feature-extraction")
    embeddings = extractor(text_series.to_list(), return_tensors=True)
    embeddings = [torch.mean(embedding, dim=1).squeeze() for embedding in embeddings]
    return pd.DataFrame(embeddings).astype(float)


def sentence_transformers_embedding(
    text_series: pd.Series, model_name: str
) -> pd.DataFrame:
    """Embeds the text data using a sentence-transformers model. To use this
    in timeseriesflattener, you need to write a wrapper with your desired model name."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_series.to_list())
    return pd.DataFrame(embeddings)
