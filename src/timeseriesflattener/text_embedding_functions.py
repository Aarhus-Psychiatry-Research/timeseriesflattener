import pandas as pd
import torch
from pandas import Series
from sentence_transformers import SentenceTransformer
from transformers import pipeline


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
