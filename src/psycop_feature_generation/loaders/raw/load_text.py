# """Load text data from a database and featurise it using a tf-idf
# vectorizer."""

# # pylint: disable=E0211,E0213,missing-function-docstring

# from functools import partial
# from multiprocessing import Pool
# from pathlib import Path
# from typing import Optional, Union

# import dill as pkl
# import pandas as pd

# # import torch
# from transformers import AutoModel, AutoTokenizer
# from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

# from psycop_feature_generation.loaders.raw.sql_load import sql_load
# from psycop_feature_generation.utils import PROJECT_ROOT, data_loaders


# def get_all_valid_note_types() -> set[str]:
#     """Returns a set of valid note types. Notice that 'Konklusion' is replaced
#     by 'Vurdering/konklusion' in 2020, so make sure to use both. 'Ordination'
#     was replaced by 'Ordination, Psykiatry' in 2022, but 'Ordination,
#     Psykiatri' is not included in the table. Use with caution.

#     Returns:
#         Set[str]: Set of valid note types
#     """
#     return {
#         "Observation af patient, Psykiatri",
#         "Samtale med behandlingssigte",
#         "Ordination",  # OBS replaced "Ordination, Psykiatri" in 01/02-22
#         # but is not included in this table. Use with caution
#         "Aktuelt psykisk",
#         "Aktuelt socialt, Psykiatri",
#         "Aftaler, Psykiatri",
#         "Medicin",
#         "Aktuelt somatisk, Psykiatri",
#         "Objektivt psykisk",
#         "Kontaktårsag",
#         "Telefonkonsultation",
#         "Journalnotat",
#         "Telefonnotat",
#         "Objektivt, somatisk",
#         "Plan",
#         "Semistruktureret diagnostisk interview",
#         "Vurdering/konklusion",
#     }


# def _load_notes_for_year(
#     note_types: Union[str, list[str]],
#     year: str,
#     view: Optional[str] = "FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret",
#     n_rows: Optional[int] = None,
# ) -> pd.DataFrame:
#     """Loads clinical notes from sql from a specified year and matching
#     specified note types.

#     Args:
#         note_names (Union[str, list[str]]): Which types of notes to load.
#         year (str): Which year to load
#         view (str, optional): Which table to load.
#             Defaults to "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret".
#         n_rows (Optional[int], optional): Number of rows to load. Defaults to None.

#     Returns:
#         pd.DataFrame: Dataframe with clinical notes
#     """

#     sql = (
#         "SELECT dw_ek_borger, datotid_senest_aendret_i_sfien, fritekst"
#         + f" FROM [fct].[{view}_{year}_inkl_2021_feb2022]"
#         + f" WHERE overskrift IN {note_types}"
#     )
#     return sql_load(
#         sql,
#         database="USR_PS_FORSK",
#         chunksize=None,
#         n_rows=n_rows,
#     )


# def _tfidf_featurize(
#     df: pd.DataFrame,
#     tfidf_path: Path,
#     text_col: str = "text",
# ) -> pd.DataFrame:
#     """TF-IDF featurize text. Assumes `df` to have a column named `text`.

#     Args:
#         df (pd.DataFrame): Dataframe with text column
#         tfidf_path (Optional[Path]): Path to a sklearn tf-idf vectorizer
#         text_col (str, optional): Name of text column. Defaults to "text".

#     Returns:
#         pd.DataFrame: Original dataframe with tf-idf features appended
#     """
#     with open(tfidf_path, "rb") as f:
#         tfidf = pkl.load(f)

#     vocab = ["tfidf-" + word for word in tfidf.get_feature_names()]

#     text = df[text_col].values
#     df = df.drop(text_col, axis=1).reset_index(drop=True)

#     text = tfidf.transform(text)
#     text = pd.DataFrame(text.toarray(), columns=vocab)
#     return pd.concat([df, text], axis=1)


# def _mean_pooling(
#     model_output: BaseModelOutputWithPoolingAndCrossAttentions,
#     attention_mask: torch.Tensor,
# ) -> torch.Tensor:
#     """Mean Pooling - take attention mask into account for correct averaging.

#     Args:
#         model_output (BaseModelOutputWithPoolingAndCrossAttentions): model output from pretrained Huggingface transformer
#         attention_mask (torch.Tensor): attention mask from from pretrained Hugginface tokenizer

#     Returns:
#         np.ndarray: numpy array with mean pooled embeddings
#     """
#     token_embeddings = model_output[
#         0
#     ]  # first element of model_output contains all token embeddings
#     input_mask_expanded = (
#         attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     )
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#         input_mask_expanded.sum(1),
#         min=1e-9,
#     )


def _chunk_text(text: str, seq_length: int) -> list[str]:
    """Chunk text into sequences of length `seq_length`, where `seq_length`
    refers to number of words.

    Args:
        text (str): text to chunk
        seq_length (int): length of sequence (number of words)
    Returns:
        list[str]: list of text chunks
    """
    words = text.split(" ")
    # If text is not longer than allowed sequence length, extract and save embeddings
    if len(words) <= seq_length:
        return [text]
    # If text is longer than allowed sequence length, split text into chunks
    else:
        words_in_chunks = [
            words[i - seq_length : i]
            for i in range(seq_length, len(words) + seq_length, seq_length)
        ]
        chunks = [
            " ".join(word_list)
            for word_list in words_in_chunks
            if len(word_list) == seq_length
        ]  # drop small remainder of shorter size
        return chunks


# def _huggingface_featurize(
#     df: pd.DataFrame,
#     model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#     text_col: str = "text",
# ) -> pd.DataFrame:
#     """Featurize text using a huggingface model and generate a dataframe with
#     the embeddings. If the text is longer than the maximum sequence length of
#     the model, the text is split into chunks and embeddings are averaged across
#     chunks.

#     Args:
#         df (pd.DataFrame): Dataframe with text column
#         model_id (str): Which huggingface model to use. See https://huggingface.co/models for a list of models. Assumes the model is a transformer model and has both a tokenizer and a model.
#         text_col (str, optional): Name of text column. Defaults to "text".

#     Returns:
#         pd.DataFrame: Original dataframe with huggingface embeddings appended

#     Example:
#         >>> p = PROJECT_ROOT / "tests" / "test_data" / "raw"
#         >>> huggingface_model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#         >>> df_p = p / "synth_txt_data.csv"

#         >>> df = pd.read_csv(df_p)
#         >>> df = df.dropna()

#         >>> x = _huggingface_featurize(df, huggingface_model_id)
#     """
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModel.from_pretrained(model_id)

#     df = df[df[text_col].notna()]
#     text = df[text_col].values
#     df = df.drop(text_col, axis=1)

#     max_seq_length = int(
#         tokenizer.model_max_length / 1.5,
#     )  # allowing space for more word piece tokens than words in original sequence

#     list_of_embeddings = []
#     for txt in text:
#         chunks = _chunk_text(txt, max_seq_length)

#         encoded_input = tokenizer(
#             chunks,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#         )

#         with torch.no_grad():
#             model_output = model(**encoded_input)

#         embedding = _mean_pooling(model_output, encoded_input["attention_mask"])

#         if len(chunks) > 1:
#             list_of_embeddings.append(torch.mean(embedding, axis=0).numpy())  # type: ignore
#         else:
#             list_of_embeddings.append(embedding.numpy()[0])

#     embeddings_df = pd.DataFrame(list_of_embeddings)
#     embeddings_df.columns = [
#         "embedding-" + str(dimension) for dimension in range(embeddings_df.shape[1])
#     ]

#     return pd.concat([df, embeddings_df], axis=1)


# def _load_and_featurize_notes_per_year(
#     year: str,
#     note_types: Union[str, list[str]],
#     view: str,
#     n_rows: int,
#     featurizer: str,
#     featurizer_kwargs: dict,
# ) -> pd.DataFrame:
#     """Loads clinical notes and features them.

#     Args:
#         note_types (Union[str, list[str]]): Which note types to load.
#         year (str): Which year to load
#         view (str): Which view to load
#         n_rows (int): How many rows to load
#         featurizer (str): Which featurizer to use (tfidf or huggingface)
#         featurizer_kwargs (dict): kwargs for the featurizer

#     Returns:
#         pd.DataFrame: Dataframe of notes and features
#     """

#     df = _load_notes_for_year(
#         note_types=note_types,
#         year=year,
#         view=view,
#         n_rows=n_rows,
#     )
#     if featurizer == "tfidf":
#         df = _tfidf_featurize(df, **featurizer_kwargs)
#     elif featurizer == "huggingface":
#         df = _huggingface_featurize(df, **featurizer_kwargs)
#     return df


# def load_and_featurize_notes(
#     note_types: Union[str, list[str]],
#     featurizer: str,
#     featurizer_kwargs: Optional[dict] = None,
#     n_rows: Optional[int] = None,
# ) -> pd.DataFrame:
#     """Loads all clinical notes that match the specified note from all years.
#     Featurizes the notes using the specified featurizer (tf-idf or huggingface
#     model). Kwargs passed to.

#     Args:
#         note_types (Union[str, list[str]]): Which note types to load. See
#             `get_all_valid_note_types()` for valid note types.
#         featurizer (str): Which featurizer to use. Either 'tf-idf' or 'huggingface' or
#             `None` to return the raw text.
#         featurizer_kwargs (Optional[dict]): Kwargs passed to the featurizer. Defaults to None.
#             For tf-idf, this is `tfidf_path` to the vectorizer. For huggingface,
#             this is `model_id` to the model.
#         n_rows (Optional[int], optional): How many rows to load. Defaults to None.

#     Raises:
#         ValueError: If given invalid featurizer
#         ValueError: If given invlaid note type

#     Returns:
#         pd.DataFrame: Featurized clinical notes
#     """

#     valid_featurizers = {"tfidf", "huggingface", None}
#     if featurizer not in valid_featurizers:
#         raise ValueError(
#             f"featurizer must be one of {valid_featurizers}, got {featurizer}",
#         )

#     if isinstance(note_types, str):
#         note_types = list(note_types)  # pylint: disable=W0642
#     # check for invalid note types
#     if not set(note_types).issubset(get_all_valid_note_types()):
#         raise ValueError(
#             "Invalid note type. Valid note types are: "
#             + str(get_all_valid_note_types()),
#         )

#     # convert note_types to sql query
#     note_types = "('" + "', '".join(note_types) + "')"  # pylint: disable=W0642

#     view = "FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret"

#     load_and_featurize = partial(
#         _load_and_featurize_notes_per_year,
#         note_types=note_types,
#         view=view,
#         n_rows=n_rows,
#         featurizer=featurizer,
#         featurizer_kwargs=featurizer_kwargs,
#     )

#     years = list(range(2011, 2022))

#     with Pool(processes=len(years)) as p:
#         dfs = p.map(load_and_featurize, [str(y) for y in years])

#     df = pd.concat(dfs)

#     df = df.rename(
#         {"datotid_senest_aendret_i_sfien": "timestamp", "fritekst": "text"},
#         axis=1,
#     )
#     return df


# @data_loaders.register("all_notes")
# def load_all_notes(
#     featurizer: str,
#     n_rows: Optional[int] = None,
#     featurizer_kwargs: Optional[dict] = None,
# ) -> pd.DataFrame:
#     """Returns all notes from all years. Featurizes the notes using the
#     specified featurizer ('tfidf', 'huggingface', or `None` for raw text).
#     `featurizer_kwargs` are passed to the featurizer (e.g. "tfidf_path" for
#     tfidf, and "model_id" for huggingface).

#     Args:
#         featurizer (str): Which featurizer to use. Either 'tf-idf', 'huggingface', or None
#         n_rows (Optional[int], optional): Number of rows to load. Defaults to None.
#         featurizer_kwargs (Optional[dict], optional): Keyword arguments passed to
#             the featurizer. Defaults to None.

#     Returns:
#         pd.DataFrame: (Featurized) notes
#     """
#     return load_and_featurize_notes(
#         note_types=get_all_valid_note_types(),
#         featurizer=featurizer,
#         n_rows=n_rows,
#         featurizer_kwargs=featurizer_kwargs,
#     )


# @data_loaders.register("aktuelt_psykisk")
# def load_aktuel_psykisk(
#     featurizer: str,
#     n_rows: Optional[int] = None,
#     featurizer_kwargs: Optional[dict] = None,
# ) -> pd.DataFrame:
#     """Returns 'Aktuelt psykisk' notes from all years. Featurizes the notes
#     using the specified featurizer ('tfidf', 'huggingface', or `None` for raw
#     text). `featurizer_kwargs` are passed to the featurizer (e.g. "tfidf_path"
#     for tfidf, and "model_id" for huggingface).

#     Args:
#         featurizer (str): Which featurizer to use. Either 'tf-idf', 'huggingface', or None
#         n_rows (Optional[int], optional): Number of rows to load. Defaults to None.
#         featurizer_kwargs (Optional[dict], optional): Keyword arguments passed to
#             the featurizer. Defaults to None.

#     Returns:
#         pd.DataFrame: (Featurized) notes
#     """
#     return load_and_featurize_notes(
#         note_types="Aktuelt psykisk",
#         featurizer=featurizer,
#         n_rows=n_rows,
#         featurizer_kwargs=featurizer_kwargs,
#     )


# @data_loaders.register("load_note_types")
# def load_arbitrary_notes(
#     note_names: Union[str, list[str]],
#     featurizer: str,
#     n_rows: Optional[int] = None,
#     featurizer_kwargs: Optional[dict] = None,
# ) -> pd.DataFrame:
#     """Returns one or multiple note types from all years. Featurizes the notes
#     using the specified featurizer ('tfidf', 'huggingface', or `None` for raw
#     text). `featurizer_kwargs` are passed to the featurizer (e.g. "tfidf_path"
#     for tfidf, and "model_id" for huggingface).

#     Args:
#         note_names (Union[str, list[str]]): Which note types to load. See
#             `get_all_valid_note_types()` for a list of valid note types.
#         featurizer (str): Which featurizer to use. Either 'tf-idf', 'huggingface', or None
#         n_rows (Optional[int], optional): Number of rows to load. Defaults to None.
#         featurizer_kwargs (Optional[dict], optional): Keyword arguments passed to
#             the featurizer. Defaults to None.

#     Returns:
#         pd.DataFrame: (Featurized) notes
#     """
#     return load_and_featurize_notes(
#         note_names,
#         featurizer=featurizer,
#         n_rows=n_rows,
#         featurizer_kwargs=featurizer_kwargs,
#     )


# @data_loaders.register("synth_notes")
# def load_synth_notes(featurizer: str, **featurizer_kwargs) -> pd.DataFrame:
#     """Load (featurized) synthetic notes for testing.

#     Args:
#         featurizer (str): Which featurizer to use
#         **featurizer_kwargs: Keyword arguments passed to the featurizer

#     Raises:
#         ValueError: If given invalid featurizer

#     Returns:
#         pd.DataFrame: (Featurized) synthetic notes
#     """
#     p = PROJECT_ROOT / "tests" / "test_data"
#     df = pd.read_csv(
#         p / "raw" / "synth_txt_data.csv",
#     ).drop("Unnamed: 0", axis=1)
#     df = df.dropna()
#     df["timestamp"] = pd.to_datetime(df["timestamp"])

#     if featurizer == "tfidf":
#         return _tfidf_featurize(
#             df,
#             tfidf_path=p / "test_tfidf" / "tfidf_10.pkl",
#         )
#     elif featurizer == "huggingface":
#         return _huggingface_featurize(
#             df,
#             **featurizer_kwargs,
#         )

#     raise ValueError("Only tfidf or huggingface featurizer supported for synth notes")
