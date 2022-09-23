"""Load text data from a database and featurise it using a tf-idf
vectorizer."""

# pylint: disable=E0211,E0213,missing-function-docstring

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union

import dill as pkl
import pandas as pd

from psycopmlutils.loaders.raw.sql_load import sql_load
from psycopmlutils.utils import data_loaders


def get_all_valid_note_types() -> set[str]:
    """Returns a set of valid note types. Notice that 'Konklusion' is replaced
    by 'Vurdering/konklusion' in 2020, so make sure to use both. 'Ordination'
    was replaced by 'Ordination, Psykiatry' in 2022, but 'Ordination,
    Psykiatri' is not included in the table. Use with caution.

    Returns:
        Set[str]: Set of valid note types
    """
    return {
        "Observation af patient, Psykiatri",
        "Samtale med behandlingssigte",
        "Ordination",  # OBS replaced "Ordination, Psykiatri" in 01/02-22
        # but is not included in this table. Use with caution
        "Aktuelt psykisk",
        "Aktuelt socialt, Psykiatri",
        "Aftaler, Psykiatri",
        "Medicin",
        "Aktuelt somatisk, Psykiatri",
        "Objektivt psykisk",
        "Kontaktårsag",
        "Telefonkonsultation",
        "Journalnotat",
        "Telefonnotat",
        "Objektivt, somatisk",
        "Plan",
        "Semistruktureret diagnostisk interview",
        "Vurdering/konklusion",
    }


def _load_notes_for_year(
    note_types: Union[str, list[str]],
    year: str,
    view: Optional[str] = "FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret",
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Loads clinical notes from sql from a specified year and matching
    specified note types.

    Args:
        note_names (Union[str, list[str]]): Which types of notes to load.
        year (str): Which year to load
        view (str, optional): Which table to load.
            Defaults to "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret".
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with clinical notes
    """

    sql = (
        "SELECT dw_ek_borger, datotid_senest_aendret_i_sfien, fritekst"
        + f" FROM [fct].[{view}_{year}_inkl_2021_feb2022]"
        + f" WHERE overskrift IN {note_types}"
    )
    return sql_load(
        sql,
        database="USR_PS_FORSK",
        chunksize=None,
        n_rows=n_rows,
    )


def _tfidf_featurize(
    df: pd.DataFrame,
    tfidf_path: Optional[Path],
    text_col: str = "text",
) -> pd.DataFrame:
    """TF-IDF featurize text. Assumes `df` to have a column named `text`.

    Args:
        df (pd.DataFrame): Dataframe with text column
        tfidf_path (Optional[Path]): Path to a sklearn tf-idf vectorizer
        text_col (str, optional): Name of text column. Defaults to "text".

    Returns:
        pd.DataFrame: Original dataframe with tf-idf features appended
    """
    with open(tfidf_path, "rb") as f:
        tfidf = pkl.load(f)

    vocab = ["tfidf-" + word for word in tfidf.get_feature_names()]

    text = df[text_col].values
    df = df.drop(text_col, axis=1)

    text = tfidf.transform(text)
    text = pd.DataFrame(text.toarray(), columns=vocab)
    return pd.concat([df, text], axis=1)


def _huggingface_featurize(model_id: str) -> pd.DataFrame:
    # Load paraphrase-multilingual-MiniLM-L12-v2
    # split tokens to list of list if longer than allowed sequence length
    # which is often 128 for sentence transformers
    # encode tokens
    # average by list of list
    # return embeddings
    raise NotImplementedError


def _load_and_featurize_notes_per_year(
    year: str,
    note_types: Union[str, list[str]],
    view: str,
    n_rows: int,
    featurizer: str,
    featurizer_kwargs: dict,
) -> pd.DataFrame:
    """Loads clinical notes and features them.

    Args:
        note_types (Union[str, list[str]]): Which note types to load.
        year (str): Which year to load
        view (str): Which view to load
        n_rows (int): How many rows to load
        featurizer (str): Which featurizer to use (tfidf or huggingface)
        featurizer_kwargs (dict): kwargs for the featurizer

    Returns:
        pd.DataFrame: Dataframe of notes and features
    """

    df = _load_notes_for_year(
        note_types=note_types,
        year=year,
        view=view,
        n_rows=n_rows,
    )
    if featurizer == "tfidf":
        df = _tfidf_featurize(df, **featurizer_kwargs)
    elif featurizer == "huggingface":
        df = _huggingface_featurize(df, **featurizer_kwargs)
    return df


def load_and_featurize_notes(
    note_types: Union[str, list[str]],
    featurizer: str,
    featurizer_kwargs: Optional[dict] = None,
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Loads all clinical notes that match the specified note from all years.
    Featurizes the notes using the specified featurizer (tf-idf or huggingface
    model). Kwargs passed to.

    Args:
        note_types (Union[str, list[str]]): Which note types to load. See
            `get_all_valid_note_types()` for valid note types.
        featurizer (str): Which featurizer to use. Either 'tf-idf' or 'huggingface' or
            `None` to return the raw text.
        featurizer_kwargs (Optional[dict]): Kwargs passed to the featurizer. Defaults to None.
            For tf-idf, this is `tfidf_path` to the vectorizer. For huggingface,
            this is `model_id` to the model.
        n_rows (Optional[int], optional): How many rows to load. Defaults to None.

    Raises:
        ValueError: If given invalid featurizer
        ValueError: If given invlaid note type

    Returns:
        pd.DataFrame: Featurized clinical notes
    """

    valid_featurizers = {"tfidf", "huggingface", None}
    if featurizer not in valid_featurizers:
        raise ValueError(
            f"featurizer must be one of {valid_featurizers}, got {featurizer}",
        )

    if isinstance(note_types, str):
        note_types = list(note_types)  # pylint: disable=W0642
    # check for invalid note types
    if not set(note_types).issubset(get_all_valid_note_types()):
        raise ValueError(
            "Invalid note type. Valid note types are: "
            + str(get_all_valid_note_types()),
        )

    # convert note_types to sql query
    note_types = "('" + "', '".join(note_types) + "')"  # pylint: disable=W0642

    view = "FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret"

    load_and_featurize = partial(
        _load_and_featurize_notes_per_year,
        note_types=note_types,
        view=view,
        n_rows=n_rows,
        featurizer=featurizer,
        featurizer_kwargs=featurizer_kwargs,
    )

    years = list(range(2011, 2022))

    with Pool(processes=len(years)) as p:
        dfs = p.map(load_and_featurize, [str(y) for y in years])
    dfs = pd.concat(dfs)

    dfs = dfs.rename(
        {"datotid_senest_aendret_i_sfien": "timestamp", "fritekst": "text"},
        axis=1,
    )
    return dfs


@data_loaders.register("all_notes")
def load_all_notes(
    featurizer: str,
    n_rows: Optional[int] = None,
    featurizer_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Returns all notes from all years. Featurizes the notes using the
    specified featurizer ('tfidf', 'huggingface', or `None` for raw text).
    `featurizer_kwargs` are passed to the featurizer (e.g. "tfidf_path" for
    tfidf, and "model_id" for huggingface).

    Args:
        featurizer (str): Which featurizer to use. Either 'tf-idf', 'huggingface', or None
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.
        featurizer_kwargs (Optional[dict], optional): Keyword arguments passed to
            the featurizer. Defaults to None.

    Returns:
        pd.DataFrame: (Featurized) notes
    """
    return load_and_featurize_notes(
        note_types=get_all_valid_note_types(),
        featurizer=featurizer,
        n_rows=n_rows,
        featurizer_kwargs=featurizer_kwargs,
    )


@data_loaders.register("aktuelt_psykisk")
def load_aktuel_psykisk(
    featurizer: str,
    n_rows: Optional[int] = None,
    featurizer_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Returns 'Aktuelt psykisk' notes from all years. Featurizes the notes
    using the specified featurizer ('tfidf', 'huggingface', or `None` for raw
    text). `featurizer_kwargs` are passed to the featurizer (e.g. "tfidf_path"
    for tfidf, and "model_id" for huggingface).

    Args:
        featurizer (str): Which featurizer to use. Either 'tf-idf', 'huggingface', or None
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.
        featurizer_kwargs (Optional[dict], optional): Keyword arguments passed to
            the featurizer. Defaults to None.

    Returns:
        pd.DataFrame: (Featurized) notes
    """
    return load_and_featurize_notes(
        note_types="Aktuelt psykisk",
        featurizer=featurizer,
        n_rows=n_rows,
        featurizer_kwargs=featurizer_kwargs,
    )


@data_loaders.register("load_note_types")
def load_arbitrary_notes(
    note_names: Union[str, list[str]],
    featurizer: str,
    n_rows: Optional[int] = None,
    featurizer_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Returns one or multiple note types from all years. Featurizes the notes
    using the specified featurizer ('tfidf', 'huggingface', or `None` for raw
    text). `featurizer_kwargs` are passed to the featurizer (e.g. "tfidf_path"
    for tfidf, and "model_id" for huggingface).

    Args:
        note_names (Union[str, list[str]]): Which note types to load. See
            `get_all_valid_note_types()` for a list of valid note types.
        featurizer (str): Which featurizer to use. Either 'tf-idf', 'huggingface', or None
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.
        featurizer_kwargs (Optional[dict], optional): Keyword arguments passed to
            the featurizer. Defaults to None.

    Returns:
        pd.DataFrame: (Featurized) notes
    """
    return load_and_featurize_notes(
        note_names,
        featurizer=featurizer,
        n_rows=n_rows,
        featurizer_kwargs=featurizer_kwargs,
    )


@data_loaders.register("synth_notes")
def load_synth_notes(featurizer: str) -> pd.DataFrame:
    """Load (featurized) synthetic notes for testing.

    Args:
        featurizer (str): Which featurizer to use

    Raises:
        ValueError: If given invalid featurizer

    Returns:
        pd.DataFrame: (Featurized) synthetic notes
    """
    p = Path("tests") / "test_data"
    df = pd.read_csv(p / "raw" / "synth_txt_data.csv")
    df = df.dropna()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if featurizer == "tfidf":
        return _tfidf_featurize(
            df,
            tfidf_path=p / "test_tfidf" / "tfidf_10.pkl",
        )

    raise ValueError("Only tfidf featurizer supported for synth notes")
