"""Loaders for structured SFI-data."""

# pylint: disable = missing-function-docstring

from typing import Optional

import pandas as pd

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.utils import data_loaders


def sfi_loader(
    sfi_type: Optional[str] = None,
    element_type: Optional[str] = None,
    n_rows: Optional[int] = None,
    value_col: str = "numelementvaerdi",
) -> pd.DataFrame:
    """Load structured_sfi data. By default returns entire structured_sfi data
    view with numelementværdi as the value column.

    Args:
        sfi_type (str): Type of structured_sfi, e.g. 'broeset_violence_checklist', 'selvmordsvurdering'. Defaults to None. # noqa: DAR102
        element_type (str): elementledetekst which specifies which sub-element of the SFI, e.g. 'Sum', "Selvmordstanker". Defaults to None.
        n_rows: Number of rows to return. Defaults to None which returns entire structured_sfi data view.
        value_col: Column to return as value col. Defaults to 'numelementvaerdi'.

    Returns:
        pd.DataFrame
    """
    view = "[FOR_SFI_uden_fritekst_resultater_psyk_somatik_inkl_2021]"
    sql = f"SELECT dw_ek_borger, datotid_resultat_udfoert, aktivitetstypenavn, elementledetekst, {value_col} FROM [fct].{view} WHERE datotid_resultat_udfoert IS NOT NULL"

    if element_type:
        sql += f" AND aktivitetstypenavn = '{sfi_type}'"
    if sfi_type:
        sql += f" AND elementledetekst = '{element_type}'"

    df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)

    # Drop duplicate rows
    df = df.drop_duplicates(keep="first")

    df.rename(
        columns={
            "datotid_resultat_udfoert": "timestamp",
            value_col: "value",
        },
        inplace=True,
    )

    return df.reset_index(drop=True)


@data_loaders.register("broeset_violence_checklist")
def broeset_violence_checklist(n_rows: Optional[int] = None) -> pd.DataFrame:
    return sfi_loader(
        sfi_type="Brøset Violence Checkliste (BVC)",
        element_type="Sum",
        n_rows=n_rows,
    )


@data_loaders.register("selvmordsrisiko")
def selvmordsrisiko(n_rows: Optional[int] = None) -> pd.DataFrame:
    df = sfi_loader(
        sfi_type="Screening for selvmordsrisiko",
        element_type="ScrSelvmordlRisikoniveauKonkl",
        n_rows=n_rows,
        value_col="elementkode",
    )

    df["value"] = df["value"].replace(
        to_replace=[
            "010ScrSelvmordKonklRisikoniveau1",
            "020ScrSelvmordKonklRisikoniveau2",
            "030ScrSelvmordKonklRisikoniveau3",
        ],
        value=[1, 2, 3],
        regex=False,
    )

    return df


@data_loaders.register("hamilton_d17")
def hamilton_d17(n_rows: Optional[int] = None) -> pd.DataFrame:
    return sfi_loader(
        sfi_type="Vurdering af depressionssværhedsgrad med HAM-D17",
        element_type="Samlet score HAM-D17",
        n_rows=n_rows,
    )


@data_loaders.register("mas_m")
def mas_m(n_rows: Optional[int] = None) -> pd.DataFrame:
    return sfi_loader(
        sfi_type="MAS-M maniscoringsskema (Modificeret Bech-Rafaelsen Maniskala)",
        element_type="MAS-M score",
        n_rows=n_rows,
        value_col="numelementvaerdi",
    )
