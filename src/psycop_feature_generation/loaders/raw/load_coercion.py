"""Loaders for coercion data."""

# pylint: disable = non-ascii-name,missing-function-docstring

from typing import Optional

import pandas as pd

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.utils import data_loaders


@data_loaders.register("coercion_duration")
def coercion_duration(
    coercion_type: Optional[str] = None,
    reason_for_coercion: Optional[str] = None,
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load coercion data. By default returns entire coercion data view with
    duration in hours as the value column.

    Args:
        coercion_type (str): Type of coercion, e.g. 'tvangsindlæggelse', 'bæltefiksering'. Defaults to None. # noqa: DAR102
        reason_for_coercion (str): Reason for coercion, e.g. 'farlighed'. Defaults to None.
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    coercion_discard = """('Døraflåsning', 'Personlig afskærmning over 24 timer', 'Koordinationsplan',
    'Udskrivningsaftale', 'Særlige dørlåse', 'Personlige alarm- og pejlesystemer', 'Andet' )"""

    view = "[FOR_tvang_alt_hele_kohorten_inkl_2021]"

    sql = f"SELECT dw_ek_borger, datotid_start_sei, varighed_timer_sei, typetekst_sei FROM [fct].{view} WHERE datotid_start_sei IS NOT NULL AND typetekst_sei NOT IN {coercion_discard}"

    if coercion_type and reason_for_coercion is None:

        sql += f" AND typetekst_sei = '{coercion_type}'"

    if coercion_type is None and reason_for_coercion:

        sql += f" AND begrundtekst_sei = '{reason_for_coercion}'"

    if coercion_type and reason_for_coercion:

        sql += f" AND typetekst_sei = '{coercion_type}' AND begrundtekst_sei = '{reason_for_coercion}'"

    df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)

    # Drop duplicate rows
    df = df.drop_duplicates(keep="first")

    df.rename(
        columns={"datotid_start_sei": "timestamp", "varighed_timer_sei": "value"},
        inplace=True,
    )

    # Change NaNs to 0
    df["value"].fillna(0, inplace=True)

    return df.reset_index(drop=True)


def _concatenate_coercion(
    coercion_types_list: list[dict[str, str]],
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate multiple types of coercion with multiple reasons into one
    column.

    Args:
        coercion_types_list (list): list of dictionaries containing a 'coercion_type' key and a 'reason_for_coercion' key. If keys not in dicts, they are set to None # noqa: DAR102
        n (int, optional): Number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame
    """

    for d in coercion_types_list:  # Make sure proper keys are given
        if "coercion_type" not in d and "reason_for_coercion" not in d:
            raise KeyError(
                f'{d} does not contain either "coercion_type"  or "reason_for_coercion". At least one is required.',
            )
        if "coercion_type" not in d:
            d["coercion_type"] = None  # type: ignore
        if "reason_for_coercion" not in d:
            d["reason_for_coercion"] = None  # type: ignore

    dfs = [
        coercion_duration(
            coercion_type=d["coercion_type"],
            reason_for_coercion=d["reason_for_coercion"],
            n_rows=n_rows,
        )
        for d in coercion_types_list
    ]

    return pd.concat(dfs, axis=0).reset_index(drop=True)


# REASON (begrundtekst_sei) #


@data_loaders.register("farlighed")
def farlighed(n_rows: Optional[int] = None) -> pd.DataFrame:
    coercion_types_list = [
        {
            "reason_for_coercion": "Farlighed",
        },
        {
            "reason_for_coercion": "På grund af farlighed",
        },
    ]

    return _concatenate_coercion(
        coercion_types_list=coercion_types_list,
        n_rows=n_rows,
    )


# Røde papir ved tvangsindlæggelse/tvangstilbageholdelse
@data_loaders.register("paa_grund_af_farlighed")
def paa_grund_af_farlighed(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        reason_for_coercion="På grund af farlighed",
        n_rows=n_rows,
    )


# Gule papir ved tvangsindlæggelse/tvangstilbageholdelse
@data_loaders.register("af_helbredsmaessige_grunde")
def af_helbredsmaessige_grunde(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        reason_for_coercion=" Af helbredsmæssige grunde",
        n_rows=n_rows,
    )


@data_loaders.register("urolig_tilstand")
def urolig_tilstand(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        reason_for_coercion="Urolig tilstand",
        n_rows=n_rows,
    )


@data_loaders.register("anden_begrundelse")
def anden_begrundelse(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        reason_for_coercion="Anden begrundelse",
        n_rows=n_rows,
    )


@data_loaders.register("naerliggende_eller_vaesentlig_fare_for_patienten_eller_andre")
def naerliggende_fare(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        reason_for_coercion="Nærliggende_eller_væsentlig_fare_for_patienten_eller_andre",
        n_rows=n_rows,
    )


# GENERAL TYPE (tabeltekst) ###
# frihedsberøvelser
@data_loaders.register("skema_1")
def skema_1(n_rows: Optional[int] = None) -> pd.DataFrame:
    coercion_types_list = [
        {
            "coercion_type": "Tvangsindlæggelse",
        },
        {
            "coercion_type": "Tvangstilbageholdelse",
        },
    ]

    return _concatenate_coercion(
        coercion_types_list=coercion_types_list,
        n_rows=n_rows,
    )


# tvangsbehandlinger
@data_loaders.register("skema_2")
def skema_2(n_rows: Optional[int] = None) -> pd.DataFrame:
    coercion_types_list = [
        {
            "coercion_type": "Af legemlig lidelse",
        },
        {
            "coercion_type": "Medicinering",
        },
        {
            "coercion_type": "Ernæring",
        },
        {
            "coercion_type": "ECT",
        },
    ]

    return _concatenate_coercion(
        coercion_types_list=coercion_types_list,
        n_rows=n_rows,
    )


# magtanvendelse
@data_loaders.register("skema_3")
def skema_3(n_rows: Optional[int] = None) -> pd.DataFrame:
    coercion_types_list = [
        {
            "coercion_type": "Bælte",
        },
        {
            "coercion_type": "Remme",
        },
        {
            "coercion_type": "Fastholden",
        },
        {
            "coercion_type": "Beroligende medicin",
        },
        {
            "coercion_type": "Handsker",
        },
    ]

    # "døraflåsning" and "personlig skærmning" are not included

    return _concatenate_coercion(
        coercion_types_list=coercion_types_list,
        n_rows=n_rows,
    )


# SPECIFIC TYPE (typetekst_sei) ###
# exists in the data, but not included here: [døraflåsning, personlig afskærmning, stofbælte, særlige dørlåse, tvungen opfølgning, personlige alarm, udskrivningsaftale, koordinationsplan]


@data_loaders.register("baelte")
def baelte(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        coercion_type="Bælte",
        n_rows=n_rows,
    )


@data_loaders.register("remme")
def remme(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        coercion_type="Remme",
        n_rows=n_rows,
    )


@data_loaders.register("fastholden")
def fastholden(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        coercion_type="Fastholden",
        n_rows=n_rows,
    )


@data_loaders.register("beroligende_medicin")
def beroligende_medicin(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        coercion_type="Beroligende medicin",
        n_rows=n_rows,
    )


@data_loaders.register("handsker")
def handsker(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        coercion_type="Handsker",
        n_rows=n_rows,
    )


@data_loaders.register("tvangsindlaeggelse")
def tvangsindlaeggelse(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        coercion_type="Tvangsindlæggelse",
        n_rows=n_rows,
    )


@data_loaders.register("tvangstilbageholdelse")
def tvangstilbageholdelse(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        coercion_type="Tvangstilbageholdelse",
        n_rows=n_rows,
    )


@data_loaders.register("medicinering")
def medicinering(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        coercion_type="Medicinering",
        n_rows=n_rows,
    )


@data_loaders.register("ect")
def ect(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        coercion_type="ECT",
        n_rows=n_rows,
    )


@data_loaders.register("ernaering")
def ernaering(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        coercion_type="Ernæring",
        n_rows=n_rows,
    )


@data_loaders.register("af_legemlig_lidelse")
def af_legemlig_lidelse(n_rows: Optional[int] = None) -> pd.DataFrame:
    return coercion_duration(
        coercion_type="Af legemlig lidelse",
        n_rows=n_rows,
    )
