"""Loaders for medications."""
from typing import Optional, Union

import pandas as pd
from wasabi import msg

from psycop_feature_generation.loaders.raw.utils import load_from_codes
from psycop_feature_generation.utils import data_loaders

# pylint: disable=missing-function-docstring


def load(
    atc_code: Union[str, list[str]],
    output_col_name: Optional[str] = None,
    load_prescribed: Optional[bool] = False,
    load_administered: Optional[bool] = True,
    wildcard_code: Optional[bool] = True,
    n_rows: Optional[int] = None,
    exclude_atc_codes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load medications. Aggregates prescribed/administered if both true. If
    wildcard_atc_code, match from atc_code*. Aggregates all that match. Beware
    that data is incomplete prior to sep. 2016 for prescribed medications.

    Args:
        atc_code (str): ATC-code prefix to load. Matches atc_code_prefix*.
            Aggregates all.
        output_col_name (str, optional): Name of output_col_name. Contains 1 if
            atc_code matches atc_code_prefix, 0 if not.Defaults to
            {atc_code_prefix}_value.
        load_prescribed (bool, optional): Whether to load prescriptions. Defaults to
            False. Beware incomplete until sep 2016.
        load_administered (bool, optional): Whether to load administrations.
            Defaults to True.
        wildcard_code (bool, optional): Whether to match on atc_code* or
            atc_code.
        n_rows (int, optional): Number of rows to return. Defaults to None, in which case all rows are returned.
        exclude_atc_codes (list[str], optional): Drop rows if atc_code is a direct match to any of these. Defaults to None.

    Returns:
        pd.DataFrame: Cols: dw_ek_borger, timestamp, {atc_code_prefix}_value = 1
    """

    if load_prescribed:
        msg.warn(
            "Beware, there are missing prescriptions until september 2016. "
            "Hereafter, data is complete. See the wiki (OBS: Medication) for more details.",
        )

    df = pd.DataFrame()

    if load_prescribed and load_administered:
        n_rows = int(n_rows / 2) if n_rows else None

    if load_prescribed:
        df_medication_prescribed = load_from_codes(
            codes_to_match=atc_code,
            code_col_name="atc",
            source_timestamp_col_name="datotid_ordinationstart",
            view="FOR_Medicin_ordineret_inkl_2021_feb2022",
            output_col_name=output_col_name,
            match_with_wildcard=wildcard_code,
            n_rows=n_rows,
            exclude_codes=exclude_atc_codes,
            load_diagnoses=False,
        )

        df = pd.concat([df, df_medication_prescribed])

    if load_administered:
        df_medication_administered = load_from_codes(
            codes_to_match=atc_code,
            code_col_name="atc",
            source_timestamp_col_name="datotid_administration_start",
            view="FOR_Medicin_administreret_inkl_2021_feb2022",
            output_col_name=output_col_name,
            match_with_wildcard=wildcard_code,
            n_rows=n_rows,
            exclude_codes=exclude_atc_codes,
            load_diagnoses=False,
        )
        df = pd.concat([df, df_medication_administered])

    if output_col_name is None:
        if isinstance(atc_code, list):
            # Joint list of atc_codes
            output_col_name = "_".join(atc_code)
        else:
            output_col_name = atc_code

    df.rename(
        columns={
            output_col_name: "value",
        },
        inplace=True,
    )

    return df.reset_index(drop=True).drop_duplicates(
        subset=["dw_ek_borger", "timestamp", "value"],
        keep="first",
    )


def concat_medications(
    output_col_name: str,
    atc_code_prefixes: list[str],
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate multiple blood_sample_ids (typically NPU-codes) into one
    column.

    Args:
        output_col_name (str): Name for new column.  # noqa: DAR102
        atc_code_prefixes (list[str]): list of atc_codes.
        n_rows (int, optional): Number of atc_codes to aggregate. Defaults to None.

    Returns:
        pd.DataFrame
    """
    dfs = [
        load(
            atc_code=f"{id}",
            output_col_name=output_col_name,
            n_rows=n_rows,
        )
        for id in atc_code_prefixes
    ]

    return (
        pd.concat(dfs, axis=0)
        .drop_duplicates(
            subset=["dw_ek_borger", "timestamp", "value"],
            keep="first",
        )
        .reset_index(drop=True)
    )


# data_loaders primarly used in psychiatry
@data_loaders.register("antipsychotics")
def antipsychotics(n_rows: Optional[int] = None) -> pd.DataFrame:
    """All antipsyhotics, except Lithium. Lithium is typically considered a mood stabilizer, not an antipsychotic."""
    return load(
        atc_code="N05A",
        load_prescribed=True,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
        exclude_atc_codes=["N05AN01"],
    )


# 1. generation antipsychotics [flupentixol, pimozid, haloperidol, zuclopenthixol, melperon,pipamperon, chlorprotixen]
@data_loaders.register("first_gen_antipsychotics")
def first_gen_antipsychotics(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code=[
            "N05AF01",
            "N05AG02",
            "N05AD01",
            "N05AF05",
            "N05AD03",
            "N05AD05",
            "N05AF03",
        ],
        load_prescribed=True,
        load_administered=True,
        wildcard_code=False,
        n_rows=n_rows,
    )


# 2. generation antipsychotics [amisulpride, aripiprazole,asenapine, brexpiprazole, cariprazine, lurasidone, olanzapine, paliperidone, Quetiapine, risperidone, sertindol]
@data_loaders.register("second_gen_antipsychotics")
def second_gen_antipsychotics(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code=[
            "N05AL05",
            "N05AX12",
            "N05AH05",
            "N05AX16",
            "N05AX15",
            "N05AE02",
            "N05AE05",
            "N05AH03",
            "N05AX13",
            "N05AH04",
            "N05AX08",
            "N05AE04",
            "N05AE03",
        ],
        load_prescribed=True,
        load_administered=True,
        wildcard_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("top_10_weight_gaining_antipsychotics")
def top_10_weight_gaining_antipsychotics(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Top 10 weight gaining antipsychotics based on Huhn et al. 2019. Only 5 of them are marketed in Denmark."""
    return load(
        atc_code=[
            "N05AH03",
            "N05AE03",
            "N05AH04",
            "N05AX13",
            "N05AX08",
        ],
        load_prescribed=True,
        load_administered=True,
        wildcard_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("olanzapine")
def olanzapine(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N05AH03",
        load_prescribed=True,
        load_administered=True,
        wildcard_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("clozapine")
def clozapine(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N05AH02",
        load_prescribed=True,
        load_administered=True,
        wildcard_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("anxiolytics")
def anxiolytics(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N05B",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("benzodiazepines")
def benzodiazepines(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N05BA",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("benzodiazepine_related_sleeping_agents")
def benzodiazepine_related_sleeping_agents(
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    return load(
        atc_code=["N05CF01", "N05CF02"],
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("pregabaline")
def pregabaline(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N03AX16",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("hypnotics and sedatives")
def hypnotics(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N05C",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("antidepressives")
def antidepressives(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N06A",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


# SSRIs [escitalopram, citalopram, fluvoxamin, fluoxetin, paroxetin]
@data_loaders.register("ssri")
def ssri(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N06AB",
        load_prescribed=True,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


# SNRIs [duloxetin, venlafaxin]
@data_loaders.register("snri")
def snri(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code=["N06AX21", "N06AX16"],
        load_prescribed=True,
        load_administered=True,
        wildcard_code=False,
        n_rows=n_rows,
    )


# TCAs
@data_loaders.register("tca")
def tca(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N06AA",
        load_prescribed=True,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("selected_nassa")
def selected_nassa(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code=["N06AX11", "N06AX03"],
        load_prescribed=True,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("lithium")
def lithium(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N05AN01",
        load_prescribed=True,
        load_administered=True,
        wildcard_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("valproate")
def valproate(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N03AG01",
        load_prescribed=True,
        load_administered=True,
        wildcard_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("lamotrigine")
def lamotrigine(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N03AX09",
        load_prescribed=True,
        load_administered=True,
        wildcard_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("hyperactive disorders medications")
def hyperactive_disorders_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N06B",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("dementia medications")
def dementia_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N06D",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("anti-epileptics")
def anti_epileptics(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N03",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


# medications used in alcohol abstinence treatment [thiamin, b-combin, klopoxid, fenemal]
@data_loaders.register("alcohol_abstinence")
def alcohol_abstinence(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code=["A11DA01", "A11EA", "N05BA02", "N03AA02"],
        load_prescribed=True,
        load_administered=True,
        wildcard_code=False,
        n_rows=n_rows,
    )


# data loaders for medications primarily used outside psychiatry
@data_loaders.register("alimentary_tract_and_metabolism_medications")
def alimentary_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="A",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("blood_and_blood_forming_organs_medications")
def blood_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="B",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("cardiovascular_medications")
def cardiovascular_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="C",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("dermatologicals")
def dermatological_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="D",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("genito_urinary_system_and_sex_hormones_medications")
def genito_sex_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="G",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("systemic_hormonal_preparations")
def hormonal_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="H",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("antiinfectives")
def antiinfectives(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="J",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("antineoplastic")
def antineoplastic(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="L",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("musculoskeletal_medications")
def musculoskeletal_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="M",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("nervous_system_medications")
def nervous_system_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("analgesics")
def analgesic(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N02",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("antiparasitic")
def antiparasitic(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="P",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("respiratory_medications")
def respiratory_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="R",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("sensory_organs_medications")
def sensory_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="S",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("various_medications")
def various_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="V",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("statins")
def statins(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="C10AA",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("antihypertensives")
def antihypertensives(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="C02",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("diuretics")
def diuretics(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="C07",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("gerd_drugs")
def gerd_drugs(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Gastroesophageal reflux disease (GERD) drugs"""
    return load(
        atc_code="A02",
        load_prescribed=False,
        load_administered=True,
        wildcard_code=True,
        n_rows=n_rows,
    )
