"""Loaders for diagnosis codes.

Is growing quite a bit, loaders may have to be split out into separate
files (e.g. psychiatric, cardiovascular, metabolic etc.) over time.
"""

# pylint: disable=missing-function-docstring

from typing import Optional, Union

import pandas as pd

from psycop_feature_generation.loaders.raw.utils import load_from_codes
from psycop_feature_generation.utils import data_loaders


def concat_from_physical_visits(
    icd_codes: list[str],
    output_col_name: str,
    wildcard_icd_code: Optional[bool] = False,
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load all diagnoses matching any icd_code in icd_codes. Create
    output_col_name and set to 1.

    Args:
        icd_codes (list[str]): list of icd_codes. # noqa: DAR102
        output_col_name (str): Output column name
        wildcard_icd_code (bool, optional): Whether to match on icd_codes* or icd_codes. Defaults to False.
        n_rows: Number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame
    """

    diagnoses_source_table_info = {
        "lpr3": {
            "view": "FOR_LPR3kontakter_psyk_somatik_inkl_2021_feb2022",
            "source_timestamp_col_name": "datotid_lpr3kontaktstart",
        },
        "lpr2_inpatient": {
            "view": "FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021_feb2022",
            "source_timestamp_col_name": "datotid_indlaeggelse",
        },
        "lpr2_acute_outpatient": {
            "view": "FOR_akutambulantekontakter_psyk_somatik_LPR2_inkl_2021_feb2022",
            "source_timestamp_col_name": "datotid_start",
        },
        "lpr2_outpatient": {
            "view": "FOR_besoeg_psyk_somatik_LPR2_inkl_2021_feb2022",
            "source_timestamp_col_name": "datotid_start",
        },
    }

    dfs = [
        load_from_codes(
            codes_to_match=icd_codes,
            code_col_name="diagnosegruppestreng",
            output_col_name=output_col_name,
            match_with_wildcard=wildcard_icd_code,
            n_rows=n_rows,
            load_diagnoses=True,
            **kwargs,
        )
        for _, kwargs in diagnoses_source_table_info.items()
    ]

    df = pd.concat(dfs).drop_duplicates(
        subset=["dw_ek_borger", "timestamp", "value"],
        keep="first",
    )
    return df.reset_index(drop=True)


def from_physical_visits(
    icd_code: Union[list[str], str],
    output_col_name: Optional[str] = "value",
    n_rows: Optional[int] = None,
    wildcard_icd_code: Optional[bool] = False,
) -> pd.DataFrame:
    """Load diagnoses from all physical visits. If icd_code is a list, will
    aggregate as one column (e.g. ["E780", "E785"] into a ypercholesterolemia
    column).

    Args:
        icd_code (str): Substring to match diagnoses for. Matches any diagnoses, whether a-diagnosis, b-diagnosis etc. # noqa: DAR102
        output_col_name (str, optional): Name of new column string. Defaults to "value".
        n_rows: Number of rows to return. Defaults to None.
        wildcard_icd_code (bool, optional): Whether to match on icd_code*. Defaults to False.

    Returns:
        pd.DataFrame
    """

    diagnoses_source_table_info = {
        "lpr3": {
            "view": "FOR_LPR3kontakter_psyk_somatik_inkl_2021",
            "source_timestamp_col_name": "datotid_lpr3kontaktstart",
        },
        "lpr2_inpatient": {
            "view": "FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021",
            "source_timestamp_col_name": "datotid_indlaeggelse",
        },
        "lpr2_outpatient": {
            "view": "FOR_besoeg_psyk_somatik_LPR2_inkl_2021",
            "source_timestamp_col_name": "datotid_start",
        },
    }

    if n_rows:
        n_rows_per_df = int(n_rows / len(diagnoses_source_table_info))
    else:
        n_rows_per_df = None

    dfs = [
        load_from_codes(
            codes_to_match=icd_code,
            code_col_name="diagnosegruppestreng",
            output_col_name=output_col_name,
            n_rows=n_rows_per_df,
            match_with_wildcard=wildcard_icd_code,
            **kwargs,
            load_diagnoses=True,
        )
        for _, kwargs in diagnoses_source_table_info.items()
    ]

    df = pd.concat(dfs).drop_duplicates(
        subset=["dw_ek_borger", "timestamp", "value"],
        keep="first",
    )

    return df.reset_index(drop=True)


@data_loaders.register("essential_hypertension")
def essential_hypertension(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="I109",
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("hyperlipidemia")
def hyperlipidemia(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=[
            "E780",
            "E785",
        ],  # Only these two, as the others are exceedingly rare
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("liverdisease_unspecified")
def liverdisease_unspecified(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="K769",
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("polycystic_ovarian_syndrome")
def polycystic_ovarian_syndrome(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="E282",
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("sleep_apnea")
def sleep_apnea(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["G473", "G4732"],
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("sleep_problems_unspecified")
def sleep_problems_unspecified(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="G479",
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("copd")
def copd(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["j44"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


# Psychiatric diagnoses
# data loaders for all diagnoses in the f0-chapter (organic mental disorders)
@data_loaders.register("f0_disorders")
def f0_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f0",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("dementia")
def dementia(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f00", "f01", "f02", "f03", "f04"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("delirium")
def delirium(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f05",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("miscellaneous_organic_mental_disorders")
def misc_organic_mental_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f06", "f07", "f09"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


# data loaders for all diagnoses in the f1-chapter (mental and behavioural disorders due to psychoactive substance use)
@data_loaders.register("f1_disorders")
def f1_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f1",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("alcohol_dependency")
def alcohol_dependency(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f10",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("opioid_dependency")
def opioids_and_sedatives(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f11",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("cannabinoid_dependency")
def cannabinoid_dependency(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f12",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("sedative_dependency")
def sedative_dependency(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f13",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("stimulant_dependencies")
def stimulant_deo(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f14", "f15"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("hallucinogen_dependency")
def hallucinogen_dependency(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f16",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("tobacco_dependency")
def tobacco_dependency(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f17",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("miscellaneous_drug_dependencies")
def misc_drugs(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f18", "f19"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


# data loaders for all diagnoses in the f2-chapter (schizophrenia, schizotypal and delusional disorders)


@data_loaders.register("f2_disorders")
def f2_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f2",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("schizophrenia")
def schizophrenia(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f20",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("schizoaffective")
def schizoaffective(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f25",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("miscellaneous_psychotic_disorders")
def misc_psychosis(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f21", "f22", "f23", "f24", "f28", "f29"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


# data loaders for all diagnoses in the f3-chapter (mood (affective) disorders).


@data_loaders.register("f3_disorders")
def f3_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f3",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("manic_and_bipolar")
def manic_and_bipolar(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f30", "f31"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("depressive_disorders")
def depressive_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f32", "f33", "f34", "f38"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("miscellaneous_affective_disorders")
def misc_affective_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f38", "f39"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


# data loaders for all diagnoses in the f4-chapter (neurotic, stress-related and somatoform disorders).


@data_loaders.register("f4_disorders")
def f4_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f4",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("phobic_anxiety_ocd")
def phobic_and_anxiety(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f40", "f41", "f42"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("reaction_to_severe_stress_and_adjustment_disorders")
def stress_and_adjustment(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f43",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("dissociative_somatoform_miscellaneous")
def dissociative_somatoform_and_misc(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f44", "f45", "f48"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


# data loaders for all diagnoses in the f5-chapter (behavioural syndromes associated with physiological disturbances and physical factors).


@data_loaders.register("f5_disorders")
def f5_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f5",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("eating_disorders")
def eating_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f50",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("sleeping_and_sexual_disorders")
def sleeping_and_sexual_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f51", "f52"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("miscellaneous_f5_disorders")
def misc_f5(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f53", "f54", "f55", "f59"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


# data loaders for all diagnoses in the f6-chapter (disorders of adult personality and behaviour).
@data_loaders.register("f6_disorders")
def f6_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f6",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("cluster_a")
def cluster_a(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f600", "f601"],
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("cluster_b")
def cluster_b(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f602", "f603", "f604"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("cluster_c")
def cluster_c(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f605", "f606", "f607"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("miscellaneous_personality_disorders")
def misc_personality_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f608", "f609", "f61", "f62", "f63", "f68", "f69"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("sexual_disorders")
def misc_personality(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f65", "f66"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )

    # f64 sexual identity disorders is excluded


# data loaders for all diagnoses in the f7-chapter (mental retardation).
@data_loaders.register("f7_disorders")
def f7_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f7",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("mild_mental_retardation")
def mild_mental_retardation(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f70",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("moderate_mental_retardation")
def moderate_mental_retardation(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f71",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("severe_mental_retardation")
def severe_mental_retardation(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f72", "f73"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("miscellaneous_mental_retardation_disorders")
def misc_mental_retardation(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f78", "f79"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


# data loaders for all diagnoses in the f8-chapter (disorders of psychological development).
@data_loaders.register("f8_disorders")
def f8_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f8",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("pervasive_developmental_disorders")
def pervasive_developmental_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f84",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("miscellaneous_f8_disorders")
def misc_f8(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f80", "f81", "f82", "f83", "f88", "f89"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


# data loaders for all diagnoses in the f9-chapter (child and adolescent disorders).
@data_loaders.register("f9_disorders")
def f9_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f9",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("hyperkinetic_disorders")
def hyperkinetic_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code="f90",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("behavioural_disorders")
def behavioural_disorders(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f91", "f92", "f93", "f94"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("tics_and_miscellaneous_f9")
def tics_and_misc(n_rows: Optional[int] = None) -> pd.DataFrame:
    return from_physical_visits(
        icd_code=["f95", "f98"],
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("gerd")
def gerd(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Gastroesophageal reflux disease (GERD) diagnoses"""
    return from_physical_visits(
        icd_code="k21",
        wildcard_icd_code=True,
        n_rows=n_rows,
    )
