"""Loaders for lab results loading."""

from typing import Optional, Union

import pandas as pd

from psycop_feature_generation.loaders.non_numerical_coercer import (
    multiply_inequalities_in_df,
)
from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.utils import data_loaders

# pylint: disable=missing-function-docstring


def load_non_numerical_values_and_coerce_inequalities(
    blood_sample_id: Union[str, list[str]],
    n_rows: Optional[int],
    view: str,
    ineq2mult: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Load non-numerical values for a blood sample.

    Args:
        blood_sample_id (Union[str, list]): The blood_sample_id, typically an NPU code. If a list, concatenates the values. # noqa: DAR102
        n_rows (Optional[int]): Number of rows to return. Defaults to None.
        view (str): The view to load from.
        ineq2mult (dict[str, float]): A dictionary mapping inequalities to a multiplier. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe with the non-numerical values.
    """
    cols = "dw_ek_borger, datotid_sidstesvar, svar"

    if isinstance(blood_sample_id, list):
        npu_codes = ", ".join(
            [f"'{x}'" for x in blood_sample_id],  # pylint: disable=not-an-iterable
        )

        npu_where = f"npukode in ({npu_codes})"
    else:
        npu_where = f"npukode = '{blood_sample_id}'"

    sql = f"SELECT {cols} FROM [fct].{view} WHERE datotid_sidstesvar IS NOT NULL AND {npu_where} AND numerisksvar IS NULL AND (left(Svar,1) = '>' OR left(Svar, 1) = '<')"

    df = sql_load(
        sql,
        database="USR_PS_FORSK",
        chunksize=None,
        n_rows=n_rows,
    )

    df.rename(
        columns={"datotid_sidstesvar": "timestamp", "svar": "value"},
        inplace=True,
    )

    return multiply_inequalities_in_df(df, ineq2mult=ineq2mult)


def load_numerical_values(
    blood_sample_id: str,
    n_rows: Optional[int],
    view: str,
) -> pd.DataFrame:
    """Load numerical values for a blood sample.

    Args:
        blood_sample_id (str): The blood_sample_id, typically an NPU code.  # noqa: DAR102
        n_rows (Optional[int]): Number of rows to return. Defaults to None.
        view (str): The view to load from.

    Returns:
        pd.DataFrame: A dataframe with the numerical values.
    """

    cols = "dw_ek_borger, datotid_sidstesvar, numerisksvar"

    if isinstance(blood_sample_id, list):
        npu_codes = ", ".join(
            [f"'{x}'" for x in blood_sample_id],  # pylint: disable=not-an-iterable
        )

        npu_where = f"npukode in ({npu_codes})"
    else:
        npu_where = f"npukode = '{blood_sample_id}'"

    sql = f"SELECT {cols} FROM [fct].{view} WHERE datotid_sidstesvar IS NOT NULL AND {npu_where} AND numerisksvar IS NOT NULL"
    df = sql_load(
        sql,
        database="USR_PS_FORSK",
        chunksize=None,
        n_rows=n_rows,
    )

    df.rename(
        columns={"datotid_sidstesvar": "timestamp", "numerisksvar": "value"},
        inplace=True,
    )

    return df


def load_cancelled(
    blood_sample_id: str,
    n_rows: Optional[int],
    view: str,
) -> pd.DataFrame:
    """Load cancelled samples for a blood sample.

    Args:
        blood_sample_id (str): The blood_sample_id, typically an NPU code.  # noqa: DAR102
        n_rows (Optional[int]): Number of rows to return. Defaults to None.
        view (str): The view to load from.

    Returns:
        pd.DataFrame: A dataframe with the timestamps for cancelled values.
    """
    cols = "dw_ek_borger, datotid_sidstesvar"

    if isinstance(blood_sample_id, list):
        npu_codes = ", ".join(
            [f"'{x}'" for x in blood_sample_id],  # pylint: disable=not-an-iterable
        )

        npu_where = f"npukode in ({npu_codes})"
    else:
        npu_where = f"npukode = '{blood_sample_id}'"

    sql = f"SELECT {cols} FROM [fct].{view} {npu_where} AND datotid_sidstesvar IS NOT NULL AND Svar == 'Aflyst' AND (left(Svar,1) == '>' OR left(Svar, 1) == '<')"

    df = sql_load(
        sql,
        database="USR_PS_FORSK",
        chunksize=None,
        n_rows=n_rows,
    )

    # Create the value column == 1, since all timestamps here are from cancelled blood samples
    df["value"] = 1

    df.rename(
        columns={"datotid_sidstesvar": "timestamp"},
        inplace=True,
    )

    return df


def load_all_values(
    blood_sample_id: str,
    n_rows: Optional[int],
    view: str,
) -> pd.DataFrame:
    """Load all samples for a blood sample.

    Args:
        blood_sample_id (str): The blood_sample_id, typically an NPU code.  # noqa: DAR102
        n_rows (Optional[int]): Number of rows to return. Defaults to None.
        view (str): The view to load from.

    Returns:
        pd.DataFrame: A dataframe with all values.
    """
    cols = "dw_ek_borger, datotid_sidstesvar, svar"

    if isinstance(blood_sample_id, list):
        npu_codes = ", ".join(
            [f"'{x}'" for x in blood_sample_id],  # pylint: disable=not-an-iterable
        )

        npu_where = f"npukode in ({npu_codes})"
    else:
        npu_where = f"npukode = '{blood_sample_id}'"

    sql = f"SELECT {cols} FROM [fct].{view} WHERE datotid_sidstesvar IS NOT NULL AND {npu_where}"

    df = sql_load(
        sql,
        database="USR_PS_FORSK",
        chunksize=None,
        n_rows=n_rows,
    )

    df.rename(
        columns={"datotid_sidstesvar": "timestamp", "svar": "value"},
        inplace=True,
    )

    return df


def blood_sample(
    blood_sample_id: Union[str, list],
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    """Load a blood sample.

    Args:
        blood_sample_id (Union[str, list]): The blood_sample_id, typically an NPU code. If a list, concatenates the values. # noqa: DAR102
        n_rows: Number of rows to return. Defaults to None.
        values_to_load (str): Which values to load. Takes either "numerical", "numerical_and_coerce", "cancelled" or "all". Defaults to None, which is coerced to "all".

    Returns:
        pd.DataFrame
    """
    view = "[FOR_labka_alle_blodprover_inkl_2021_feb2022]"

    allowed_values_to_load = [
        "numerical",
        "numerical_and_coerce",
        "cancelled",
        "all",
        None,
    ]

    dfs = []

    if values_to_load not in allowed_values_to_load:
        raise ValueError(
            f"values_to_load must be one of {allowed_values_to_load}, not {values_to_load}",
        )

    if values_to_load is None:
        values_to_load = "all"

    fn_dict = {
        "coerce": load_non_numerical_values_and_coerce_inequalities,
        "numerical": load_numerical_values,
        "cancelled": load_cancelled,
        "all": load_all_values,
    }

    sources_to_load = [k for k in fn_dict if k in values_to_load]

    if n_rows:
        n_rows_per_fn = int(n_rows / len(sources_to_load))
    else:
        n_rows_per_fn = None

    for k in sources_to_load:  # pylint: disable=invalid-name
        dfs.append(
            fn_dict[k](  # type: ignore
                blood_sample_id=blood_sample_id,
                n_rows=n_rows_per_fn,
                view=view,
            ),
        )

    # Concatenate dfs
    if len(dfs) > 1:
        df = pd.concat(dfs)
    else:
        df = dfs[0]

    return df.reset_index(drop=True).drop_duplicates(
        subset=["dw_ek_borger", "timestamp", "value"],
        keep="first",
    )


@data_loaders.register("hba1c")
def hba1c(
    n_rows: Optional[int] = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU27300",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("scheduled_glc")
def scheduled_glc(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    npu_suffixes = [
        "08550",
        "08551",
        "08552",
        "08553",
        "08554",
        "08555",
        "08556",
        "08557",
        "08558",
        "08559",
        "08560",
        "08561",
        "08562",
        "08563",
        "08564",
        "08565",
        "08566",
        "08567",
        "08893",
        "08894",
        "08895",
        "08896",
        "08897",
        "08898",
        "08899",
        "08900",
        "08901",
        "08902",
        "08903",
        "08904",
        "08905",
        "08906",
        "08907",
        "08908",
        "08909",
        "08910",
        "08911",
        "08912",
        "08913",
        "08914",
        "08915",
        "08916",
    ]

    blood_sample_ids = [f"NPU{suffix}" for suffix in npu_suffixes]

    return blood_sample(
        blood_sample_id=blood_sample_ids,
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("unscheduled_p_glc")
def unscheduled_p_glc(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    npu_suffixes = [
        "02192",
        "21533",
        "21531",
    ]

    dnk_suffixes = ["35842"]

    blood_sample_ids = [f"NPU{suffix}" for suffix in npu_suffixes]
    blood_sample_ids += [f"DNK{suffix}" for suffix in dnk_suffixes]

    return blood_sample(
        blood_sample_id=blood_sample_ids,
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("triglycerides")
def triglycerides(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU04094",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("fasting_triglycerides")
def fasting_triglycerides(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU03620",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("hdl")
def hdl(
    n_rows: Optional[int] = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU01567",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("ldl")
def ldl(
    n_rows: Optional[int] = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id=["NPU01568", "AAB00101"],
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("fasting_ldl")
def fasting_ldl(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id=["NPU10171", "AAB00102"],
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("alat")
def alat(
    n_rows: Optional[int] = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU19651",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("asat")
def asat(
    n_rows: Optional[int] = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU19654",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("lymphocytes")
def lymphocytes(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU02636",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("leukocytes")
def leukocytes(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU02593",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("crp")
def crp(
    n_rows: Optional[int] = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU19748",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("creatinine")
def creatinine(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id=["NPU18016", "ASS00355", "ASS00354"],
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("egfr")
def egfr(
    n_rows: Optional[int] = None, values_to_load: str = "numerical_and_coerce"
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id=["DNK35302", "DNK35131", "AAB00345", "AAB00343"],
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("albumine_creatinine_ratio")
def albumine_creatinine_ratio(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU19661",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("cyp21a2")
def cyp21a2(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU19053",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("cyp2c19")
def cyp2c19(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU19309",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("cyp2c9")
def cyp2c9(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU32095",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("cyp3a5")
def cyp3a5(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU27992",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("cyp2d6")
def cyp2d6(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU19308",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_lithium")
def p_lithium(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU02613",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_clozapine")
def p_clozapine(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU04114",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_olanzapine")
def p_olanzapine(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU09358",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_aripiprazol")
def p_aripiprazol(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU26669",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_risperidone")
def p_risperidone(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU04868",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_paliperidone")
def p_paliperidone(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU18359",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_haloperidol")
def p_haloperidol(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU03937",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_amitriptyline")
def p_amitriptyline(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU01224",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_nortriptyline")
def p_nortriptyline(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU02923",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_clomipramine")
def p_clomipramine(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU01616",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_paracetamol")
def p_paracetamol(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU03024",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_ibuprofen")
def p_ibuprofen(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU08794",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )


@data_loaders.register("p_ethanol")
def p_ethanol(
    n_rows: Optional[int] = None,
    values_to_load: str = "numerical_and_coerce",
) -> pd.DataFrame:
    return blood_sample(
        blood_sample_id="NPU01992",
        n_rows=n_rows,
        values_to_load=values_to_load,
    )
