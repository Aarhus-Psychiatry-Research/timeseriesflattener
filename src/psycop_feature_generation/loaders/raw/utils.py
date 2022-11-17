"""Example of."""

from typing import Optional, Union

import pandas as pd

from psycop_feature_generation.loaders.raw.sql_load import sql_load


def str_to_sql_match_logic(
    codes_to_match: str,
    code_col_name: str,
    wildcard_code: bool,
    load_diagnoses: bool,
):
    """Generate SQL match logic from a single string."""
    if wildcard_code:
        match_col_sql_str = f"lower({code_col_name}) LIKE '%{codes_to_match.lower()}%'"
    else:
        match_col_sql_str = f"lower({code_col_name}) LIKE '%{codes_to_match.lower()}'"

        if load_diagnoses:
            match_col_sql_str += (
                f" OR lower({code_col_name}) LIKE '%{codes_to_match.lower()}#%'"
            )

    return match_col_sql_str


def list_to_sql_logic(
    codes_to_match: list[str],
    load_diagnoses: bool,
    code_col_name: str,
    wildcard_code: bool,
):
    """Generate SQL match logic from a list of strings."""
    match_col_sql_strings = []

    for code_str in codes_to_match:
        if wildcard_code:
            match_col_sql_strings.append(
                f"lower({code_col_name}) LIKE '%{code_str.lower()}%'",
            )
        else:
            # If the string is at the end of diagnosegruppestreng, it doesn't end with a hashtag
            match_col_sql_strings.append(
                f"lower({code_col_name}) LIKE '%{code_str.lower()}'",
            )

            if load_diagnoses:
                # If the string is at the beginning of diagnosegruppestreng, it doesn't start with a hashtag
                match_col_sql_strings.append(
                    f"lower({code_col_name}) LIKE '{code_str.lower()}%'",
                )

    return " OR ".join(match_col_sql_strings)


def load_from_codes(
    codes_to_match: Union[list[str], str],
    load_diagnoses: bool,
    code_col_name: str,
    source_timestamp_col_name: str,
    view: str,
    output_col_name: Optional[str] = None,
    wildcard_code: bool = True,
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the visits that have diagnoses that match icd_code or atc code from
    the beginning of their adiagnosekode or atc code string. Aggregates all
    that match.

    Args:
        codes_to_match (Union[list[str], str]): Substring(s) to match diagnoses or medications for.
            Diagnoses: Matches any diagnoses, whether a-diagnosis, b-diagnosis.
            Both: If a list is passed, will count as a match if any of the icd_codes or at codes in the list match.
        load_diagnoses (bool): Determines which mathing logic is employed. If True, will load diagnoses. If False, will load medications.
            Diagnoses must be able to split a string like this:
                A:DF431#+:ALFC3#B:DF329
            Which means that if wildcard_code is False, we must match on *icd_code# or *icd_code followed by nothing. If it's true, we can match on *icd_code*.
        code_col_name (str): Name of column containing either diagnosis (icd) or medication (atc) codes.
            Takes either 'diagnosegruppestreng' or 'atc' as input.
        source_timestamp_col_name (str): Name of the timestamp column in the SQL
            view.
        view (str): Name of the SQL view to load from.
        output_col_name (str, optional): Name of new column string. Defaults to
            None.
        wildcard_code (bool, optional): Whether to match on icd_code* / atc_code*.
            Defaults to true.
        n_rows: Number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and
            output_col_name = 1
    """
    fct = f"[{view}]"

    if isinstance(codes_to_match, list) and len(codes_to_match) > 1:
        match_col_sql_str = list_to_sql_logic(
            codes_to_match=codes_to_match,
            load_diagnoses=load_diagnoses,
            code_col_name=code_col_name,
            wildcard_code=wildcard_code,
        )
    elif isinstance(codes_to_match, str):
        match_col_sql_str = str_to_sql_match_logic(
            codes_to_match=codes_to_match,
            code_col_name=code_col_name,
            wildcard_code=wildcard_code,
            load_diagnoses=load_diagnoses,
        )
    else:
        raise ValueError("codes_to_match must be either a list or a string.")

    sql = (
        f"SELECT dw_ek_borger, {source_timestamp_col_name}, {code_col_name}"
        + f" FROM [fct].{fct} WHERE {source_timestamp_col_name} IS NOT NULL AND ({match_col_sql_str})"
    )

    df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)

    if output_col_name is None:
        if isinstance(codes_to_match, list):
            output_col_name = "_".join(codes_to_match)
        else:
            output_col_name = codes_to_match

    df[output_col_name] = 1

    df.drop([f"{code_col_name}"], axis="columns", inplace=True)

    return df.rename(
        columns={
            source_timestamp_col_name: "timestamp",
        },
    )
