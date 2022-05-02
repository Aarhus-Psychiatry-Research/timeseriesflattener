from pathlib import Path
from typing import List, Union

import catalogue
import pandas as pd
from psycopmlutils.loaders.sql_load import sql_load
from psycopmlutils.utils import data_loaders
from wasabi import msg


class LoadDiagnoses:
    def aggregate_from_physical_visits(
        icd_codes: List[str],
        output_col_name: str,
        wildcard_icd_10_end: bool = False,
    ) -> pd.DataFrame:
        """Load all diagnoses matching any icd_code in icd_codes. Create output_col_name and set to 1.

        Args:
            icd_codes (List[str]): List of icd_codes.
            output_col_name (str): Output column name
            wildcard_icd_10_end (bool, optional): Whether to match on icd_codes* or icd_codes. Defaults to False.

        Returns:
            pd.DataFrame
        """
        print_str = f"diagnoses matching any of {icd_codes}"
        msg.info(f"Loading {print_str}")

        diagnoses_source_table_info = {
            "lpr3": {
                "fct": "FOR_LPR3kontakter_psyk_somatik_inkl_2021",
                "source_timestamp_col_name": "datotid_lpr3kontaktstart",
            },
            "lpr2_inpatient": {
                "fct": "FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021",
                "source_timestamp_col_name": "datotid_indlaeggelse",
            },
            "lpr2_outpatient": {
                "fct": "FOR_besoeg_psyk_somatik_LPR2_inkl_2021",
                "source_timestamp_col_name": "datotid_start",
            },
        }

        # Using ._load is faster than from_physical_visits since it can process all icd_codes in the SQL request at once,
        # rather than processing one at a time and aggregating.
        dfs = [
            LoadDiagnoses._load(
                icd_code=icd_codes,
                output_col_name=output_col_name,
                wildcard_icd_10_end=wildcard_icd_10_end,
                **kwargs,
            )
            for source_name, kwargs in diagnoses_source_table_info.items()
        ]

        df = pd.concat(dfs)

        msg.good(f"Loaded {print_str}")
        return df.reset_index(drop=True)

    def from_physical_visits(
        icd_code: str,
        output_col_name: str = None,
        wildcard_icd_10_end: bool = False,
    ) -> pd.DataFrame:
        """Load diagnoses from all physical visits. If icd_code is a list, will aggregate as one column (e.g. ["E780", "E785"] into a ypercholesterolemia column).

        Args:
            icd_code (str): Substring to match diagnoses for. Matches any diagnoses, whether a-diagnosis, b-diagnosis etc.
            output_col_name (str, optional): Name of new column string. Defaults to None.

        Returns:
            pd.DataFrame
        """
        print_str = f"diagnoses matching ICD-code {icd_code}"
        msg.info(f"Loading {print_str}")

        diagnoses_source_table_info = {
            "lpr3": {
                "fct": "FOR_LPR3kontakter_psyk_somatik_inkl_2021",
                "source_timestamp_col_name": "datotid_lpr3kontaktstart",
            },
            "lpr2_inpatient": {
                "fct": "FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021",
                "source_timestamp_col_name": "datotid_indlaeggelse",
            },
            "lpr2_outpatient": {
                "fct": "FOR_besoeg_psyk_somatik_LPR2_inkl_2021",
                "source_timestamp_col_name": "datotid_start",
            },
        }

        dfs = [
            LoadDiagnoses._load(
                icd_code=icd_code,
                output_col_name=output_col_name,
                wildcard_icd_10_end=wildcard_icd_10_end,
                **kwargs,
            )
            for source_name, kwargs in diagnoses_source_table_info.items()
        ]

        df = pd.concat(dfs)

        msg.good(f"Loaded {print_str}")
        return df.reset_index(drop=True)

    def _load(
        icd_code: Union[List[str], str],
        source_timestamp_col_name: str,
        fct: str,
        output_col_name: str = None,
        wildcard_icd_10_end: bool = True,
    ) -> pd.DataFrame:
        """Load the visits that have diagnoses that match icd_code from the beginning of their adiagnosekode string.
        Aggregates all that match.

        Args:
            icd_code (Union[List[str], str]): Substring(s) to match diagnoses for. Matches any diagnoses, whether a-diagnosis, b-diagnosis etc.
            source_timestamp_col_name (str): Name of the timestamp column in the SQL table.
            view (str): Which view to use, e.g. "FOR_Medicin_ordineret_inkl_2021_feb2022"
            output_col_name (str, optional): Name of new column string. Defaults to None.
            wildcard_icd_10_end (bool, optional): Whether to match on icd_code*. Defaults to true.

        Returns:
            pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and output_col_name = 1
        """
        fct = f"[{fct}]"

        # Add a % at the end of the SQL match as a wildcard, so e.g. F20 matches F200.
        sql_ending = "%" if wildcard_icd_10_end else ""

        if isinstance(icd_code, list):
            match_col_sql_strings = [
                f"lower(diagnosegruppestreng) LIKE lower('%{diag_str}{sql_ending}')"
                for diag_str in icd_code
            ]

            match_col_sql_str = " OR ".join(match_col_sql_strings)
        else:
            match_col_sql_str = (
                f"lower(diagnosegruppestreng) LIKE lower('%{icd_code}{sql_ending})'"
            )

        sql = f"SELECT dw_ek_borger, {source_timestamp_col_name}, diagnosegruppestreng FROM [fct].{fct} WHERE ({match_col_sql_str})"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        if output_col_name is None:
            output_col_name = icd_code

        df[output_col_name] = 1

        df.drop(["diagnosegruppestreng"], axis="columns", inplace=True)

        return df.rename(
            columns={
                source_timestamp_col_name: "timestamp",
            }
        )

    @data_loaders.register("t2d_times")
    def t2d_times():
        msg.info("Loading t2d event times")

        full_csv_path = Path(
            r"C:\Users\adminmanber\Desktop\manber-t2d\csv\first_t2d_diagnosis.csv"
        )

        df = pd.read_csv(str(full_csv_path))
        df = df[["dw_ek_borger", "datotid_first_t2d_diagnosis"]]
        df["value"] = 1

        df.rename(columns={"datotid_first_t2d_diagnosis": "timestamp"}, inplace=True)

        msg.good("Finished loading t2d event times")
        return df.reset_index(drop=True)

    @data_loaders.register("essential_hypertension")
    def essential_hypertension():
        return LoadDiagnoses.from_physical_visits(
            icd_code="I109",
            wildcard_icd_10_end=False,
            output_col_name="essential_hypertension",
        )

    @data_loaders.register("hyperlipidemia")
    def hyperlipidemia():
        return LoadDiagnoses.from_physical_visits(
            icd_code=[
                "E780",
                "E785",
            ],  # Only these two, as the others are exceedingly rare
            output_col_name="hyperlipidemia",
            wildcard_icd_10_end=False,
        )

    @data_loaders.register("liverdisease_unspecified")
    def liverdisease_unspecified():
        return LoadDiagnoses.from_physical_visits(
            icd_code="K769",
            wildcard_icd_10_end=False,
            output_col_name="liverdisease_unspecified",
        )

    @data_loaders.register("polycystic_ovarian_syndrome")
    def polycystic_ovarian_syndrome():
        return LoadDiagnoses.from_physical_visits(
            icd_code="E282",
            wildcard_icd_10_end=False,
            output_col_name="polycystic_ovarian_syndrome",
        )

    @data_loaders.register("sleep_apnea")
    def sleep_apnea():
        return LoadDiagnoses.from_physical_visits(
            icd_code=["G473", "G4732"],
            wildcard_icd_10_end=False,
            output_col_name="sleep_apnea",
        )

    @data_loaders.register("sleep_problems_unspecified")
    def sleep_problems_unspecified():
        return LoadDiagnoses.from_physical_visits(
            icd_code="G479",
            wildcard_icd_10_end=False,
            output_col_name="sleep_problems_unspecified",
        )
