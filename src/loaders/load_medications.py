import pandas as pd
from wasabi import msg

from loaders.sql_load import sql_load


class LoadMedications:
    def aggregate_medications(
        output_col_name: str, atc_code_prefixes: list
    ) -> pd.DataFrame:
        """Aggregate multiple blood_sample_ids (typically NPU-codes) into one column.

        Args:
            output_col_name (str): Name for new column.
            atc_codes (list): List of atc_codes.

        Returns:
            pd.DataFrame
        """
        dfs = [
            LoadMedications.load(
                blood_sample_id=f"{id}", output_col_name=output_col_name
            )
            for id in atc_code_prefixes
        ]

        return pd.concat(dfs, axis=0).reset_index(drop=True)

    def load(
        atc_code: str,
        output_col_name: str = None,
        load_prescribed: bool = True,
        load_administered: bool = True,
        wildcard_at_end: bool = True,
    ) -> pd.DataFrame:
        """Load medications. Aggregates prescribed/administered if both true. If wildcard_atc_at_end, match from atc_code*.
        Aggregates all that match. Beware that data is incomplete prior to sep. 2016 for prescribed medications.

        Args:
            atc_code (str): ATC-code prefix to load. Matches atc_code_prefix*. Aggregates all.
            output_col_name (str, optional): Name of output_col_name. Contains 1 if atc_code matches atc_code_prefix, 0 if not.Defaults to {atc_code_prefix}_value.
            load_prescribed (bool, optional): Whether to load prescriptions. Defaults to True. Beware incomplete until sep 2016.
            load_administered (bool, optional): Whether to load administrations. Defaults to True.
            wildcard_atc_at_end (bool, optional): Whether to match on atc_code* or atc_code.

        Returns:
            pd.DataFrame: Cols: dw_ek_borger, timestamp, {atc_code_prefix}_value = 1
        """
        print_str = f"medications matching NPU-code {atc_code}"
        msg.info(f"Loading {print_str}")

        if load_prescribed:
            msg.warn(
                "Beware, there are missing prescriptions until september 2019. Hereafter, data is complete."
            )

        df = pd.DataFrame()

        if load_prescribed:
            df_medication_prescribed = LoadMedications._load_one_source(
                atc_code=atc_code,
                source_timestamp_col_name="datotid_ordinationstart",
                view="FOR_Medicin_ordineret_inkl_2021_feb2022",
                output_col_name=output_col_name,
                wildcard_atc_at_end=wildcard_at_end,
            )
            df = pd.concat([df, df_medication_prescribed])

        if load_administered:
            df_medication_administered = LoadMedications._load_one_source(
                atc_code=atc_code,
                source_timestamp_col_name="datotid_administration_start",
                view="FOR_Medicin_administreret_inkl_2021_feb2022",
                output_col_name=output_col_name,
                wildcard_atc_at_end=wildcard_at_end,
            )
            df = pd.concat([df, df_medication_administered])

        if output_col_name == None:
            output_col_name = atc_code

        df.rename(
            columns={
                atc_code: f"{output_col_name}_value",
            },
            inplace=True,
        )

        msg.good(f"Loaded {print_str}")
        return df.reset_index(drop=True)

    def _load_one_source(
        atc_code: str,
        source_timestamp_col_name: str,
        view: str,
        output_col_name: str = None,
        wildcard_atc_at_end: bool = False,
    ) -> pd.DataFrame:
        """Load the prescribed medications that match atc. If wildcard_atc_at_end, match from atc_code*.
        Aggregates all that match. Beware that data is incomplete prior to sep. 2016 for prescribed medications.

        Args:
            atc_code (str): ATC string to match on.
            source_timestamp_col_name (str): Name of the timestamp column in the SQL table.
            view (str): Which view to use, e.g. "FOR_Medicin_ordineret_inkl_2021_feb2022"
            output_col_name (str, optional): Name of new column string. Defaults to None.
            wildcard_atc_at_end (bool, optional): Whether to match on atc_code* or atc_code.

        Returns:
            pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and output_col_name = 1
        """

        if wildcard_atc_at_end:
            end_of_sql = "%"
        else:
            end_of_sql = ""

        view = f"[{view}]"
        sql = f"SELECT dw_ek_borger, {source_timestamp_col_name}, atc FROM [fct].{view} WHERE (lower(atc)) LIKE lower('{atc_code}{end_of_sql}')"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        if output_col_name is None:
            output_col_name = atc_code

        df[output_col_name] = 1

        df.drop(["atc"], axis="columns", inplace=True)

        return df.rename(
            columns={
                source_timestamp_col_name: "timestamp",
            }
        )
