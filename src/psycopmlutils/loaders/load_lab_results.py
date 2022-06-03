import catalogue
import pandas as pd
from psycopmlutils.loaders.sql_load import sql_load
from psycopmlutils.utils import data_loaders
from wasabi import msg


class LoadLabResults:
    def blood_sample(blood_sample_id: str) -> pd.DataFrame:
        """Load a blood sample.

        Args:
            blood_sample_id (str): The blood_sample_id, typically an NPU code.

        Returns:
            pd.DataFrame
        """
        print_str = f"blood samples matching NPU-code {blood_sample_id}"
        # msg.info(f"Loading {print_str}")

        view = "[FOR_labka_alle_blodprover_inkl_2021_feb2022]"
        sql = f"SELECT dw_ek_borger, datotid_sidstesvar, numerisksvar FROM [fct].{view} WHERE NPUkode = '{blood_sample_id}'"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        df.rename(
            columns={"datotid_sidstesvar": "timestamp", "numerisksvar": "value"},
            inplace=True,
        )

        # msg.good(f"Loaded {print_str}")
        return df.reset_index(drop=True)

    def _aggregate_blood_samples(blood_sample_ids: list) -> pd.DataFrame:
        """Aggregate multiple blood_sample_ids (typically NPU-codes) into one column.

        Args:
            blood_sample_ids (list): List of blood_sample_id, typically an NPU-codes.

        Returns:
            pd.DataFrame
        """
        dfs = [
            LoadLabResults.blood_sample(blood_sample_id=f"{id}")
            for id in blood_sample_ids
        ]

        return pd.concat(dfs, axis=0).reset_index(drop=True)

    @data_loaders.register("hba1c")
    def hba1c():
        return LoadLabResults.blood_sample(blood_sample_id="NPU27300")

    @data_loaders.register("scheduled_glc")
    def scheduled_glc():
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

        return LoadLabResults._aggregate_blood_samples(
            blood_sample_ids=blood_sample_ids,
        )

    @data_loaders.register("unscheduled_p_glc")
    def unscheduled_p_glc():
        npu_suffixes = [
            "02192",
            "21533",
            "21531",
        ]

        dnk_suffixes = ["35842"]

        blood_sample_ids = [f"NPU{suffix}" for suffix in npu_suffixes]
        blood_sample_ids += [f"DNK{suffix}" for suffix in dnk_suffixes]

        return LoadLabResults._aggregate_blood_samples(
            blood_sample_ids=blood_sample_ids
        )

    @data_loaders.register("triglycerides")
    def triglycerides():
        return LoadLabResults.blood_sample(blood_sample_id="NPU04094")

    @data_loaders.register("fasting_triglycerides")
    def fasting_triglycerides():
        return LoadLabResults.blood_sample(blood_sample_id="NPU03620")

    @data_loaders.register("hdl")
    def hdl():
        return LoadLabResults.blood_sample(blood_sample_id="NPU01567")

    @data_loaders.register("ldl")
    def ldl():
        return LoadLabResults._aggregate_blood_samples(
            blood_sample_ids=["NPU01568", "AAB00101"],
        )

    @data_loaders.register("fasting_ldl")
    def fasting_ldl():
        return LoadLabResults._aggregate_blood_samples(
            blood_sample_ids=["NPU10171", "AAB00102"]
        )

    @data_loaders.register("alat")
    def alat():
        return LoadLabResults.blood_sample(blood_sample_id="NPU19651")

    @data_loaders.register("asat")
    def asat():
        return LoadLabResults.blood_sample(blood_sample_id="NPU19654")

    @data_loaders.register("lymphocytes")
    def lymphocytes():
        return LoadLabResults.blood_sample(blood_sample_id="NPU02636")

    @data_loaders.register("leukocytes")
    def leukocytes():
        return LoadLabResults.blood_sample(blood_sample_id="NPU02593")

    @data_loaders.register("crp")
    def crp():
        return LoadLabResults.blood_sample(blood_sample_id="NPU19748")

    @data_loaders.register("creatinine")
    def creatinine():
        return LoadLabResults._aggregate_blood_samples(
            blood_sample_ids=["NPU18016", "ASS00355", "ASS00354"]
        )

    @data_loaders.register("egfr")
    def egfr():
        return LoadLabResults._aggregate_blood_samples(
            blood_sample_ids=["DNK35302", "DNK35131", "AAB00345", "AAB00343"]
        )

    @data_loaders.register("albumine_creatinine_ratio")
    def albumine_creatinine_ratio():
        return LoadLabResults.blood_sample(blood_sample_id="NPU19661")
