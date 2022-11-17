"""Example of how to load demographic features."""
import pandas as pd

from psycop_feature_generation.loaders.raw.load_structured_sfi import (
    bmi,
    height_in_cm,
    weight_in_kg,
)
from psycop_feature_generation.loaders.raw.sql_load import sql_load

if __name__ == "__main__":
    df = sql_load(
        query="SELECT * FROM [fct].[FOR_SFI_vaegt_hoejde_BMI_psyk_somatik_inkl_2021]",
        database="USR_PS_FORSK",
        chunksize=None,
        n_rows=1_000,
    )[["aktivitetstypenavn", "elementledetekst", "numelementvaerdi", "elementvaerdi"]]

    df_bmi = bmi(n_rows=1_000)
    df_height = height_in_cm(n_rows=100_000)
    df_weight = weight_in_kg(n_rows=100_000)


def unique_and_percentage(series: pd.Series) -> pd.Series:
    """Return unique values and their percentage of the total number of values
    in the series.

    Args:
        series (pd.Series): Series to get unique values and percentage of.

    Returns:
        pd.Series: Series with unique values as index and percentage as values.
    """
    unique_values = series.value_counts(normalize=True)
    return unique_values
