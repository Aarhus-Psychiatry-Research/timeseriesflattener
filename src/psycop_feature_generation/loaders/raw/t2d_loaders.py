from psycopmlutils.sql.loader import sql_load

from psycop_feature_generation.utils import data_loaders


@data_loaders.register("timestamp_exclusion")
def timestamp_exclusion():
    """Loads timestamps for the broad definition of diabetes used for wash-in.

    See R files for details.
    """
    timestamp_any_diabetes = sql_load(
        query="SELECT * FROM [fct].[psycop_t2d_first_diabetes_any]",
        format_timestamp_cols_to_datetime=False,
    )[["dw_ek_borger", "datotid_first_diabetes_any"]]

    timestamp_any_diabetes = timestamp_any_diabetes.rename(
        columns={"datotid_first_diabetes_any": "timestamp"},
    )

    return timestamp_any_diabetes
