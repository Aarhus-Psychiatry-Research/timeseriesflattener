import pandas as pd

from psycop_feature_generation.utils import df_contains_duplicates


class ValidateInitFlattenedDataset:
    def _check_timestamp_col_type(self):
        """Check that the timestamp column is of type datetime."""
        timestamp_col_type = type(self.df[self.timestamp_col_name][0]).__name__

        if timestamp_col_type not in ["Timestamp"]:
            try:
                self.df[self.timestamp_col_name] = pd.to_datetime(
                    self.df[self.timestamp_col_name],
                )
            except Exception as exc:
                raise ValueError(
                    f"prediction_times_df: {self.timestamp_col_name} is of type {timestamp_col_type}, and could not be converted to 'Timestamp' from Pandas. Will cause problems. Convert before initialising FlattenedDataset. More info: {exc}",
                ) from exc

    def _check_for_duplicate_rows(self):
        """Check that there are no duplicate rows in the initial dataframe."""
        if df_contains_duplicates(
            df=self.df,
            col_subset=[self.id_col_name, self.timestamp_col_name],
        ):
            raise ValueError(
                "Duplicate patient/timestamp combinations in prediction_times_df, aborting",
            )

    def _check_that_timestamp_and_id_columns_exist(self):
        """Check that the required columns are present in the initial
        dataframe."""

        for col_name in (self.timestamp_col_name, self.id_col_name):
            if col_name not in self.df.columns:
                raise ValueError(
                    f"{col_name} does not exist in prediction_times_df, change the df or set another argument",
                )

    def __init__(self, df: pd.DataFrame, timestamp_col_name: str, id_col_name: str):
        self.df = df
        self.timestamp_col_name = timestamp_col_name
        self.id_col_name = id_col_name

    def validate_dataset(self):
        self._check_that_timestamp_and_id_columns_exist()
        self._check_for_duplicate_rows()
        self._check_timestamp_col_type()
