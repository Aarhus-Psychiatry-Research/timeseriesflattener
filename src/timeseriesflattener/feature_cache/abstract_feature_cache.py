"""Base method that defines a feature cache."""
from abc import ABCMeta, abstractmethod
from typing import Any

import pandas as pd

from timeseriesflattener.feature_spec_objects import TemporalSpec


class FeatureCache(metaclass=ABCMeta):
    """Base class that defines a feature cache."""

    @abstractmethod
    def __init__(
        self,
        *args: Any,
        prediction_times_df: pd.DataFrame,
        pred_time_uuid_col_name: str = "pred_time_uuid",
        entity_id_col_name: str = "entity_id",
        timestamp_col_name: str = "timestamp",
    ):
        """Initialize a FeatureCache.

        Args:
            *args: Arguments to pass to the subclass.
            prediction_times_df (Optional[pd.DataFrame], optional): DataFrame containing prediction times.
                Must be set at some point, but doesn't have to be set at init.
                Useful when e.g. used as a component in TimeseriesFlattener, which already knows the prediction_times_df and can set it as a pointer during initialization. Defaults to None. Defaults to None.
            pred_time_uuid_col_name (str, optional): Name of column containing prediction time uuids.
            entity_id_col_name (str, optional): Name of column containing entity ids. Defaults to "entity_id".
            timestamp_col_name (str, optional): Name of column containing timestamps. Defaults to "timestamp".
            Defaults to "pred_time_uuid".
        """
        self.prediction_times_df = prediction_times_df
        self.pred_time_uuid_col_name = pred_time_uuid_col_name

    @abstractmethod
    def feature_exists(self, feature_spec: TemporalSpec) -> bool:
        """Check if feature exists in cache."""

    @abstractmethod
    def read_feature(self, feature_spec: TemporalSpec) -> pd.DataFrame:
        """Read feature from cache."""

    @abstractmethod
    def write_feature(self, feature_spec: TemporalSpec, df: pd.DataFrame) -> None:
        """Write feature to cache."""
