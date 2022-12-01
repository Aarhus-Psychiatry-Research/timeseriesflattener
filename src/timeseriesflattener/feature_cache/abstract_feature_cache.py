"""Abstract method that defines a feature cache."""
from abc import ABCMeta, abstractmethod
from typing import Optional

import pandas as pd

from timeseriesflattener.feature_spec_objects import TemporalSpec


class FeatureCache(metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        *args,
        prediction_times_df: Optional[pd.DataFrame] = None,
        pred_time_uuid_col_name: str = "pred_time_uuid",
    ):
        """Initialize a FeatureCache.

        Args:
            prediction_times_df (Optional[pd.DataFrame], optional): DataFrame containing prediction times.
                Must be set at some point, but doesn't have to be set at init.
                Useful when e.g. used as a component in TimeseriesFlattener, which already knows the prediction_times_df and can set it as a pointer. Defaults to None.
            pred_time_uuid_col_name (str, optional): Name of column containing prediction time uuids. Defaults to "pred_time_uuid".
        """
        self.prediction_times_df = prediction_times_df
        self.pred_time_uuid_col_name = pred_time_uuid_col_name

    @abstractmethod
    def feature_exists(self, feature_spec: TemporalSpec) -> bool:
        pass

    @abstractmethod
    def read_feature(self, feature_spec: TemporalSpec) -> pd.DataFrame:
        pass

    @abstractmethod
    def write_feature(self, feature_spec: TemporalSpec, df: pd.DataFrame) -> None:
        pass
