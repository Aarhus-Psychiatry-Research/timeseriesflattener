"""Abstract method that defines a feature cache."""
from abc import ABC, abstractmethod

import pandas as pd

from timeseriesflattener.feature_spec_objects import TemporalSpec


class FeatureCache(ABC):
    @abstractmethod
    def __init__(self, validate: bool = True, prediction_times_df: pd.DataFrame = None):
        """Initialize a feature cache.

        Args:
            validate (bool): Whether to validate the cache. Defaults to True.
            prediction_times_df (pd.DataFrame): Prediction times dataframe. Required for validation. Defaults to None.
        """
        pass

    @abstractmethod
    def feature_exists(self, feature_spec: TemporalSpec, validate: bool = True) -> bool:
        pass

    @abstractmethod
    def get_feature(self, feature_spec: TemporalSpec) -> pd.DataFrame:
        pass

    @abstractmethod
    def write_feature(self, feature_spec: TemporalSpec, df: pd.DataFrame) -> None:
        pass
