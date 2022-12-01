"""Abstract method that defines a feature cache."""
from abc import ABC, abstractmethod

import pandas as pd

from timeseriesflattener.feature_spec_objects import TemporalSpec


class FeatureCache(ABC):
    @abstractmethod
    def __init__(self, prediction_times_df: pd.DataFrame = None):
        pass

    @abstractmethod
    def feature_exists(self, feature_spec: TemporalSpec) -> bool:
        pass

    @abstractmethod
    def read_feature(self, feature_spec: TemporalSpec) -> pd.DataFrame:
        pass

    @abstractmethod
    def write_feature(self, feature_spec: TemporalSpec, df: pd.DataFrame) -> None:
        pass
