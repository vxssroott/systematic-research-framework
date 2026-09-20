import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class FeatureTransformer(ABC):
    """Base class for time-series feature transformations."""
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class ZScoreTransformer(FeatureTransformer):
    """Computes rolling Z-score to normalize features."""
    def __init__(self, window: int = 20):
        self.window = window

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        rolling = data.rolling(window=self.window)
        return (data - rolling.mean()) / rolling.std()

class LogReturnTransformer(FeatureTransformer):
    """Computes log returns for a price series."""
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return np.log(data / data.shift(1))
