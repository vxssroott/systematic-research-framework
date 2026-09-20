import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

class PortfolioAllocator(ABC):
    """Base class for portfolio weight allocation strategies."""
    @abstractmethod
    def allocate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Returns a DataFrame of weights per asset over time."""
        pass

class EqualWeightAllocator(PortfolioAllocator):
    """Allocates weights equally across all available assets."""
    def allocate(self, data: pd.DataFrame) -> pd.DataFrame:
        # Assume data is a DataFrame of signals or returns
        n_assets = data.shape[1]
        weights = pd.DataFrame(1.0 / n_assets, index=data.index, columns=data.columns)
        return weights

class RiskParityAllocator(PortfolioAllocator):
    """
    Implements a basic Inverse-Volatility Risk Parity allocation.
    Weights are proportional to the inverse of the asset's volatility.
    """
    def __init__(self, window: int = 21):
        self.window = window

    def allocate(self, returns: pd.DataFrame) -> pd.DataFrame:
        # Compute rolling volatility
        vol = returns.rolling(window=self.window).std()
        
        # Inverse volatility
        inv_vol = 1.0 / vol
        
        # Normalize weights to sum to 1 across assets for each timestamp
        row_sum = inv_vol.sum(axis=1)
        weights = inv_vol.div(row_sum, axis=0)
        
        return weights.fillna(0)
