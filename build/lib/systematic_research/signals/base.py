import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class Signal(ABC):
    """Base class for quantitative signals."""
    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.Series:
        pass

class MovingAverageCrossover(Signal):
    """Simple baseline signal: Short MA > Long MA."""
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate(self, data: pd.DataFrame) -> pd.Series:
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        return (short_ma > long_ma).astype(int) - (short_ma < long_ma).astype(int)
