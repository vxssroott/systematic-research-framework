import pytest
import pandas as pd
import numpy as np
from systematic_research.backtest.engine import BacktestEngine

class MockStrategy:
    def generate_signal(self, data):
        return pd.Series(1, index=data.index)

def test_backtest_equity_growth():
    # Create synthetic uptrend data
    dates = pd.date_range("2023-01-01", periods=100)
    df = pd.DataFrame({"close": np.linspace(100, 200, 100), "volume": 1000}, index=dates)
    
    engine = BacktestEngine(df)
    curve = engine.run(MockStrategy())
    
    assert curve[-1] > 100000.0  # Should grow in an uptrend
