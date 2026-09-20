import pandas as pd
import numpy as np

def compute_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Computes rolling annualized volatility."""
    return returns.rolling(window=window).std() * np.sqrt(252)

def compute_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Computes historical Value at Risk (VaR)."""
    return np.percentile(returns, (1 - confidence) * 100)
