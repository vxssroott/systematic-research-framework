import pandas as pd
import numpy as np
from typing import Union

def compute_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Computes rolling annualized volatility."""
    return returns.rolling(window=window).std() * np.sqrt(252)

def compute_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Computes historical Value at Risk (VaR)."""
    return np.percentile(returns, (1 - confidence) * 100)

def compute_conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Computes Conditional Value at Risk (CVaR) / Expected Shortfall.
    CVaR is the average loss given that the loss exceeds the VaR threshold.
    """
    var = compute_value_at_risk(returns, confidence)
    tail_losses = returns[returns <= var]
    return tail_losses.mean() if not tail_losses.empty else var

def compute_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Computes the Omega Ratio.
    The ratio of probability-weighted gains over losses relative to a threshold.
    """
    gains = returns[returns > threshold].sum()
    losses = abs(returns[returns < threshold].sum())
    return gains / losses if losses != 0 else np.nan
