import pandas as pd
import numpy as np
from typing import Protocol, Optional, Union
from ..portfolio.allocator import PortfolioAllocator

class Strategy(Protocol):
    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        ...

class BacktestEngine:
    """
    Vectorized backtesting engine supporting portfolios, 
    transaction costs, and strict boundary controls.
    """
    def __init__(self, data: Union[pd.DataFrame, any], initial_capital: float = 100000.0, 
                 commission: float = 0.0, slippage: float = 0.0):
        # Normalize data to a pandas DataFrame for consistency
        if hasattr(data, 'close') and isinstance(data.close, pd.DataFrame):
            self.data = data.close
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise TypeError("Data must be a pandas DataFrame or a wrapper with a '.close' DataFrame attribute")
            
        self.data = self.data.sort_index()
        self.initial_capital = initial_capital
        self.commission = commission 
        self.slippage = slippage

    def run(self, strategy: Strategy, allocator: Optional[PortfolioAllocator] = None):
        # Pass the normalized DataFrame (self.data) to the strategy
        signals = strategy.generate_signal(self.data)
        
        if allocator:
            weights = allocator.allocate(signals)
        else:
            active_count = signals.sum(axis=1)
            weights = signals.div(active_count, axis=0).fillna(0)

        execution_weights = weights.shift(1).fillna(0)
        
        # Compute asset returns from normalized data
        asset_returns = self.data.pct_change()

        portfolio_returns = (execution_weights * asset_returns).sum(axis=1)
        
        weight_change = execution_weights.diff().abs().sum(axis=1).fillna(0)
        costs = weight_change * (self.commission + self.slippage)
        
        net_returns = portfolio_returns - costs
        equity_curve = (1 + net_returns).cumprod() * self.initial_capital
        return equity_curve

    def evaluate(self, equity_curve: pd.Series):
        returns = equity_curve.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        max_dd = (equity_curve / equity_curve.cummax() - 1).min()
        return {
            "cumulative_return": (equity_curve.iloc[-1] / self.initial_capital) - 1,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd
        }
