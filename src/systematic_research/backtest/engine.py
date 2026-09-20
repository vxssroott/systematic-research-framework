import pandas as pd
import numpy as np
from typing import Protocol, Optional
from ..portfolio.allocator import PortfolioAllocator

class Strategy(Protocol):
    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        ...

class BacktestEngine:
    """
    Vectorized backtesting engine supporting portfolios, 
    transaction costs, and strict boundary controls.
    """
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000.0, 
                 commission: float = 0.0, slippage: float = 0.0):
        self.data = data.sort_index() if hasattr(data, 'sort_index') else data
        self.initial_capital = initial_capital
        self.commission = commission 
        self.slippage = slippage

    def run(self, strategy: Strategy, allocator: Optional[PortfolioAllocator] = None):
        # generate_signal now returns a DataFrame of signals per asset
        signals = strategy.generate_signal(self.data)
        
        if allocator:
            # Map signals to weights using the allocator
            weights = allocator.allocate(signals)
        else:
            # Default to equal weight of active signals
            active_count = signals.sum(axis=1)
            weights = signals.div(active_count, axis=0).fillna(0)

        # Prevent look-ahead bias: Weights at t are applied to returns at t+1
        execution_weights = weights.shift(1).fillna(0)
        
        # Asset returns
        if isinstance(self.data, pd.DataFrame) and 'close' in self.data.columns:
             # Handle single-asset case if 'close' is a column
             asset_returns = self.data['close'].pct_change().to_frame()
        elif hasattr(self.data, 'close'):
             # Handle the wrapper case used in examples
             if isinstance(self.data.close, pd.DataFrame):
                 asset_returns = self.data.close.pct_change()
             else:
                 asset_returns = self.data.close.pct_change().to_frame()
        else:
            # Fallback: assume the data itself is the price DF
            asset_returns = self.data.pct_change()

        # Portfolio returns: sum(weight_i * return_i)
        portfolio_returns = (execution_weights * asset_returns).sum(axis=1)
        
        # Transaction costs: apply when weights change
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
            "cumulative_return": (equity_curve[-1] / self.initial_capital) - 1,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd
        }
