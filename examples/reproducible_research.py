import pandas as pd
import numpy as np
from systematic_research.backtest.engine import BacktestEngine
from systematic_research.portfolio.allocator import RiskParityAllocator
from systematic_research.signals.base import MovingAverageCrossover
from systematic_research.risk.analytics import compute_conditional_var, compute_omega_ratio

class MultiAssetMAStrategy:
    """A simple multi-asset MA crossover strategy."""
    def __init__(self):
        self.signal_gen = MovingAverageCrossover()

    def generate_signal(self, data):
        # Assume data is a DataFrame of prices for multiple assets
        # Return a DataFrame of signals
        signals = pd.DataFrame(index=data.index)
        for col in data.columns:
            # Create a temporary DF for the signal generator
            temp_df = data[[col]].rename(columns={col: 'close'})
            signals[col] = self.signal_gen.generate(temp_df)
        return signals

def run_reproducible_example():
    print("Generating synthetic multi-asset data...")
    dates = pd.date_range("2023-01-01", periods=500)
    assets = ['ASSET_A', 'ASSET_B', 'ASSET_C']
    
    # Create synthetic prices with different volatilities
    np.random.seed(42)
    data_dict = {
        'ASSET_A': 100 * (1 + np.random.normal(0.0005, 0.01, 500).cumprod()),
        'ASSET_B': 100 * (1 + np.random.normal(0.0003, 0.02, 500).cumprod()),
        'ASSET_C': 100 * (1 + np.random.normal(0.0008, 0.005, 500).cumprod()),
    }
    df_prices = pd.DataFrame(data_dict, index=dates)
    
    # We wrap the price DF in a way the engine expects (data['close'] as a DF)
    # In the current engine, we can just pass the price DF as the 'data' and 
    # modify the engine slightly or pass it as a dictionary.
    # Let's use a helper class to simulate the data structure.
    class PriceData:
        def __init__(self, df):
            self.close = df
            self.index = df.index
        def sort_index(self):
            return self
            
    data_wrapper = PriceData(df_prices)
    
    # 1. Initialize Engine & Strategy
    engine = BacktestEngine(data_wrapper, commission=0.0001)
    strategy = MultiAssetMAStrategy()
    allocator = RiskParityAllocator()
    
    # 2. Run Backtest
    equity_curve = engine.run(strategy, allocator=allocator)
    
    # 3. Evaluate
    metrics = engine.evaluate(equity_curve)
    returns = equity_curve.pct_change().dropna()
    cvar = compute_conditional_var(returns)
    omega = compute_omega_ratio(returns)
    
    print("\n--- Research Results ---")
    print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"CVaR (95%): {cvar:.4f}")
    print(f"Omega Ratio: {omega:.2f}")

if __name__ == "__main__":
    run_reproducible_example()
