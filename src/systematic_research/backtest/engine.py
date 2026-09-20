import pandas as pd
import numpy as np
class BacktestEngine:
    def __init__(self, data):
        self.data = data
    def run(self, strategy):
        signals = strategy.generate_signal(self.data)
        trades = signals.shift(1).fillna(0)
        returns = trades * self.data['close'].pct_change()
        return (1 + returns).cumprod()