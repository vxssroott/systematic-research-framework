import pandas as pd
from typing import List
from .engine import BacktestEngine

class WalkForwardEvaluator:
    """
    Splits data into expanding windows to evaluate strategy robustness.
    """
    def __init__(self, data: pd.DataFrame, train_size: int, test_size: int):
        self.data = data
        self.train_size = train_size
        self.test_size = test_size

    def evaluate(self, strategy):
        results = []
        for i in range(self.train_size, len(self.data) - self.test_size, self.test_size):
            train = self.data.iloc[:i]
            test = self.data.iloc[i : i + self.test_size]
            
            # In a real scenario, we would optimize parameters on 'train' here
            engine = BacktestEngine(test)
            curve = engine.run(strategy)
            results.append(engine.evaluate(curve))
            
        return results
