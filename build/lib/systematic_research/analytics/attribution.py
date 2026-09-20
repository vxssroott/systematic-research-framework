import pandas as pd
import numpy as np

def compute_attribution(portfolio_returns: pd.Series, asset_returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the contribution of each asset to the total portfolio return.
    Contribution = weight_t-1 * return_t
    """
    # Shift weights to align t-1 weights with t returns (prevent look-ahead)
    shifted_weights = weights.shift(1).fillna(0)
    
    # Element-wise multiplication of weights and returns
    contributions = shifted_weights * asset_returns
    
    return contributions
