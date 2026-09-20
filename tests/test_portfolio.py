import pytest
import pandas as pd
import numpy as np
from systematic_research.risk.analytics import compute_conditional_var, compute_omega_ratio
from systematic_research.portfolio.allocator import EqualWeightAllocator, RiskParityAllocator

def test_cvar_numerical_invariant():
    # For a symmetric distribution, VaR and CVaR should be negative
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    cvar = compute_conditional_var(returns)
    assert cvar < 0

def test_omega_ratio_positive_drift():
    # Returns with positive drift should have Omega > 1
    returns = pd.Series(np.random.normal(0.01, 0.01, 1000))
    omega = compute_omega_ratio(returns)
    assert omega > 1

def test_equal_weight_sum():
    data = pd.DataFrame(np.random.rand(10, 3), columns=['A', 'B', 'C'])
    allocator = EqualWeightAllocator()
    weights = allocator.allocate(data)
    assert np.allclose(weights.sum(axis=1), 1.0)

def test_risk_parity_normalization():
    # Synthetic returns with different vols
    returns = pd.DataFrame({
        'HighVol': np.random.normal(0, 0.05, 100),
        'LowVol': np.random.normal(0, 0.01, 100)
    })
    allocator = RiskParityAllocator()
    weights = allocator.allocate(returns)
    # Low Vol should have higher weight
    assert (weights['LowVol'] > weights['HighVol']).all()
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)
