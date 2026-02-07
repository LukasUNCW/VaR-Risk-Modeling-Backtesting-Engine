import numpy as np
import pandas as pd

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price levels."""
    return np.log(prices / prices.shift(1)).dropna()

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    s = w.sum()
    if s == 0:
        raise ValueError("weights sum to 0")
    return w / s

def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute portfolio return series given asset returns (T x N) and weights (N,)."""
    w = normalize_weights(weights)
    if returns.shape[1] != len(w):
        raise ValueError("weights length must match number of assets")
    return returns @ w