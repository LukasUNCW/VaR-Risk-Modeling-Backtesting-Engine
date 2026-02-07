import numpy as np
import pandas as pd
from scipy.stats import norm

def var_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """Historical 1-day VaR (positive number = loss)."""
    q = np.quantile(r, 1 - alpha)
    return float(-q)

def cvar_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """Historical Expected Shortfall (CVaR)."""
    cutoff = np.quantile(r, 1 - alpha)
    tail = r[r <= cutoff]
    return float(-tail.mean())

def var_parametric_normal(r: pd.Series, alpha: float = 0.99) -> float:
    """Parametric VaR assuming Normal returns."""
    mu = r.mean()
    sigma = r.std(ddof=1)
    z = norm.ppf(1 - alpha)  # negative for alpha>0.5
    return float(-(mu + z * sigma))

def var_monte_carlo_portfolio(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.99,
    n_sims: int = 50_000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo VaR for a portfolio using multivariate normal with empirical mean/cov.
    returns: T x N
    weights: N
    """
    rng = np.random.default_rng(seed)
    mu = returns.mean().to_numpy()
    cov = returns.cov().to_numpy()

    sims = rng.multivariate_normal(mu, cov, size=n_sims)  # S x N
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    port = sims @ w  # S
    q = np.quantile(port, 1 - alpha)
    return float(-q)