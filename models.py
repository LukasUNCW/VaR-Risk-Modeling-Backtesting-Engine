import numpy as np
import pandas as pd
from scipy.stats import norm # normal distribution

def var_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """
    historical 1 day VaR

    parameters
        r: pd.Series
            time series of returns
        alpha: float
            confidence level

    returns
        float
            positive VaR value represents loss threshold
    
    """
    q = np.quantile(r, 1 - alpha) #compute the (1- alpha) empirical quantile of returns
    # example: alpha = 0.99 (99% confidence) -> 1% left tail
    return float(-q) # quantile is typically negative
    # multiply by -1 so VaR is reported as a positive loss number

def cvar_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """
    historical conditional VaR aka expected shortfall

    this measures the average loss given the VaR threshold has already been breached

    parameters
        r: pd.Series
            historical returns
        alpha: float
            confidence level

    returns
        float
            positive expected loss beyond VaR
    
    """
    cutoff = np.quantile(r, 1 - alpha) # compute VaR cutoff (left-tail quantile)
    tail = r[r <= cutoff] # select returns that are worse than or equal VaR cutoff
    return float(-tail.mean()) # computes mean of tail losses, negate so CVaR is expressed as positive loss

def var_parametric_normal(r: pd.Series, alpha: float = 0.99) -> float:
    """
    parametric VaR assuming returns follow a norm distribution

    uses
        VaR = -(mu + z_alpha * sd)

    parameters
        r: pd.Series
            historical returns
        alpha: float
            confidence level
    """
    mu = r.mean() # mean return 
    sigma = r.std(ddof=1) # estimate sd, ddof=1 uses unbiased sample sd
    z = norm.ppf(1 - alpha)  # compute z-score linked to left tail, alpha=0.99 -> norm.ppf(0.01) ~ -2.33
    return float(-(mu + z * sigma)) # combine mean and vol into VaR formula, multiply by -1

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
