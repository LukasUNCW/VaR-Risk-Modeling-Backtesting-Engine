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
    Monte Carlo VaR for multi-asset port

    assumes returns follow multivariate norm distro with empirical mean and covariance from data

    parameters
        returns: pd.DataFrame
            T x N matrix of asset returns
        weights: np.ndarray
            port weights length of N
        alpha: float
            confidence level
        n_sims: int
            n of Monte Carlo simulations 
        seed: int
            random seed for reproduciblity

        returns:
            float
                port VaR positive loss format
    """
    rng = np.random.default_rng(seed) # NumPy rand num generator
    mu = returns.mean().to_numpy() # compute mean return vector length N
    cov = returns.cov().to_numpy() # compute covariance matrix N x N

    sims = rng.multivariate_normal(mu, cov, size=n_sims) # sim multivariate normal returns, output shape = n_sims, N
    w = np.asarray(weights, dtype=float) # converts weights to NumPy array
    w = w / w.sum() # normalize weights to ensure they sum to 1, protects against user input error 
    port = sims @ w  # compute simulated port returns, matrix multiplication
    q = np.quantile(port, 1 - alpha) # left-tail quantile of port returns

    return float(-q) # convert to positive loss value

