import numpy as np
import pandas as pd
from scipy.stats import norm, genpareto

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


def var_garch(r: pd.Series, alpha: float = 0.99) -> float:
    """
    1-day-ahead parametric VaR using GARCH(1,1) conditional volatility.

    Fits a GARCH(1,1) model with normal innovations on the return window r
    and uses the one-step-ahead variance forecast to estimate VaR:

        VaR = -(mu + z_alpha * sigma_{T+1|T})

    This captures volatility clustering that rolling-std ignores: after a
    calm period sigma is low; after a turbulent period sigma is high.

    parameters
        r: pd.Series
            historical return window (typically 250 trading days)
        alpha: float
            confidence level (e.g. 0.99 for 99% VaR)

    returns
        float
            positive VaR value (loss convention)
    """
    from arch import arch_model

    # arch works in percentage space for numerical stability
    scaled = r * 100
    am = arch_model(scaled, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off")

    # one-step-ahead variance forecast at horizon 1
    forecast = res.forecast(horizon=1, reindex=False)
    sigma_next = float(np.sqrt(forecast.variance.values[-1, 0])) / 100

    mu = float(r.mean())
    z  = norm.ppf(1 - alpha)
    return float(-(mu + z * sigma_next))


def var_evt_pot(
    r: pd.Series,
    alpha: float = 0.99,
    threshold_quantile: float = 0.95,
) -> float:
    """
    1-day VaR via Peaks-over-Threshold (POT) with Generalized Pareto Distribution.

    The POT method fits a GPD only to the extreme tail of the loss distribution,
    which is more efficient than fitting a parametric model to the whole sample.

    Steps
    -----
    1. Convert returns to losses: L = -r
    2. Choose threshold u = quantile(L, threshold_quantile)
    3. Extract exceedances: e = L[L > u] - u
    4. Fit GPD(xi, beta) to e using MLE with loc fixed at 0
    5. Invert the POT survival function to get VaR at level alpha:

           VaR = u + (beta/xi) * [(n/N_u * (1-alpha))^(-xi) - 1]   (xi ≠ 0)
           VaR = u + beta * log(n / (N_u * (1-alpha)))              (xi ≈ 0)

    parameters
        r: pd.Series
            historical return window
        alpha: float
            VaR confidence level (must satisfy threshold_quantile < alpha)
        threshold_quantile: float
            quantile of losses used as the GPD threshold (default 0.95)

    returns
        float
            positive VaR value (loss convention)

    raises
        ValueError
            if fewer than 10 exceedances remain after applying the threshold
    """
    losses = -r.to_numpy(dtype=float)
    u      = float(np.quantile(losses, threshold_quantile))

    exceedances = losses[losses > u] - u
    if len(exceedances) < 10:
        raise ValueError(
            f"Only {len(exceedances)} exceedances above threshold — need ≥ 10 for GPD fit."
        )

    # MLE fit with location fixed at 0; returns (shape=xi, loc=0, scale=beta)
    xi, _, beta = genpareto.fit(exceedances, floc=0)

    n   = len(losses)
    n_u = len(exceedances)

    # probability mass above u assigned to the POT tail
    p_exceed = (1 - alpha) * n / n_u   # fraction of exceedances needed

    if p_exceed <= 0 or p_exceed >= 1:
        raise ValueError(
            f"Invalid p_exceed={p_exceed:.4f}. Check threshold_quantile < alpha."
        )

    if abs(xi) < 1e-8:                             # exponential limit (xi → 0)
        var_evt = u + beta * np.log(1.0 / p_exceed)
    else:
        var_evt = u + (beta / xi) * (p_exceed ** (-xi) - 1.0)

    return float(var_evt)