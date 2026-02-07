import numpy as np
import pandas as pd
from scipy.stats import chi2

def exception_series(realized_returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """
    Exception when loss > VaR  <=>  -r_t > VaR_t
    Returns 1 for exception else 0.
    """
    aligned = pd.concat([realized_returns, var_series], axis=1).dropna()
    r = aligned.iloc[:, 0]
    v = aligned.iloc[:, 1]
    return (-r > v).astype(int)

def kupiec_pof_test(exceptions: pd.Series, alpha: float) -> dict:
    """
    Kupiec Proportion of Failures test.
    H0: exception probability = (1-alpha)
    """
    exc = exceptions.dropna().to_numpy()
    x = int(exc.sum())
    n = int(len(exc))
    p = 1 - alpha

    if n == 0:
        raise ValueError("No exceptions to test (empty series).")

    # Edge cases: all or none exceptions => test statistic degenerates
    if x == 0 or x == n:
        return {"LR_pof": float("inf"), "p_value": 0.0, "exceptions": x, "n": n}

    phat = x / n
    lr = -2 * ((n - x) * np.log((1 - p) / (1 - phat)) + x * np.log(p / phat))
    pval = 1 - chi2.cdf(lr, df=1)

    return {"LR_pof": float(lr), "p_value": float(pval), "exceptions": x, "n": n, "hit_rate": phat}