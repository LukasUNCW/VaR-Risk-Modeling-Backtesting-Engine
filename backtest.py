import numpy as np
import pandas as pd
from scipy.stats import chi2

def exception_series(realized_returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """
    computes a VaR exception (also called a 'hit') series.

    an exception occurs when the realized loss exceeds the VaR estimate:
    loss > VaR <-> -r_t > VaR_t

    returns:
        a pandas series of 1s and 0s
        1 -> exception occured
        0 -> no exception
    """
    # combine the realized returns and VaR series into a single DataFrame
    # this aligns them by date/index
    aligned = pd.concat([realized_returns, var_series], axis=1).dropna()
    r = aligned.iloc[:, 0] # extract the realized returns (first column)
    v = aligned.iloc[:, 1] # extract the VaR estimates (second column)
    return (-r > v).astype(int) # checks whether the realized loss exceeded VaR
    # -r converts returns into losses
    # (-r > v) produces a bool Series
    # .astype(int) converts T or F into 1 or 0 respectively

def kupiec_pof_test(exceptions: pd.Series, alpha: float) -> dict:
    """
    Kupiec Proportion of Failures (POF )test

    this tests whether the observed exception frequency matches the expected
    exception probability implied by the VaR confidence level.

    hypotheses:
    H0: P(exception) = 1 - alpha
    H1: P(exception) != 1 - alpha

    this returns a dicttionary containing test statistic, p-value, and summary metrics
    """
    exc = exceptions.dropna().to_numpy() # removes missing values and converts the expection Series into a NumPy array
    x = int(exc.sum()) # count total number of expections (sum of 1's)
    n = int(len(exc)) # total of observations 
    p = 1 - alpha # expected exception probability under null hypothesis
    # example: alpha = 0.99 -> p = 0.01

    if n == 0: # no data means test cannon be computed
        raise ValueError("No exceptions to test (empty series).")

    # if zero exceptions or exceptions every period
    # likelihood ratio formula breaks
    if x == 0 or x == n:
        return {
            "LR_pof": float("inf"), 
            "p_value": 0.0, 
            "exceptions": x, 
            "n": n}

    phat = x / n # exception rate observed in data

    # likelihood ratio for POF test
    # likelihood under null hypothesis and under empirical probability (p and phat)
    lr = -2 * (
        (n - x) * np.log((1 - p) / (1 - phat)) +
        x * np.log(p / phat)
    )

    # compute the p-value, degree of freedom = 1
    pval = 1 - chi2.cdf(lr, df=1)


    return {
            "LR_pof": float(lr), # likelihood ratio stat
            "p_value": float(pval), # stat significance
            "exceptions": x, # num of VaR breaches
            "n": n, # num of oberservations
            "hit_rate": phat} # exception freq
