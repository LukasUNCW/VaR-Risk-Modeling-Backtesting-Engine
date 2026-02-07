import pandas as pd
from varlab.data import get_prices
from varlab.returns import log_returns
from varlab.backtest import exception_series, kupiec_pof_test
from varlab.plots import plot_var_backtest
from scipy.stats import norm

def main():
    ticker = "SPY"
    start = "2015-01-01"
    alpha = 0.99
    window = 250

    prices = get_prices([ticker], start=start)
    r = log_returns(prices)[ticker]

    # Rolling Historical VaR
    var_hist = -r.rolling(window).quantile(1 - alpha)

    # Rolling Parametric Normal VaR
    z = norm.ppf(1 - alpha)
    mu = r.rolling(window).mean()
    sigma = r.rolling(window).std(ddof=1)
    var_param = -(mu + z * sigma)

    out = pd.DataFrame({
        "Loss": -r,
        "VaR_hist": var_hist,
        "VaR_param": var_param
    }).dropna()

    exc_hist = exception_series(r, out["VaR_hist"])
    exc_param = exception_series(r, out["VaR_param"])

    print(f"\nSingle-Asset VaR Backtest: {ticker} | alpha={alpha} | window={window}")
    print("Historical:", kupiec_pof_test(exc_hist, alpha))
    print("Parametric:", kupiec_pof_test(exc_param, alpha))

    plot_var_backtest(out, f"Rolling 1-Day VaR Backtest: {ticker} (alpha={alpha}, window={window})")

if __name__ == "__main__":
    main()