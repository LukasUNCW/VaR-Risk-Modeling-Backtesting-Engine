import pandas as pd
from varlab.data import get_prices
from varlab.returns import log_returns, portfolio_returns
from varlab.models import var_monte_carlo_portfolio
from varlab.backtest import exception_series, kupiec_pof_test
from varlab.plots import plot_var_backtest
from scipy.stats import norm

def main():
    tickers = ["SPY", "QQQ", "TLT", "GLD"]
    weights = [0.4, 0.3, 0.2, 0.1]   # must match tickers order
    start = "2015-01-01"
    alpha = 0.99
    window = 250

    prices = get_prices(tickers, start=start)
    rets = log_returns(prices)

    port_r = portfolio_returns(rets, weights)

    # Rolling Historical & Parametric for portfolio (using port returns)
    var_hist = -port_r.rolling(window).quantile(1 - alpha)
    z = norm.ppf(1 - alpha)
    mu = port_r.rolling(window).mean()
    sigma = port_r.rolling(window).std(ddof=1)
    var_param = -(mu + z * sigma)

    # Rolling Monte Carlo VaR (use asset returns window each day)
    var_mc = []
    idx = []
    for i in range(window, len(rets)):
        w_rets = rets.iloc[i-window:i]  # T x N window
        v = var_monte_carlo_portfolio(w_rets, weights, alpha=alpha, n_sims=25_000, seed=42)
        var_mc.append(v)
        idx.append(rets.index[i])

    var_mc = pd.Series(var_mc, index=idx, name="VaR_mc")

    out = pd.DataFrame({
        "Loss": -port_r,
        "VaR_hist": var_hist,
        "VaR_param": var_param,
    }).join(var_mc, how="inner").dropna()

    exc_hist = exception_series(port_r, out["VaR_hist"])
    exc_param = exception_series(port_r, out["VaR_param"])
    exc_mc = exception_series(port_r, out["VaR_mc"])

    print(f"\nPortfolio VaR Backtest: {tickers} | weights={weights} | alpha={alpha} | window={window}")
    print("Historical:", kupiec_pof_test(exc_hist, alpha))
    print("Parametric:", kupiec_pof_test(exc_param, alpha))
    print("MonteCarlo:", kupiec_pof_test(exc_mc, alpha))

    plot_var_backtest(out, f"Rolling 1-Day Portfolio VaR (alpha={alpha}, window={window})")

if __name__ == "__main__":
    main()