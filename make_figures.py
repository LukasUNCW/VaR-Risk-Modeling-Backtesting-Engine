"""
Regenerates the figures and backtest table used in the README.

    python make_figures.py

Runs a rolling, strictly out-of-sample 1-day VaR backtest on the default
portfolio for five models, then writes light and dark versions of each figure
to assets/ plus assets/backtest_results.csv.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from data import get_prices
from returns import log_returns, portfolio_returns
from models import var_monte_carlo_portfolio, var_garch, var_evt_pot
from backtest import exception_series, christoffersen_cc_test

TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
WEIGHTS = [0.4, 0.3, 0.2, 0.1]
START, END = "2015-01-01", "2026-09-26"   # fixed end date keeps results reproducible
ALPHA = 0.99
WINDOW = 250

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
CACHE = os.environ.get("VAR_FIG_CACHE")  # optional pickle path to skip recomputing

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

MODELS = ["Historical", "Parametric", "Monte Carlo", "GARCH(1,1)", "EVT-POT"]

# fixed categorical order, light / dark steps (colorblind-validated palette)
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", loss="#b5b3ab", critical="#d03b3b"),
    "dark":  dict(surface="#0d1117", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", loss="#5c5b56", critical="#e66767"),
}


# ── computation ──────────────────────────────────────────────────────────────

def rolling_apply(r: pd.Series, fn) -> pd.Series:
    """VaR for day i estimated from the WINDOW returns strictly before i."""
    vals = []
    for i in range(WINDOW, len(r)):
        try:
            vals.append(fn(r.iloc[i - WINDOW:i]))
        except ValueError:
            vals.append(np.nan)
    return pd.Series(vals, index=r.index[WINDOW:])


def compute() -> tuple[pd.Series, pd.DataFrame]:
    prices = get_prices(TICKERS, start=START, end=END)
    rets = log_returns(prices)
    port_r = portfolio_returns(rets, WEIGHTS)

    z = norm.ppf(1 - ALPHA)
    var = pd.DataFrame(index=port_r.index)
    var["Historical"] = -port_r.rolling(WINDOW).quantile(1 - ALPHA).shift(1)
    var["Parametric"] = -(port_r.rolling(WINDOW).mean().shift(1)
                          + z * port_r.rolling(WINDOW).std(ddof=1).shift(1))
    var["Monte Carlo"] = pd.Series(
        [var_monte_carlo_portfolio(rets.iloc[i - WINDOW:i], WEIGHTS, ALPHA, n_sims=25_000)
         for i in range(WINDOW, len(rets))],
        index=rets.index[WINDOW:],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var["GARCH(1,1)"] = rolling_apply(port_r, lambda w: var_garch(w, ALPHA))
        var["EVT-POT"] = rolling_apply(port_r, lambda w: var_evt_pot(w, ALPHA))

    var = var.dropna()
    return port_r.loc[var.index], var


def backtest_table(port_r: pd.Series, var: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        res = christoffersen_cc_test(exception_series(port_r, var[m]), ALPHA)
        p_pof = 1 - chi2.cdf(res["LR_pof"], 1)
        p_ind = 1 - chi2.cdf(res["LR_ind"], 1)
        rows.append({
            "Model": m,
            "Exceptions": res["exceptions"],
            "Days": res["n"],
            "Hit rate": res["hit_rate"],
            "Kupiec p": p_pof,
            "Independence p": p_ind,
            "Cond. coverage p": res["p_value"],
            "Verdict (5%)": "Pass" if res["p_value"] > 0.05 else "Reject",
        })
    return pd.DataFrame(rows).set_index("Model")


# ── plotting ─────────────────────────────────────────────────────────────────

def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0, pad=6)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(t, w=10, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    return fig, ax


def titles(fig, t, title, subtitle):
    fig.text(0.012, 0.965, title, color=t["ink"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.012, 0.905, subtitle, color=t["ink2"], fontsize=9.5, va="top")


def legend(ax, t, **kw):
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=t["ink2"], handlelength=1.6, **kw)
    return leg


def pct(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def save(fig, name, mode):
    suffix = "" if mode == "light" else "_dark"
    fig.savefig(os.path.join(OUT, f"{name}{suffix}.png"), facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_backtest(port_r, var, t, c, mode, name, start=None, end=None, models=None, sub=""):
    """Realized loss vs VaR forecasts, exceptions marked for the first model."""
    models = models or ["Historical", "Parametric", "GARCH(1,1)"]
    loss = (-port_r).loc[start:end]
    v = var.loc[start:end]

    fig, ax = new_fig(t)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.10)
    ax.bar(loss.index, loss.clip(lower=0), width=1.0, color=t["loss"], linewidth=0,
           label="Realized loss")
    for m in models:
        ax.plot(v.index, v[m], color=c[MODELS.index(m)], lw=1.6 if start else 1.2,
                label=f"{m} VaR", solid_capstyle="round")
    exc = loss[loss > v[models[0]]]
    ax.scatter(exc.index, exc, s=22, color=t["critical"], edgecolor=t["surface"],
               linewidth=1.2, zorder=5, label=f"Exception ({models[0]})")

    pct(ax)
    ax.set_ylim(0, None)
    if start:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_xlim(loss.index[0] - pd.Timedelta(days=2), loss.index[-1] + pd.Timedelta(days=2))
    else:
        ax.margins(x=0.01)
    legend(ax, t, loc="upper left", ncol=len(models) + 2, bbox_to_anchor=(0, 1.1))
    titles(fig, t, "Rolling 1-day 99% VaR vs. realized portfolio loss", sub)
    save(fig, name, mode)


def fig_exception_rate(table, t, c, mode):
    fig, ax = new_fig(t, h=3.6)
    fig.subplots_adjust(left=0.14, right=0.93, top=0.78, bottom=0.12)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)

    rates = table["Hit rate"][::-1]
    ys = np.arange(len(rates))
    ax.barh(ys, rates, height=0.5, color=c[0])
    ax.axvline(1 - ALPHA, color=t["ink"], lw=1)
    ax.text(1 - ALPHA, len(rates) - 0.35, " expected 1.0%", color=t["ink2"], fontsize=9, va="bottom")
    for y, (m, r) in zip(ys, rates.items()):
        row = table.loc[m]
        ax.text(r + 0.0004, y, f"{r:.2%}  ·  {row['Exceptions']} of {row['Days']:,} days  ·  "
                f"{row['Verdict (5%)'].lower()}", color=t["ink2"], fontsize=9, va="center")
    ax.set_yticks(ys, rates.index, color=t["ink"], fontsize=10)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_xlim(0, rates.max() * 1.9)
    ax.spines["bottom"].set_visible(False)
    titles(fig, t, "Exception rate by model",
           "Share of days the realized loss exceeded the 99% VaR forecast; "
           "verdict = Christoffersen conditional coverage test at 5%")
    save(fig, "exception_rate", mode)


def fig_timeline(port_r, var, t, c, mode):
    fig, ax = new_fig(t, h=3.4)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.78, bottom=0.12)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    for k, m in enumerate(MODELS[::-1]):
        e = exception_series(port_r, var[m])
        d = e.index[e == 1]
        ax.scatter(d, np.full(len(d), k), s=34, color=c[MODELS.index(m)],
                   edgecolor=t["surface"], linewidth=1.2, zorder=3)
    ax.set_yticks(range(len(MODELS)), MODELS[::-1], color=t["ink"], fontsize=10)
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.spines["bottom"].set_visible(False)
    ax.margins(x=0.01)
    titles(fig, t, "When each model failed",
           "Each dot is a day the realized loss exceeded that model's 99% VaR. "
           "Clusters mean the model reacted too slowly to a volatility shift")
    save(fig, "exception_timeline", mode)


def fig_distribution(port_r, t, c, mode):
    loss = -port_r
    fig, ax = new_fig(t, h=4.2)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.12)

    bins = np.linspace(loss.min(), loss.max(), 120)
    ax.hist(loss, bins=bins, density=True, color=c[0], alpha=0.85, linewidth=0,
            label="Daily portfolio loss")
    x = np.linspace(loss.min(), loss.max(), 600)
    ax.plot(x, norm.pdf(x, loss.mean(), loss.std()), color=c[1], lw=2, label="Normal fit")

    var_h = float(np.quantile(loss, ALPHA))
    es_h = float(loss[loss >= var_h].mean())
    var_n = float(loss.mean() + norm.ppf(ALPHA) * loss.std())
    ax.set_yscale("log")
    ax.set_ylim(0.02, 2000)
    for xv, lab, y, ha in [(var_n, f"Normal VaR {var_n:.2%} ", 600, "right"),
                           (var_h, f" Historical VaR {var_h:.2%}", 600, "left"),
                           (es_h, f" Expected shortfall {es_h:.2%}", 150, "left")]:
        ax.axvline(xv, color=t["ink"], lw=1)
        ax.text(xv, y, lab, color=t["ink2"], fontsize=9, va="center", ha=ha,
                bbox=dict(facecolor=t["surface"], edgecolor="none", pad=1.5))

    ax.set_xlim(-0.03, loss.max() * 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_yticks([])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    legend(ax, t, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.1))
    titles(fig, t, "Loss distribution: fat tails vs. the Normal assumption",
           "Full sample, log scale. The Normal curve collapses in the tail while real losses keep "
           "occurring out to 7%, so Normal VaR understates the empirical 99% quantile")
    save(fig, "loss_distribution", mode)


def main():
    os.makedirs(OUT, exist_ok=True)
    if CACHE and os.path.exists(CACHE):
        port_r, var = pd.read_pickle(CACHE)
    else:
        port_r, var = compute()
        if CACHE:
            pd.to_pickle((port_r, var), CACHE)

    table = backtest_table(port_r, var)
    table.to_csv(os.path.join(OUT, "backtest_results.csv"), float_format="%.4f")
    print(table.to_string(float_format=lambda f: f"{f:.4f}"))
    print("\nLatest VaR (", var.index[-1].date(), "):\n", var.iloc[-1].map("{:.2%}".format), sep="")

    period = f"{var.index[0]:%b %Y} – {var.index[-1]:%b %Y}"
    port = " / ".join(f"{w:.0%} {tk}" for tk, w in zip(TICKERS, WEIGHTS))
    for mode in ("light", "dark"):
        t, c = THEMES[mode], SERIES[mode]
        fig_backtest(port_r, var, t, c, mode, "var_backtest",
                     sub=f"{port} · {WINDOW}-day rolling window · {period}")
        fig_backtest(port_r, var, t, c, mode, "var_backtest_covid", "2020-01-15", "2020-07-15",
                     sub="COVID crash close-up: GARCH reacts within days and decays as markets calm; "
                         "Historical VaR lags, then stays pinned at its peak for a year")
        fig_exception_rate(table, t, c, mode)
        fig_timeline(port_r, var, t, c, mode)
        fig_distribution(port_r, t, c, mode)


if __name__ == "__main__":
    main()
