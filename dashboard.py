"""
Interactive VaR Risk Modeling & Backtesting Dashboard
======================================================
Run with:   streamlit run dashboard.py

Controls (sidebar)
------------------
  Tickers          space- or comma-separated ticker symbols
  Weights          matching weights (auto-normalised)
  Start date       earliest price date to fetch
  Confidence level 90 / 95 / 99 %
  Rolling window   lookback window in trading days

Models
------
  Historical          empirical quantile (non-parametric)
  Parametric Normal   rolling mean + rolling std, Normal z-score
  GARCH(1,1)          ARCH-filtered conditional volatility (single full-sample
                      fit; parameters fixed, conditional vol updated in-sample)
  EVT-POT             Generalised Pareto fit to exceedances above the 95th-
                      percentile loss threshold — rolling window

Backtesting
-----------
  Kupiec POF               unconditional coverage (LR ~ chi2(1))
  Christoffersen Ind.      serial independence of exceptions (LR ~ chi2(1))
  Christoffersen CC        joint conditional coverage (LR ~ chi2(2))
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm, genpareto

# ── local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data     import get_prices
from returns  import log_returns, portfolio_returns, normalize_weights
from backtest import (
    exception_series,
    kupiec_pof_test,
    christoffersen_independence_test,
    christoffersen_cc_test,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VaR Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Cached computation helpers
# Each function is pure (no st.* calls) so @st.cache_data can hash its args.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return get_prices(list(tickers), start=start)


@st.cache_data(show_spinner=False)
def rolling_historical_var(port_r: pd.Series, window: int, alpha: float) -> pd.Series:
    return -port_r.rolling(window).quantile(1 - alpha)


@st.cache_data(show_spinner=False)
def rolling_parametric_var(port_r: pd.Series, window: int, alpha: float) -> pd.Series:
    z     = norm.ppf(1 - alpha)
    mu    = port_r.rolling(window).mean()
    sigma = port_r.rolling(window).std(ddof=1)
    return -(mu + z * sigma)


@st.cache_data(show_spinner=False)
def rolling_garch_var(port_r: pd.Series, alpha: float) -> pd.Series:
    """
    Fit GARCH(1,1) once on the full portfolio-return series.

    Using the full-sample MLE parameters with in-sample filtered conditional
    volatility is a fast, practically adequate approximation for exploration.
    For production backtesting, a proper rolling/recursive fit is required to
    avoid parameter look-ahead bias.
    """
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am  = arch_model(port_r * 100, vol="Garch", p=1, q=1,
                         dist="normal", rescale=False)
        res = am.fit(disp="off")

    cond_vol              = res.conditional_volatility / 100
    cond_vol.index        = port_r.index
    z                     = norm.ppf(1 - alpha)
    mu                    = float(port_r.mean())
    return -(mu + z * cond_vol).rename("VaR_garch")


@st.cache_data(show_spinner=False)
def rolling_evt_var(
    port_r: pd.Series,
    window: int,
    alpha: float,
    threshold_q: float = 0.95,
) -> pd.Series:
    """Proper rolling POT-GPD VaR — refits GPD at each step."""
    losses   = -port_r
    var_vals = []
    idx_vals = []

    for i in range(window, len(losses)):
        wl  = losses.iloc[i - window : i].to_numpy(dtype=float)
        u   = float(np.quantile(wl, threshold_q))
        exc = wl[wl > u] - u

        if len(exc) < 10:
            var_vals.append(np.nan)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xi, _, beta = genpareto.fit(exc, floc=0)

            n_obs     = len(wl)
            n_u       = len(exc)
            p_exceed  = (1 - alpha) * n_obs / n_u

            if p_exceed <= 0 or p_exceed >= 1:
                var_vals.append(np.nan)
            elif abs(xi) < 1e-8:
                var_vals.append(float(u + beta * np.log(1.0 / p_exceed)))
            else:
                var_vals.append(float(u + (beta / xi) * (p_exceed ** (-xi) - 1.0)))

        idx_vals.append(losses.index[i])

    return pd.Series(var_vals, index=idx_vals, name="VaR_evt")


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

_COLOURS = {
    "Loss":            "#7f8c8d",
    "Historical":      "#2980b9",
    "Parametric":      "#27ae60",
    "GARCH":           "#e67e22",
    "EVT-POT":         "#8e44ad",
}

def _var_chart(df_plot: pd.DataFrame, alpha: float) -> go.Figure:
    """Line chart of realised loss vs one or more VaR series."""
    fig = go.Figure()
    col_map = {
        "Loss":      ("Realised Loss",          _COLOURS["Loss"],       "lines",       0.5),
        "VaR_hist":  ("VaR – Historical",        _COLOURS["Historical"], "lines",       1.8),
        "VaR_param": ("VaR – Parametric Normal", _COLOURS["Parametric"], "lines",       1.8),
        "VaR_garch": ("VaR – GARCH(1,1)",        _COLOURS["GARCH"],      "lines",       1.8),
        "VaR_evt":   ("VaR – EVT-POT",           _COLOURS["EVT-POT"],    "lines",       1.8),
    }
    for col, (label, colour, mode, width) in col_map.items():
        if col not in df_plot.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot[col],
            name=label, mode=mode,
            line=dict(color=colour, width=width),
        ))
    fig.update_layout(
        title=f"Rolling 1-Day Portfolio VaR  ({int(alpha*100)}% confidence)",
        xaxis_title="Date", yaxis_title="Loss / VaR",
        hovermode="x unified", height=480,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=50, r=20, t=50, b=60),
    )
    return fig


def _exception_chart(exc_df: pd.DataFrame) -> go.Figure:
    """
    Timeline of exception events per model.

    Each row in exc_df is a date; each column is a model.
    Exceptions (value == 1) are shown as vertical markers.
    """
    fig = go.Figure()
    colours = [_COLOURS["Historical"], _COLOURS["Parametric"],
               _COLOURS["GARCH"], _COLOURS["EVT-POT"]]

    for idx, col in enumerate(exc_df.columns):
        exc_dates = exc_df.index[exc_df[col] == 1]
        if len(exc_dates) == 0:
            continue
        y_offset = idx + 1
        fig.add_trace(go.Scatter(
            x=exc_dates,
            y=[y_offset] * len(exc_dates),
            mode="markers",
            name=col,
            marker=dict(
                symbol="line-ns",
                size=10,
                line=dict(width=1.5, color=colours[idx % len(colours)]),
            ),
        ))

    fig.update_layout(
        title="Exception Timeline (VaR Breaches)",
        xaxis_title="Date",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(1, len(exc_df.columns) + 1)),
            ticktext=list(exc_df.columns),
            showgrid=False,
        ),
        hovermode="x unified",
        height=320,
        margin=dict(l=120, r=20, t=50, b=50),
    )
    return fig


def _annual_exception_heatmap(exc_df: pd.DataFrame) -> go.Figure:
    combined = exc_df.max(axis=1)
    combined.name = "any_exception"
    df_time = combined.to_frame()
    df_time["year"]  = df_time.index.year
    df_time["month"] = df_time.index.month

    pivot = df_time.groupby(["year", "month"])["any_exception"].sum().unstack(fill_value=0)
    
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index.tolist()],
        colorscale="Reds",
        showscale=True,
        colorbar=dict(title="# exceptions"),
    ))
    fig.update_layout(
        title="Exception Clustering  (any model, by month)",
        xaxis_title="Month", yaxis_title="Year",
        height=max(280, len(pivot) * 30 + 80),
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def _backtest_table(results: dict[str, dict], alpha: float) -> pd.DataFrame:
    """
    Build a styled summary DataFrame from per-model backtest dicts.
    Columns: model, exceptions, hit_rate, expected_rate,
             LR_pof, p_pof, LR_ind, p_ind, LR_cc, p_cc, verdict.
    """
    rows = []
    expected = 1 - alpha
    for model, r in results.items():
        verdict = "PASS" if r["p_cc"] > 0.05 else "FAIL"
        rows.append({
            "Model":         model,
            "Exceptions":    r["exceptions"],
            "Hit rate":      f"{r['hit_rate']:.4f}",
            "Expected":      f"{expected:.4f}",
            "LR_pof":        f"{r['LR_pof']:.3f}",
            "p (POF)":       f"{r['p_pof']:.4f}",
            "LR_ind":        f"{r['LR_ind']:.3f}",
            "p (Ind.)":      f"{r['p_ind']:.4f}",
            "LR_cc":         f"{r['LR_cc']:.3f}",
            "p (CC)":        f"{r['p_cc']:.4f}",
            "Verdict (5%)":  verdict,
        })
    return pd.DataFrame(rows).set_index("Model")


def _gpd_tail_chart(port_r: pd.Series, threshold_q: float = 0.95) -> go.Figure:
    """
    Plot the empirical tail and fitted GPD survival function.
    Uses the full sample for illustration.
    """
    losses = -port_r.dropna().to_numpy(dtype=float)
    u      = float(np.quantile(losses, threshold_q))
    exc    = losses[losses > u] - u

    if len(exc) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient exceedances for GPD fit",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False)
        return fig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi, _, beta = genpareto.fit(exc, floc=0)

    # empirical survival function for exceedances
    sorted_exc = np.sort(exc)
    n_u        = len(exc)
    emp_sf     = np.arange(n_u, 0, -1) / n_u  # P(E > e)

    # fitted GPD survival
    x_fit  = np.linspace(0, sorted_exc.max() * 1.05, 300)
    gpd_sf = (1 + xi * x_fit / beta) ** (-1 / xi) if abs(xi) > 1e-8 \
             else np.exp(-x_fit / beta)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_exc, y=emp_sf,
        mode="markers", name="Empirical tail",
        marker=dict(color="#2980b9", size=5, opacity=0.7),
    ))
    fig.add_trace(go.Scatter(
        x=x_fit, y=gpd_sf,
        mode="lines", name=f"GPD fit  ξ={xi:.3f}  β={beta:.5f}",
        line=dict(color="#e74c3c", width=2),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="grey",
                  annotation_text="threshold u", annotation_position="top right")
    fig.update_layout(
        title="Generalised Pareto Tail Fit  (full sample, exceedances above 95th pct.)",
        xaxis_title="Exceedance above threshold  (loss units)",
        yaxis_title="P(E > x)  —  survival probability",
        yaxis_type="log",
        height=420,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=60, r=20, t=60, b=70),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("Portfolio")
    tickers_raw = st.text_input(
        "Tickers (space or comma separated)",
        value="SPY QQQ TLT GLD",
    )
    weights_raw = st.text_input(
        "Weights (must match ticker order)",
        value="0.4 0.3 0.2 0.1",
    )
    start_date = st.text_input("Start date (YYYY-MM-DD)", value="2015-01-01")

    st.subheader("Model Parameters")
    conf_pct = st.select_slider(
        "Confidence level",
        options=[90, 95, 99],
        value=99,
        format_func=lambda x: f"{x}%",
    )
    alpha = conf_pct / 100.0

    window = st.slider("Rolling window (trading days)", 60, 500, 250, step=10)

    st.subheader("Models to run")
    run_hist  = st.checkbox("Historical",          value=True)
    run_param = st.checkbox("Parametric Normal",   value=True)
    run_garch = st.checkbox("GARCH(1,1)",           value=True)
    run_evt   = st.checkbox("EVT-POT (GPD)",        value=True)

    st.divider()
    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("📉 VaR Risk Modeling & Backtesting Dashboard")
st.caption(
    "Historical · Parametric Normal · GARCH(1,1) · EVT-POT  ·  "
    "Kupiec POF · Christoffersen Independence & Conditional Coverage"
)

# ─────────────────────────────────────────────────────────────────────────────
# Main logic — runs only when the button is clicked
# ─────────────────────────────────────────────────────────────────────────────

if not run_btn:
    st.info(
        "Configure your portfolio and model parameters in the sidebar, "
        "then click **▶ Run Backtest**."
    )
    st.stop()

# ── parse inputs ──────────────────────────────────────────────────────────────
import re as _re

tickers_list = [t.upper() for t in _re.split(r"[,\s]+", tickers_raw.strip()) if t]
try:
    weights_list = [float(w) for w in _re.split(r"[,\s]+", weights_raw.strip()) if w]
except ValueError:
    st.error("Could not parse weights — enter space- or comma-separated numbers.")
    st.stop()

if len(tickers_list) != len(weights_list):
    st.error(
        f"Number of tickers ({len(tickers_list)}) must match "
        f"number of weights ({len(weights_list)})."
    )
    st.stop()

weights_arr = normalize_weights(np.array(weights_list))

# ── data loading ──────────────────────────────────────────────────────────────
with st.spinner("Downloading price data from Yahoo Finance…"):
    try:
        prices = load_data(tuple(tickers_list), start_date)
    except Exception as exc:
        st.error(f"Data download failed: {exc}")
        st.stop()

rets    = log_returns(prices)
port_r  = portfolio_returns(rets, weights_arr)

# display portfolio composition
st.subheader("Portfolio composition")
comp_cols = st.columns(len(tickers_list))
for col, ticker, w in zip(comp_cols, tickers_list, weights_arr):
    col.metric(ticker, f"{w:.1%}")

st.markdown(
    f"**{len(port_r):,} daily returns**  ·  "
    f"{port_r.index[0].date()} → {port_r.index[-1].date()}  ·  "
    f"α = {alpha:.2f}  ·  window = {window} days"
)

# ── rolling VaR estimation ────────────────────────────────────────────────────
var_series: dict[str, pd.Series] = {}

if run_hist:
    with st.spinner("Computing Historical VaR…"):
        var_series["VaR_hist"] = rolling_historical_var(port_r, window, alpha)

if run_param:
    with st.spinner("Computing Parametric Normal VaR…"):
        var_series["VaR_param"] = rolling_parametric_var(port_r, window, alpha)

if run_garch:
    with st.spinner("Fitting GARCH(1,1) — this may take a few seconds…"):
        var_series["VaR_garch"] = rolling_garch_var(port_r, alpha)

if run_evt:
    with st.spinner("Computing EVT-POT VaR (rolling GPD fits)…"):
        var_series["VaR_evt"] = rolling_evt_var(port_r, window, alpha)

if not var_series:
    st.warning("Select at least one model to run.")
    st.stop()

# ── assemble output DataFrame ─────────────────────────────────────────────────
out = pd.DataFrame({"Loss": -port_r})
for key, series in var_series.items():
    series = series.rename(key)   # force the name to match the dict key
    out = out.join(series, how="left")

# ── exception series & backtest results ───────────────────────────────────────
label_map = {
    "VaR_hist":  "Historical",
    "VaR_param": "Parametric Normal",
    "VaR_garch": "GARCH(1,1)",
    "VaR_evt":   "EVT-POT",
}
exc_dict: dict[str, pd.Series]    = {}
bt_results: dict[str, dict]       = {}

for key in var_series:
    exc = exception_series(port_r, out[key])
    exc_dict[label_map[key]] = exc

    pof = kupiec_pof_test(exc, alpha)
    ind = christoffersen_independence_test(exc)
    cc  = christoffersen_cc_test(exc, alpha)

    bt_results[label_map[key]] = {
        "exceptions": pof["exceptions"],
        "hit_rate":   pof.get("hit_rate", float("nan")),
        "LR_pof":     pof["LR_pof"],
        "p_pof":      pof["p_value"],
        "LR_ind":     ind["LR_ind"],
        "p_ind":      ind["p_value"],
        "LR_cc":      cc["LR_cc"],
        "p_cc":       cc["p_value"],
    }

# ── summary metric strip ──────────────────────────────────────────────────────
st.subheader("Summary")
metric_cols = st.columns(len(bt_results))
for col, (model, res) in zip(metric_cols, bt_results.items()):
    verdict = "✅ PASS" if res["p_cc"] > 0.05 else "❌ FAIL"
    col.metric(
        label=model,
        value=f"{res['exceptions']} exceptions",
        delta=f"hit {res['hit_rate']:.3%} · CC {verdict}",
    )

# ── tabbed results ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 VaR Lines", "🚨 Exception Timeline", "📊 Backtest Statistics", "🔬 Tail Analysis"]
)

with tab1:
    st.plotly_chart(_var_chart(out, alpha), use_container_width=True)
    if run_garch:
        st.caption(
            "**GARCH(1,1) note:** The conditional volatility shown here is estimated "
            "from a single full-sample GARCH fit (in-sample filtered vol). "
            "GARCH parameters are fixed using all available data, which introduces "
            "minor look-ahead bias in the parameters. For a proper walk-forward "
            "backtest run `run_port.py` with the `var_garch` model."
        )

with tab2:
    exc_df = pd.DataFrame(exc_dict)
    st.plotly_chart(_exception_chart(exc_df), use_container_width=True)
    st.plotly_chart(_annual_exception_heatmap(exc_df), use_container_width=True)

with tab3:
    st.markdown(
        "**Test legend** · "
        "**POF** = Kupiec Proportion-of-Failures (H₀: correct unconditional coverage, χ²(1)) · "
        "**Ind.** = Christoffersen independence (H₀: exceptions i.i.d., χ²(1)) · "
        "**CC** = Conditional Coverage = POF + Ind. (χ²(2)) · "
        "Verdict uses 5% significance level."
    )
    df_bt = _backtest_table(bt_results, alpha)

    def _style_verdict(val):
        colour = "#27ae60" if "PASS" in str(val) else "#e74c3c"
        return f"color: {colour}; font-weight: bold"

    styled = df_bt.style.map(_style_verdict, subset=["Verdict (5%)"])
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Transition matrices (Christoffersen independence test)")
    ind_cols = st.columns(len(exc_dict))
    for col, (model, exc) in zip(ind_cols, exc_dict.items()):
        ind = christoffersen_independence_test(exc)
        col.markdown(f"**{model}**")
        tm = pd.DataFrame(
            [[ind["T00"], ind["T01"]], [ind["T10"], ind["T11"]]],
            index=["prev: no exc", "prev: exc"],
            columns=["curr: no exc", "curr: exc"],
        )
        col.dataframe(tm)
        pi01 = ind["pi_01"]
        pi11 = ind["pi_11"]
        if not (np.isnan(pi01) or np.isnan(pi11)):
            col.caption(
                f"π₀₁ = {pi01:.4f}  ·  π₁₁ = {pi11:.4f}  "
                f"({'clustering detected' if pi11 > pi01 * 2 else 'no strong clustering'})"
            )

with tab4:
    st.markdown(
        "**Peaks-over-Threshold (POT):** exceedances above the 95th-percentile "
        "loss are fitted to a Generalised Pareto Distribution.  "
        "The tail shape parameter **ξ > 0** indicates a heavy tail "
        "(Fréchet domain); **ξ ≈ 0** indicates an exponential tail (Gumbel); "
        "**ξ < 0** is a bounded tail (Weibull)."
    )
    st.plotly_chart(_gpd_tail_chart(port_r), use_container_width=True)

    # GPD parameters summary
    losses_full = -port_r.dropna().to_numpy(dtype=float)
    u_full      = float(np.quantile(losses_full, 0.95))
    exc_full    = losses_full[losses_full > u_full] - u_full

    if len(exc_full) >= 10:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xi_f, _, beta_f = genpareto.fit(exc_full, floc=0)

        g1, g2, g3 = st.columns(3)
        g1.metric("Shape  ξ",     f"{xi_f:.4f}")
        g2.metric("Scale  β",     f"{beta_f:.6f}")
        g3.metric("Threshold  u", f"{u_full:.6f}")

        tail_desc = (
            "Heavy tail (Fréchet) — extreme losses larger than Normal predicts."
            if xi_f > 0.05 else
            "Near-exponential tail (Gumbel) — consistent with Normal extremes."
            if abs(xi_f) <= 0.05 else
            "Bounded tail (Weibull) — losses have a finite upper bound."
        )
        st.info(tail_desc)