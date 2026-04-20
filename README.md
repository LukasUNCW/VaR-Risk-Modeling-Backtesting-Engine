# VaR Risk Modeling & Backtesting Engine

A Python-based project that implements Value-at-Risk (VaR) models and industry-standard backtesting to measure and validate market risk for single and multi-asset portfolios.

This project mirrors how risk engines are built and evaluated in real financial institutions, emphasizing:
- Statistical rigor
- Clean software design
- Reproducibility
- Realistic assumptions (rolling windows, no look-ahead bias)

---

## Table of Contents
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [How to Run](#how-to-run)
- [Visualizations](#visualizations)

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.14 | Core language |
| yfinance | Market data retrieval |
| NumPy / pandas | Data manipulation and return computation |
| SciPy | Statistical testing (Kupiec POF) |
| Matplotlib | Visualization |

---

## Key Features

### Risk Models
- **Historical VaR** (non-parametric, quantile-based): Estimates risk by taking the empirical loss quantile from historical returns without assuming any specific return distribution.
- **Parametric VaR** (Normal): Models risk by assuming returns follow a normal distribution and computing VaR from the rolling mean and standard deviation.
- **Monte Carlo VaR**: Simulates thousands of correlated asset return scenarios using estimated means and covariances to estimate portfolio-level VaR.
- **Expected Shortfall (CVaR)**: Measures the average loss conditional on losses exceeding the VaR threshold, providing insight into the severity of extreme outcomes.

### Backtesting
- Rolling-window VaR estimation
- Exception detection (losses exceeding VaR)
- Kupiec Proportion-of-Failures (POF) test for statistical validation

### Visualization
- VaR vs. realized-loss plots for diagnostic analysis
- Exception timeline charts
- Loss distribution histograms with VaR threshold overlay
- Model comparison bar charts

---

## Prerequisites & Installation

**1. Clone the repository**
```bash
git clone https://github.com/LukasUNCW/VaR-Risk-Modeling-Backtesting-Engine.git
cd VaR-Risk-Modeling-Backtesting-Engine
```

**2. Install dependencies**
```bash
pip install yfinance numpy pandas scipy matplotlib
```

**3. Ensure you are using Python 3.14**
```bash
python --version
```

---

## How to Run

Run both scripts from inside the project's root directory using the commands below.

### Single Asset (`run_single_asset.py`)
Analyzes one asset (e.g., SPY). Computes rolling Historical and Parametric VaR, runs backtesting, and plots results.

```bash
py -3.14 -m scripts.run_single_asset
```

### Portfolio (`run_port.py`)
Analyzes multiple assets (configurable). Computes portfolio returns, runs Historical, Parametric, and Monte Carlo VaR, backtests all models, and plots portfolio risk.

```bash
py -3.14 -m scripts.run_port
```

> **Note:** To configure the portfolio assets, edit the ticker list inside `scripts/run_port.py`.

---

## Visualizations

### Rolling VaR Backtest (VaR vs. Realized Loss)

<img width="1260" height="938" alt="var_backtest_portfolio" src="https://github.com/user-attachments/assets/b94c85da-f32f-4b86-91b0-a459ed3b113e" />

**What this chart shows:**
- X-axis: Time (trading days)
- Y-axis: Loss magnitude (positive = bad)
- Lines:
  - **Realized loss**: actual next-day portfolio loss
  - **VaR lines** (Historical / Parametric / Monte Carlo): predicted maximum loss at confidence level α (e.g., 99%)

This is a rolling, out-of-sample risk forecast.

**How to read it:**
- Most of the time, the loss line stays below VaR — this is expected behavior
- When the loss spikes above a VaR line, that is called an **exception**
- During volatile periods, VaR lines rise (risk adapts)
- During calm periods, VaR compresses (risk declines)

---

### Exception Timeline (VaR Failures)

<img width="1260" height="938" alt="exceptions_mc_portfolio" src="https://github.com/user-attachments/assets/98dc224f-7e5d-448c-bd7b-4b52914c5580" />

**What this chart shows:**
- Each dot = one VaR exception (a day where the realized loss exceeded the VaR estimate)
- X-axis: Date
- Y-axis: 1 = loss exceeded VaR

This is a binary diagnostic of VaR model correctness.

**How to read it:**
- At a 99% confidence level (α = 0.99), you expect exceptions on roughly **1% of trading days**
- **Clustered exceptions** indicate the model is slow to react — common during market crises or sudden volatility spikes
- Parametric VaR tends to cluster more; Monte Carlo typically improves clustering behavior

---

### Loss Distribution with VaR Threshold

<img width="1260" height="938" alt="loss_dist_var_portfolio" src="https://github.com/user-attachments/assets/ddf4892c-8788-44df-aa76-a49c5a2a0770" />

**What this chart shows:**
- Histogram of historical portfolio losses
- Vertical dashed line = VaR threshold
- Area to the right of the line = tail risk

This is the statistical meaning of VaR, visualized directly.

**How to read it:**
- VaR is a **quantile**, not a worst-case scenario
- At 99% VaR: 99% of losses fall to the left of the line; 1% fall to the right (tail losses)
- The tail is often skewed and heavier than a normal distribution assumes — this is why Historical and Monte Carlo VaR often outperform Parametric VaR

---

### Model Comparison Bar Chart

<img width="1260" height="938" alt="model_compare_portfolio" src="https://github.com/user-attachments/assets/ec517f2d-f595-4ad0-a191-18489f66518e" />

**What this chart shows:**
- One bar per VaR model, all computed on the same date, same portfolio, and same confidence level
- Isolates the effect of model assumptions on the VaR estimate

**How to read it:**

| Model | Behavior |
|---|---|
| Historical VaR | Data-driven; sensitive to recent history |
| Parametric VaR | Assumes normality; often the lowest estimate (underestimates tails) |
| Monte Carlo VaR | Incorporates correlations; typically the most conservative |

Differences between bars represent **model risk** — the uncertainty introduced by the choice of methodology.
