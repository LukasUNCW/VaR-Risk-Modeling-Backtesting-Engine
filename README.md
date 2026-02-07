# VaR Risk Modeling Backtesting Engine

A Python-based  project that implements Value-at-Risk (VaR) models and industry-standard backtesting to measure and validate market risk for single and multi asset portfolios.

This project mirrors how risk engines are built and evaluated in real financial institutions, emphasizing:
- statistical rigor
- clean software design
- reproducibility
- realistic assumptions (rolling windows, no look ahead bias)

## Key Features

### Risk Models
- Historical VaR (non-parametric, quantile-based): Estimates risk by taking the empirical loss quantile from historical returns without assuming any specific return distribution.
- Parametric VaR (Normal): Models risk by assuming returns follow a normal distribution and computing VaR from the rolling mean and standard deviation. 
- Monte Carlo VaR: Simulates thousands of correlated asset return scenarios using estimated means and covariances to estimate portfolio level VaR.
- Expected Shortfall (CVaR): Measures the average loss conditional on losses exceeding the VaR threshold, providing insight into the severity of extreme outcomes. 

### Backtesting
- Rolling-window VaR estimation
- Exception detection (losses exceeding VaR)
- Kupiec Proportion-of-Failures (POF) test for statistical validation

### Visualization
- Market data sourced via Yahoo Finance (yfinance) Python library
- Log return modeling for numerical stability
- VaR vs realized-loss plots for diagnostic analysis

## How it works

Files: 
- data.py -> Downloads and cleans price data
- returns.py -> Computes log returns and portfolio returns
- models.py -> Implements Historical, Parametric, Monte Carlo VaR + CVaR
- backtest.py -> Detects exceptions and runs Kupiec POF test
- plots.py -> Produces VaR vs loss plots

Scripts:
</br>
run_single_asset.py
- Loads one asset (e.g., SPY)
- Computes rolling Historical + Parametric VaR
- Runs backtesting
- Plots results

run the following inside desired IDE terminal and inside correct directory "py -3.14 -m scripts.run_single_asset"

run_port.py
- Loads multiple assets, this is configurable
- Computes portfolio returns
- Runs Historical, Parametric, Monte Carlo VaR
- Backtests all models
- Plots portfolio risk

run the following inside desired IDE terminal and inside correct directory "py -3.14 -m scripts.run_port"

## Visualizations

Rolling VaR Backtest (VaR vs Realized Loss)

<img width="1260" height="938" alt="var_backtest_portfolio" src="https://github.com/user-attachments/assets/b94c85da-f32f-4b86-91b0-a459ed3b113e" />

What this chart shows:
- X-axis: Time (trading days)
- Y-axis: Loss magnitude (positive = bad)
- Lines:
  - Realized loss: actual next-day porfolio loss
  - VaR lines (Historical / Parametric / Monte Carlo): predicted maximum loss at confidence level α (e.g., 99%)

This is a rolling, out of sample risk forecast.

How to read it: 
- Most of the time:
  - The loss line stays below VaR
- Occasionally:
  - The loss spikes above VaR -> these are exceptions
- During volatile periods:
  - VaR lines rise (risk adapts)
- During calm periods:
  - VaR compresses (risk declines)

Exception Timeline (VaR Failures)

<img width="1260" height="938" alt="exceptions_mc_portfolio" src="https://github.com/user-attachments/assets/98dc224f-7e5d-448c-bd7b-4b52914c5580" />

What this chart shows:
- Each dot = one VaR exception
- X-axis: date
- Y-axis: 1 -> loss exceeded VaR

This is a binary diagnositc of VaR correctness.

How to read it: 
- You expect exceptions at roughly:
  - (1 − α)% of days
  - For α = 99% -> ~1% of days

What clustering means:
- Clusters -> VaR is slow to react
- Often happens:
  - During crises
  - When volatility jumps suddenly
- Parametric VaR trends to cluster more
- Monte Carlo usually improves clustering behavior

Loss Distribution with VaR Threshold

<img width="1260" height="938" alt="loss_dist_var_portfolio" src="https://github.com/user-attachments/assets/ddf4892c-8788-44df-aa76-a49c5a2a0770" />



