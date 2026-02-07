# VaR Risk Modeling Backtesting Engine

A Python-based  project that implements Value-at-Risk (VaR) models and industry-standard backtesting to measure and validate market risk for single and multi asset portfolios.

This project mirrors how risk engines are built and evaluated in real financial institutions, emphasizing:
- statistical rigor
- clean software design
- reproducibility
- realistic assumptions (rolling windows, no look ahead bias)

---

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

---

## Project Structure
<img width="581" height="357" alt="image" src="https://github.com/user-attachments/assets/a2134880-ecfe-4b2e-a07a-152bee29c8cf" />
</br>
Make sure to have your files set up as listed above. 

## How it works

var-lab/ -> the engine
</br>
This folder is the libray. Nothing in this folder should be run directly

files: 
data.py -> Downloads and cleans price data
returns.py -> Computes log returns and portfolio returns
models.py -> Implements Historical, Parametric, Monte Carlo VaR + CVaR
backtest.py -> Detects exceptions and runs Kupiec POF test
plots.py -> Produces VaR vs loss plots

scripts/ -> the applications
</br>
These files wire everything together and are what users will actually run. 

run_single_asset.py
- Loads one asset (e.g., SPY)
- Computes rolling Historical + Parametric VaR
- Runs backtesting
- Plots results
Run it with the following command inside your desired IDE:
py -3.14 -m scripts.run_single_asset

run_port.py
- Loads multiple assets, this is configurable
- Computes portfolio returns
- Runs Historical, Parametric, Monte Carlo VaR
- Backtests all models
- Plots portfolio risk
Run it with the following command inside your desired IDE:
py -3.14 -m scripts.run_port
