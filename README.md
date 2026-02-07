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
