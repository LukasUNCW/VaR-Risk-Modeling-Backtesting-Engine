import matplotlib.pyplot as plt
import pandas as pd

def plot_var_backtest(df: pd.DataFrame, title: str):
    """
    df must contain columns: Loss, VaR_hist, VaR_param (optional VaR_mc)
    """
    plt.figure()
    plt.plot(df.index, df["Loss"], label="Realized Loss (-r)")
    if "VaR_hist" in df.columns:
        plt.plot(df.index, df["VaR_hist"], label="VaR (Historical)")
    if "VaR_param" in df.columns:
        plt.plot(df.index, df["VaR_param"], label="VaR (Parametric Normal)")
    if "VaR_mc" in df.columns:
        plt.plot(df.index, df["VaR_mc"], label="VaR (Monte Carlo)")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Loss / VaR")
    plt.legend()
    plt.tight_layout()
    plt.show()