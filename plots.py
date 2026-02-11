import matplotlib.pyplot as plt
import pandas as pd

def plot_var_backtest(df: pd.DataFrame, title: str):
    """
    plots realized losses against one or more VaR estimates.

    parameters
        df: pd.DataFrame
            must contain the following: 
                loss: realized losses -returns
                VaR_hist: historical VaR 
                VaR_param: parametric VaR
                VaR_mc: Monte Carlo VaR 
            index datatime like
        title: str
            plot title
    """
    
    plt.figure() # create new figure, avoids overwriting 
    
    plt.plot(
        df.index,
        df["Loss"], 
        label="Realized Loss (-r)"
    ) # plot realized losses over time, actual port or asset losses
    
    if "VaR_hist" in df.columns:
        plt.plot(
            df.index, 
            df["VaR_hist"], 
            label="VaR (Historical)"
        ) # plot historical VaR if available, risk threshold
        
    if "VaR_param" in df.columns:
        plt.plot(
            df.index,
            df["VaR_param"],
            label="VaR (Parametric Normal)"
        ) # plot parametric normal VaR if available, Gaussian risk threshold
        
    if "VaR_mc" in df.columns:
        plt.plot(
            df.index, 
            df["VaR_mc"],
            label="VaR (Monte Carlo)"
        ) # plot Monte Carlo VaR if available, sim based port risk

    plt.title(title) # adds plot title
    plt.xlabel("Date") # add date @ x-axis
    plt.ylabel("Loss / VaR") # add loss magnitude @ y-axis
    plt.legend() # displays legend to distinguish plotted series
    plt.tight_layout() # adjust spacing to prevent label/title overlap

    plt.show() # render plot 
