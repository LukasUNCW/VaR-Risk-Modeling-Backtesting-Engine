import pandas as pd
import yfinance as yf

def get_prices(tickers: list[str], start: str = "2015-01-01") -> pd.DataFrame:
    """
    downloads adjusted historical price data for one or more tickers (single asset vs port)

    parameters
        tickers : list[str]
            list of ticker symbols (e.g., ["SPY", "QQQ", "TLT"])
        start : str
            Start date for historical data (YYYY-MM-DD)
    returns
            pandas DataFrame indexed by date with one column per ticker
            containing adjusted closing prices.
    """
    df = yf.download( #downloading price data from Yahoo Finance (yfinance)
        tickers, # individual stocks
        start=start, # start date
        auto_adjust=True,   # ensures prices are total return consistent
        progress=False
    )

    # edge case, if yf returns no data, raise exception
    if df.empty:
        raise ValueError("No data returned. Check tickers/start date.")

    # yfinance returns multi-index columns when multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].copy()
    # only activates else statement if only a single ticker is requested (single-level column)
    else:
        prices = df[["Close"]].copy()
        # rename column to match ticker name, makes sure output format is consistent
        prices.columns = tickers

    prices = prices.dropna(how="all") # drops rows where all tickers are missing prices (non-trading days or partial data)

    return prices
