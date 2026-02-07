import pandas as pd
import yfinance as yf

def get_prices(tickers: list[str], start: str = "2015-01-01") -> pd.DataFrame:
    """
    Download adjusted prices for tickers.
    Returns DataFrame with columns=tickers, index=datetime.
    """
    df = yf.download(
        tickers,
        start=start,
        auto_adjust=True,   # adjusts for splits/dividends
        progress=False
    )

    if df.empty:
        raise ValueError("No data returned. Check tickers/start date.")

    # yfinance returns multi-index columns when multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].copy()
    else:
        prices = df[["Close"]].copy()
        prices.columns = tickers

    prices = prices.dropna(how="all")
    return prices