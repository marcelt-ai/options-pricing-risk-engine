import numpy as np
import pandas as pd
import yfinance as yf


def fetch_log_returns(ticker: str, period: str = "2y") -> pd.Series:
    """Download close prices for `ticker` and return daily log returns."""
    prices = yf.download(ticker, period=period, progress=False)["Close"]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = "log_return"
    return log_returns
