import numpy as np
import pandas as pd
import yfinance as yf


def fetch_log_returns(ticker: str, period: str = "2y") -> pd.Series:
    """Download close prices for `ticker` and return daily log returns."""
    # yfinance returns a single-column DataFrame (with a "Ticker" column level) even
    # for one ticker; squeeze it down to a plain Series so downstream .mean()/.std()
    # return scalars instead of length-1 Series.
    prices = yf.download(ticker, period=period, progress=False, auto_adjust=True)["Close"].squeeze()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = "log_return"
    return log_returns


def fetch_spot_price(ticker: str) -> float:
    """Most recent close price for `ticker`."""
    prices = yf.download(ticker, period="5d", progress=False, auto_adjust=True)["Close"]
    return float(prices.iloc[-1].item())
