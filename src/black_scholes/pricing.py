import numpy as np
from scipy.stats import norm


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * np.sqrt(T)


def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes price of a European call option.

    S: spot price, K: strike, T: time to expiry (years),
    r: risk-free rate (annualized), sigma: volatility (annualized).
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes price of a European put option. Same parameters as call_price."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
