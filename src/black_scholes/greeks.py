import numpy as np
from scipy.stats import norm

from .pricing import _d1, _d2


def delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = _d1(S, K, T, r, sigma)
    return norm.cdf(d1)


def delta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = _d1(S, K, T, r, sigma)
    return norm.cdf(d1) - 1


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Same for call and put."""
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Same for call and put. Raw value per unit change in sigma (e.g. 0.01 -> 1%)."""
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def theta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Raw value per unit change in T (i.e. per year). Divide by 365 for per-day."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    return term1 - term2


def theta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Raw value per unit change in T (i.e. per year). Divide by 365 for per-day."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return term1 + term2


def rho_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Raw value per unit change in r (e.g. 0.01 -> 1%)."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    return K * T * np.exp(-r * T) * norm.cdf(d2)


def rho_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Raw value per unit change in r (e.g. 0.01 -> 1%)."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)
