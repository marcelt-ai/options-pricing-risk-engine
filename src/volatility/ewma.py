import numpy as np

TRADING_DAYS_PER_YEAR = 252


def ewma_volatility(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Daily EWMA (RiskMetrics) volatility series.

    sigma_t^2 = lam * sigma_{t-1}^2 + (1 - lam) * r_{t-1}^2
    The first value is seeded with the sample variance of `returns`.
    """
    returns = np.asarray(returns)
    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return np.sqrt(sigma2)


def latest_ewma_vol(returns: np.ndarray, lam: float = 0.94, annualize: bool = True) -> float:
    """Most recent EWMA volatility estimate, optionally annualized."""
    sigma = ewma_volatility(returns, lam)[-1]
    return sigma * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else sigma
