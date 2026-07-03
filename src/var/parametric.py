import numpy as np
from scipy.stats import norm


def parametric_var(returns: np.ndarray, confidence: float = 0.95, portfolio_value: float | None = None) -> float:
    """Parametric (variance-covariance) VaR: assumes returns ~ Normal(mu, sigma).

    VaR = -(mu + z * sigma), where z is the (negative) normal quantile at
    (1 - confidence). Returns a positive fraction, or a currency amount if
    `portfolio_value` is given.
    """
    returns = np.asarray(returns)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1 - confidence)
    var_pct = -(mu + z * sigma)
    return var_pct * portfolio_value if portfolio_value is not None else var_pct


def parametric_cvar(returns: np.ndarray, confidence: float = 0.95, portfolio_value: float | None = None) -> float:
    """Parametric (variance-covariance) CVaR / Expected Shortfall, using the
    closed-form expression for the tail mean of a Normal(mu, sigma).

    CVaR = -mu + sigma * phi(z) / alpha, where alpha = 1 - confidence and
    phi is the standard normal density evaluated at z = ppf(alpha).
    Returns a positive fraction, or a currency amount if `portfolio_value` is given.
    """
    returns = np.asarray(returns)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    alpha = 1 - confidence
    z = norm.ppf(alpha)
    cvar_pct = -mu + sigma * norm.pdf(z) / alpha
    return cvar_pct * portfolio_value if portfolio_value is not None else cvar_pct
