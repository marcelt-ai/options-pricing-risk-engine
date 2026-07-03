import numpy as np


def historical_var(returns: np.ndarray, confidence: float = 0.95, portfolio_value: float | None = None) -> float:
    """Historical (empirical) VaR: the loss at the given confidence level, read
    directly off the empirical distribution of past returns.

    Returns a positive fraction (e.g. 0.025 = 2.5% loss), or a currency amount
    if `portfolio_value` is given.
    """
    returns = np.asarray(returns)
    loss_percentile = (1 - confidence) * 100
    var_pct = -np.percentile(returns, loss_percentile)
    return var_pct * portfolio_value if portfolio_value is not None else var_pct


def historical_cvar(returns: np.ndarray, confidence: float = 0.95, portfolio_value: float | None = None) -> float:
    """Historical (empirical) CVaR / Expected Shortfall: the average loss among
    the scenarios at least as bad as the VaR threshold.

    Returns a positive fraction, or a currency amount if `portfolio_value` is given.
    """
    returns = np.asarray(returns)
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= threshold]
    cvar_pct = -tail_losses.mean()
    return cvar_pct * portfolio_value if portfolio_value is not None else cvar_pct
