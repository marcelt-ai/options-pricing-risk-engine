import numpy as np

from monte_carlo.simulation import simulate_terminal_prices


def mc_var(
    S: float,
    T: float,
    r: float,
    sigma: float,
    confidence: float = 0.95,
    n_sims: int = 100_000,
    seed: int | None = None,
    portfolio_value: float | None = None,
) -> float:
    """Monte Carlo VaR: simulate S_T under GBM, take the empirical percentile
    of simulated returns over the horizon T. Reuses the same simulation engine
    as the Monte Carlo pricing module.

    Returns a positive fraction, or a currency amount if `portfolio_value` is given.
    """
    S_T = simulate_terminal_prices(S, T, r, sigma, n_sims, seed)
    simple_returns = S_T / S - 1
    loss_percentile = (1 - confidence) * 100
    var_pct = -np.percentile(simple_returns, loss_percentile)
    return var_pct * portfolio_value if portfolio_value is not None else var_pct


def mc_cvar(
    S: float,
    T: float,
    r: float,
    sigma: float,
    confidence: float = 0.95,
    n_sims: int = 100_000,
    seed: int | None = None,
    portfolio_value: float | None = None,
) -> float:
    """Monte Carlo CVaR / Expected Shortfall: average simulated return among
    the scenarios at least as bad as the VaR threshold.

    Returns a positive fraction, or a currency amount if `portfolio_value` is given.
    """
    S_T = simulate_terminal_prices(S, T, r, sigma, n_sims, seed)
    simple_returns = S_T / S - 1
    threshold = np.percentile(simple_returns, (1 - confidence) * 100)
    tail_losses = simple_returns[simple_returns <= threshold]
    cvar_pct = -tail_losses.mean()
    return cvar_pct * portfolio_value if portfolio_value is not None else cvar_pct
