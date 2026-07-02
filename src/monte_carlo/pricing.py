from typing import NamedTuple

import numpy as np

from .simulation import simulate_terminal_prices


class MCResult(NamedTuple):
    price: float
    std_error: float
    ci_low: float
    ci_high: float


def _mc_price(payoffs: np.ndarray, r: float, T: float) -> MCResult:
    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    # std_error of the mean = sample std / sqrt(n)
    std_error = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return MCResult(
        price=price,
        std_error=std_error,
        ci_low=price - 1.96 * std_error,
        ci_high=price + 1.96 * std_error,
    )


def mc_call_price(
    S: float, K: float, T: float, r: float, sigma: float, n_sims: int, seed: int | None = None
) -> MCResult:
    S_T = simulate_terminal_prices(S, T, r, sigma, n_sims, seed)
    payoffs = np.maximum(S_T - K, 0.0)
    return _mc_price(payoffs, r, T)


def mc_put_price(
    S: float, K: float, T: float, r: float, sigma: float, n_sims: int, seed: int | None = None
) -> MCResult:
    S_T = simulate_terminal_prices(S, T, r, sigma, n_sims, seed)
    payoffs = np.maximum(K - S_T, 0.0)
    return _mc_price(payoffs, r, T)
