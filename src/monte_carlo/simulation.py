import numpy as np


def simulate_terminal_prices(
    S: float, T: float, r: float, sigma: float, n_sims: int, seed: int | None = None
) -> np.ndarray:
    """Simulate S_T under the risk-neutral GBM, using the exact (closed-form) solution.

    Valid only for the terminal value of a European option (no path dependency).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)
    return S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
