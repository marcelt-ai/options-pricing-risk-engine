import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from var.historical import historical_var, historical_cvar
from var.parametric import parametric_var, parametric_cvar
from var.monte_carlo import mc_var, mc_cvar

rng = np.random.default_rng(42)
RETURNS = rng.normal(loc=0.0005, scale=0.02, size=2000)


def test_historical_var_is_positive_fraction():
    var = historical_var(RETURNS, confidence=0.95)
    assert 0 < var < 1


def test_historical_var_scales_to_portfolio_value():
    var_pct = historical_var(RETURNS, confidence=0.95)
    var_usd = historical_var(RETURNS, confidence=0.95, portfolio_value=1_000_000)
    assert np.isclose(var_usd, var_pct * 1_000_000)


def test_higher_confidence_means_higher_var():
    var_95 = historical_var(RETURNS, confidence=0.95)
    var_99 = historical_var(RETURNS, confidence=0.99)
    assert var_99 > var_95


def test_parametric_matches_normal_formula():
    from scipy.stats import norm
    mu, sigma = RETURNS.mean(), RETURNS.std(ddof=1)
    expected = -(mu + norm.ppf(0.05) * sigma)
    assert np.isclose(parametric_var(RETURNS, confidence=0.95), expected)


def test_mc_var_close_to_parametric_for_normal_gbm_returns():
    # Under GBM with these parameters, simple returns are approximately normal
    # over a short horizon, so MC VaR should be close to the parametric estimate.
    S, T, r, sigma = 100, 1 / 252, 0.05, 0.2
    mc = mc_var(S, T, r, sigma, confidence=0.95, n_sims=200_000, seed=42)
    gbm_returns = np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * rng.standard_normal(200_000)) - 1
    param = parametric_var(gbm_returns, confidence=0.95)
    assert abs(mc - param) < 0.01


def test_cvar_is_never_smaller_than_var():
    # CVaR (average of the tail) must be at least as large as VaR (the tail's boundary)
    assert historical_cvar(RETURNS, confidence=0.95) >= historical_var(RETURNS, confidence=0.95)
    assert parametric_cvar(RETURNS, confidence=0.95) >= parametric_var(RETURNS, confidence=0.95)

    S, T, r, sigma = 100, 1 / 252, 0.05, 0.2
    assert mc_cvar(S, T, r, sigma, confidence=0.95, n_sims=200_000, seed=42) >= \
        mc_var(S, T, r, sigma, confidence=0.95, n_sims=200_000, seed=42)


def test_parametric_cvar_matches_closed_form():
    from scipy.stats import norm
    mu, sigma = RETURNS.mean(), RETURNS.std(ddof=1)
    alpha = 0.05
    z = norm.ppf(alpha)
    expected = -mu + sigma * norm.pdf(z) / alpha
    assert np.isclose(parametric_cvar(RETURNS, confidence=0.95), expected)
