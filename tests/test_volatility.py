import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volatility.ewma import ewma_volatility, latest_ewma_vol
from volatility.garch import fit_garch, garch_forecast_vol

rng = np.random.default_rng(42)
RETURNS = rng.normal(loc=0.0, scale=0.01, size=500)


def test_ewma_matches_manual_recursion():
    lam = 0.94
    sigma = ewma_volatility(RETURNS, lam)
    # manually recompute the first few steps
    sigma2_0 = np.var(RETURNS)
    sigma2_1 = lam * sigma2_0 + (1 - lam) * RETURNS[0] ** 2
    assert np.isclose(sigma[0], np.sqrt(sigma2_0))
    assert np.isclose(sigma[1], np.sqrt(sigma2_1))


def test_ewma_reacts_to_a_shock():
    # sigma[t] is computed from returns[t-1], so the shock's effect shows up
    # one step after it occurs - append a calm return after the shock to observe it.
    calm = np.full(200, 0.001)
    shocked = np.concatenate([calm, [0.10], [0.001]])
    sigma = ewma_volatility(shocked)
    assert sigma[-1] > sigma[-2]


def test_latest_ewma_vol_is_annualized():
    daily = latest_ewma_vol(RETURNS, annualize=False)
    annual = latest_ewma_vol(RETURNS, annualize=True)
    assert np.isclose(annual, daily * np.sqrt(252))


def test_garch_forecast_is_positive_and_reasonable():
    result = fit_garch(RETURNS)
    sigma = garch_forecast_vol(result)
    assert 0 < sigma < 2.0  # annualized vol should be a small positive number, not e.g. 50.0
