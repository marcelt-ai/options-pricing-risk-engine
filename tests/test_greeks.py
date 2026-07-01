import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from black_scholes.greeks import (
    delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put,
)

# S=100, K=100, T=1, r=0.05, sigma=0.2
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2


def test_delta():
    assert round(delta_call(S, K, T, r, sigma), 4) == 0.6368
    assert round(delta_put(S, K, T, r, sigma), 4) == -0.3632


def test_delta_call_minus_delta_put_is_one():
    # Delta_call - Delta_put = 1 always (put-call parity differentiated w.r.t. S)
    assert round(delta_call(S, K, T, r, sigma) - delta_put(S, K, T, r, sigma), 8) == 1


def test_gamma():
    assert round(gamma(S, K, T, r, sigma), 6) == 0.018762


def test_vega():
    assert round(vega(S, K, T, r, sigma), 3) == 37.524


def test_theta():
    assert round(theta_call(S, K, T, r, sigma), 3) == -6.414
    assert round(theta_put(S, K, T, r, sigma), 4) == -1.6579


def test_rho():
    assert round(rho_call(S, K, T, r, sigma), 4) == 53.2325
    assert round(rho_put(S, K, T, r, sigma), 4) == -41.8905
