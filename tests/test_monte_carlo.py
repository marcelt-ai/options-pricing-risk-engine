import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from black_scholes.pricing import call_price, put_price
from monte_carlo.pricing import mc_call_price, mc_put_price

S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
N_SIMS = 200_000
SEED = 42


def test_mc_call_matches_black_scholes():
    bs = call_price(S, K, T, r, sigma)
    mc = mc_call_price(S, K, T, r, sigma, N_SIMS, seed=SEED)
    assert mc.ci_low < bs < mc.ci_high


def test_mc_put_matches_black_scholes():
    bs = put_price(S, K, T, r, sigma)
    mc = mc_put_price(S, K, T, r, sigma, N_SIMS, seed=SEED)
    assert mc.ci_low < bs < mc.ci_high


def test_mc_std_error_shrinks_with_more_sims():
    small = mc_call_price(S, K, T, r, sigma, 1_000, seed=SEED)
    large = mc_call_price(S, K, T, r, sigma, 200_000, seed=SEED)
    assert large.std_error < small.std_error
