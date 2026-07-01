import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from black_scholes.pricing import call_price, put_price


def test_call_price_known_value():
    # S=100, K=100, T=1, r=0.05, sigma=0.2 -> textbook value ~10.4506
    price = call_price(S=100, K=100, T=1, r=0.05, sigma=0.2)
    assert round(price, 4) == 10.4506


def test_put_price_known_value():
    price = put_price(S=100, K=100, T=1, r=0.05, sigma=0.2)
    assert round(price, 4) == 5.5735


def test_put_call_parity():
    S, K, T, r, sigma = 100, 95, 0.5, 0.03, 0.25
    c = call_price(S, K, T, r, sigma)
    p = put_price(S, K, T, r, sigma)
    # C - P = S - K * e^(-rT)
    assert round(c - p, 6) == round(S - K * 2.718281828459045 ** (-r * T), 6)
