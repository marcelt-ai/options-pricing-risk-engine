# Options Pricing & Risk Engine

Portfolio project integrating four modules:

- **Black-Scholes** - closed-form pricing of European call/put options.
- **Monte Carlo** - numerical verification of Black-Scholes via GBM simulation.
- **Volatility Modeling** - GARCH/EWMA to feed a better σ into Black-Scholes than flat historical volatility.
- **VaR** - Value at Risk (historical, parametric, Monte Carlo).

## Setup

```
pip install -r requirements.txt
```

## Status

- [x] Black-Scholes pricing
- [x] Option Greeks (Delta, Gamma, Vega, Theta, Rho)
- [x] Monte Carlo verification
- [x] Volatility modeling (GARCH/EWMA)
- [ ] VaR
