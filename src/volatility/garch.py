import numpy as np
from arch import arch_model

from .ewma import TRADING_DAYS_PER_YEAR


def fit_garch(returns: np.ndarray):
    """Fit a GARCH(1,1) model. `arch` expects returns scaled as percentages
    (e.g. 0.01 -> 1.0) for numerical stability of its optimizer.
    """
    returns_pct = np.asarray(returns) * 100
    model = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal")
    return model.fit(disp="off")


def garch_forecast_vol(fitted_result, horizon: int = 1, annualize: bool = True) -> float:
    """Forecast volatility `horizon` days ahead from a fitted GARCH model."""
    forecast = fitted_result.forecast(horizon=horizon)
    variance_pct2 = forecast.variance.iloc[-1, -1]
    sigma = np.sqrt(variance_pct2) / 100
    return sigma * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else sigma
