"""Statsmodels benchmarks: OLS with Fourier terms, and SARIMAX.

Why a linear model at all, when XGBoost is available?

1. **It is the literature's benchmark.** Electricity price forecasting has used
   regression with calendar dummies and exogenous load for decades. A gradient
   boosting result that is not compared against one is not comparable to
   anything published.
2. **It is interpretable.** Coefficients on residual load and heating degrees
   have signs you can check against economics. If residual load came out
   *negative*, something upstream is wrong — and no amount of feature importance
   from a tree gives you that check as directly.
3. **Diagnostics.** Residual autocorrelation tells you what structure the model
   has not captured, which is a more useful debugging signal than a single MAE.

**On SARIMAX specifically.** A seasonal ARIMA with a 48-period seasonal cycle on
~100k observations is computationally brutal and, in practice, barely better than
OLS with Fourier terms here — the seasonality is deterministic (it is driven by
the clock and by demand), not stochastic. So SARIMAX is fitted on a *daily mean*
series as a genuine reference point, and the half-hourly workhorse benchmark is
the Fourier OLS. That is a real modelling judgement, made explicitly rather than
by silently omitting the harder model.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

log = logging.getLogger(__name__)


def fit_ols(
    train: pd.DataFrame, features: list[str], target: str = "price"
) -> tuple[Any, list[str]]:
    """OLS with an intercept, on the same features the tree model sees.

    Rows with any missing feature are dropped: unlike XGBoost, OLS has no
    native NaN handling, and imputing here would be a different model than the
    one being compared.
    """
    frame = train[features + [target]].dropna()
    X = sm.add_constant(frame[features], has_constant="add")
    model = sm.OLS(frame[target], X).fit()
    return model, features


def predict_ols(model: Any, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = sm.add_constant(test[features], has_constant="add")
    X = X.fillna(X.median(numeric_only=True))
    return np.asarray(model.predict(X))


def coefficient_table(model: Any) -> pd.DataFrame:
    """Coefficients with t-statistics, largest effect first."""
    return (
        pd.DataFrame(
            {
                "coefficient": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p": model.pvalues,
            }
        )
        .assign(abs_t=lambda d: d["t"].abs())
        .sort_values("abs_t", ascending=False)
        .drop(columns="abs_t")
    )


def residual_diagnostics(model: Any, lags: int = 48) -> dict[str, Any]:
    """Ljung-Box on the residuals.

    A significant statistic means there is autocorrelation the model has not
    captured — which on half-hourly power data is essentially guaranteed, and
    saying so is more honest than not testing.
    """
    residuals = pd.Series(model.resid).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ljung = acorr_ljungbox(residuals, lags=[min(lags, len(residuals) // 5)], return_df=True)

    return {
        "ljung_box_stat": float(ljung["lb_stat"].iloc[0]),
        "ljung_box_p": float(ljung["lb_pvalue"].iloc[0]),
        "residual_std": float(residuals.std()),
        "durbin_watson": float(sm.stats.durbin_watson(residuals)),
        "r_squared": float(model.rsquared),
        "n": int(len(residuals)),
    }


def fit_sarimax_daily(
    panel: pd.DataFrame,
    order: tuple[int, int, int] = (2, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7),
) -> Any:
    """SARIMAX on the daily mean price, with a weekly seasonal cycle.

    Daily rather than half-hourly: a 48-period seasonal state space on 100k
    observations does not finish in reasonable time, and the intraday cycle is
    deterministic anyway (see the module docstring). The weekly cycle, by
    contrast, is a genuine candidate for stochastic seasonality.
    """
    daily = panel.groupby("settlement_date")["price"].mean().dropna()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D").interpolate()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            daily,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    log.info("SARIMAX%s x %s  AIC %.1f", order, seasonal_order, model.aic)
    return model
