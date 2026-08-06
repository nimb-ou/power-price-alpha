"""The XGBoost day-ahead forecaster.

Hyperparameters are chosen for a time-series regression with ~100k rows and ~45
features, and two of them are unconventional enough to be worth stating:

- **`objective="reg:absoluteerror"`.** We are judged on MAE, so we optimise MAE.
  The default squared error would chase price spikes, because one 300 GBP/MWh
  scarcity hour contributes as much squared loss as a hundred ordinary hours.
  That produces a model that is better at four hours a year and worse the rest
  of the time — the exact failure the metrics module warns about.

- **No `early_stopping` on a random split.** Any validation split for early
  stopping must be a *later* time slice, never a random sample; a random
  holdout leaks future information into the stopping decision. The walk-forward
  harness handles this by fixing the tree count and relying on regularisation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ppa.config import STUDY

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    # See module docstring: the metric we report is the metric we optimise.
    "objective": "reg:absoluteerror",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": STUDY.random_seed,
}


def build(params: dict[str, Any] | None = None) -> XGBRegressor:
    return XGBRegressor(**{**DEFAULT_PARAMS, **(params or {})})


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str = "price",
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, XGBRegressor]:
    """Fit on a training window and predict the next block.

    XGBoost handles NaN natively by learning a default direction per split, so
    early rows with incomplete lags are kept rather than dropped. Dropping them
    would silently remove the first two weeks of every training window.
    """
    model = build(params)
    model.fit(train[features], train[target])
    return model.predict(test[features]), model


# The interval the strategy and the report both use. Deliberately not 5/95: at
# 90% nominal coverage an empirical 88% is a readable miss, whereas at 99% the
# tail is so thin that 77k observations barely pin it down.
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


def fit_predict_quantiles(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str = "price",
    quantiles: tuple[float, ...] = QUANTILES,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Fit one multi-quantile model and predict all quantiles at once.

    Returned shape is `(len(test), len(quantiles))`, column order matching
    `quantiles`.

    One model rather than three. Fitting separate models per quantile is the
    common recipe and it has a defect that shows up immediately on power prices:
    nothing couples the three fits, so on a volatile evening the P10 model and
    the P90 model can cross, and an interval whose lower bound exceeds its upper
    bound is not a thing you can hand to a trading desk. XGBoost's multi-quantile
    objective fits them jointly against a shared tree structure, which does not
    make crossing impossible but makes it rare — and `walkforward.score` counts
    the crossings that remain rather than assuming there are none.

    The point forecast still comes from the separate `reg:absoluteerror` model.
    The P50 here is a *median* fitted under pinball loss; the two land close but
    are not the same estimator, and reporting a headline MAE from whichever one
    happened to win would be exactly the kind of quiet metric-shopping the rest
    of this repo is built to avoid.
    """
    model = XGBRegressor(
        **{
            **DEFAULT_PARAMS,
            "objective": "reg:quantileerror",
            "quantile_alpha": np.array(quantiles),
            **(params or {}),
        }
    )
    model.fit(train[features], train[target])
    predicted: np.ndarray = np.asarray(model.predict(test[features]))
    return predicted.reshape(len(test), len(quantiles))


def importance(model: XGBRegressor, features: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": features, "gain": model.feature_importances_})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
