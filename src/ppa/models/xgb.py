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


def importance(model: XGBRegressor, features: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": features, "gain": model.feature_importances_})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
