"""Seasonal naive baselines.

"Beat a seasonal naive baseline" is only meaningful if you say *which* one, and
say it before you see the results. There are several, they differ by a lot, and
choosing the weakest after the fact is the most common way to manufacture an
impressive improvement.

Three candidates, all respecting the day-ahead information set (nothing later
than 11:00 on T-1):

| Variant | Prediction for period p on day T | Legitimate? |
|---|---|---|
| `same_period_last_week` | price at p on T-7 | yes |
| `same_period_two_days_ago` | price at p on T-2 | yes |
| `last_known_morning_shape` | T-2's price at p, scaled by T-1 morning vs T-2 morning | yes |
| ~~`same_period_yesterday`~~ | price at p on T-1 | **no — not knowable at gate closure** |

**`same_period_last_week` is the headline baseline**, declared here in code
before any model is fitted. It is the standard choice in the electricity price
forecasting literature (Weron 2014 and after) because a week-lag preserves both
the intraday shape and the weekday/weekend distinction, which a two-day lag does
not.

The fourth row is included precisely because it is the tempting one. A model
compared against `same_period_yesterday` would look worse than it is *and* that
baseline is not implementable — which is the sort of thing worth noticing before
someone else does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HEADLINE_BASELINE = "same_period_last_week"


def same_period_last_week(features: pd.DataFrame) -> np.ndarray:
    """T-7, same settlement period. The declared headline baseline."""
    return features["price_lag_7d"].to_numpy()


def same_period_two_days_ago(features: pd.DataFrame) -> np.ndarray:
    """T-2, same settlement period. The freshest legitimate period-level lag."""
    return features["price_lag_2d"].to_numpy()


def last_known_morning_shape(features: pd.DataFrame) -> np.ndarray:
    """T-2's intraday shape, rescaled by how T-1's morning compared to T-2's.

    A genuinely competitive baseline: it keeps the shape of a recent day and
    applies the one piece of same-week level information that gate closure
    allows. Any model has to beat this, not just a raw lag.
    """
    shape = features["price_lag_2d"].to_numpy()
    recent = features["price_am_mean"].to_numpy()

    # price_am_mean is T-1's morning; shifting it by another day approximates
    # T-2's morning without adding a new leakage surface.
    reference = (
        features.groupby("settlement_date")["price_am_mean"]
        .transform("mean")
        .shift(48)
        .to_numpy()
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(reference) > 1e-6, recent / reference, 1.0)
    ratio = np.clip(np.nan_to_num(ratio, nan=1.0), 0.5, 2.0)
    return shape * ratio


BASELINES = {
    "same_period_last_week": same_period_last_week,
    "same_period_two_days_ago": same_period_two_days_ago,
    "last_known_morning_shape": last_known_morning_shape,
}


def predict(features: pd.DataFrame, variant: str = HEADLINE_BASELINE) -> np.ndarray:
    if variant not in BASELINES:
        raise KeyError(f"unknown baseline {variant!r}; choose from {sorted(BASELINES)}")
    return BASELINES[variant](features)


def predict_all(features: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: fn(features) for name, fn in BASELINES.items()})
