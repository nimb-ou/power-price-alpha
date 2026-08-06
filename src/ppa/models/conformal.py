"""Conformalised quantile regression — making the interval mean what it says.

A quantile model fitted under pinball loss achieves its nominal coverage on the
data it was fitted on, and that is all it promises. Over the full walk-forward
the raw P10-P90 band covers **50.2%** of outturns against a nominal 80%.

Before concluding the estimator is broken, check it on a single fitted window —
that separates a bad estimator from a moving market:

    in-sample coverage   0.810   (nominal 0.80 — correct)
    out-of-sample        0.487   (same fit, next slice)

The estimator is fine. The market moved: across that slice the mean price went
from 36.4 to 42.7 GBP/MWh and the standard deviation from 14.8 to 21.3. An
interval calibrated on a calm window is simply too narrow for a volatile one,
and a P10-P90 band that contains the outturn half the time is worse than no band
at all, because a trading desk will size positions against it.

Conformal prediction (Vovk et al.; the quantile-regression form is Romano,
Patterson & Candès 2019) fixes this without touching the model. Hold out a
calibration window the model never saw, measure how far outside its own interval
the outturn actually fell on that window, and widen the interval by the
appropriate quantile of those misses. Under exchangeability this gives finite-
sample marginal coverage at the nominal rate — a guarantee, not an asymptotic
hope, and one that holds for *any* underlying model.

Two caveats that matter more than the guarantee, and both are measured rather
than asserted in `walkforward.score`:

**Exchangeability does not hold here.** Prices are non-stationary; that is the
whole reason the raw interval failed. So the calibration window is the most
*recent* slice of the training data rather than a random sample — the nearest
thing to exchangeable that a time series offers. Coverage is restored well but
not exactly, and it degrades as the test block gets further from the calibration
window.

**A single offset widens the whole day equally.** Uncertainty at 04:00 and at
the 17:30 peak are not the same quantity, and one scalar cannot express that.
`offset_by_period` computes a per-settlement-period offset instead, which costs
nothing extra and is what the strategy layer consumes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# How much of each training window is held back to calibrate the interval.
# 60 days is a compromise: too short and the offset is noisy, too long and it
# averages over a regime the model is no longer predicting. At 30-day refits
# this means the calibration window is always the two months immediately
# preceding the block being forecast.
CALIBRATION_DAYS = 60


@dataclass(frozen=True)
class Conformaliser:
    """Additive offsets that restore coverage, fitted on a calibration window."""

    alpha: float
    global_offset: float
    per_period: dict[int, float]
    n_calibration: int

    def apply(
        self, lower: np.ndarray, upper: np.ndarray, periods: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Widen an interval. Falls back to the global offset per period."""
        if periods is None:
            offsets = np.full(len(lower), self.global_offset)
        else:
            offsets = np.array(
                [self.per_period.get(int(p), self.global_offset) for p in periods]
            )
        return lower - offsets, upper + offsets

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "global_offset": self.global_offset,
            "n_calibration": self.n_calibration,
            "min_period_offset": min(self.per_period.values()) if self.per_period else None,
            "max_period_offset": max(self.per_period.values()) if self.per_period else None,
        }


def conformity_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """How far outside its own interval each observation fell.

    Negative when the observation was comfortably inside, which is what allows
    the offset to be *negative* — a model whose interval is too wide gets it
    narrowed. A one-sided score, clipped at zero, could only ever widen, and
    would leave an over-wide interval over-wide forever.
    """
    return np.maximum(lower - y, y - upper)


def _quantile_level(n: int, alpha: float) -> float:
    """The finite-sample corrected level, (1-alpha)(1+1/n), capped at 1.

    The `1 + 1/n` is not a rounding detail. It is what converts an asymptotic
    statement into the finite-sample guarantee, by accounting for the test point
    itself being one of the n+1 exchangeable observations.
    """
    return float(min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)) if n else 1.0


def fit(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.2,
    periods: np.ndarray | None = None,
    min_per_period: int = 30,
) -> Conformaliser:
    """Fit offsets from a calibration window the model has not seen."""
    scores = conformity_scores(np.asarray(y), np.asarray(lower), np.asarray(upper))
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return Conformaliser(alpha=alpha, global_offset=0.0, per_period={}, n_calibration=0)

    global_offset = float(np.quantile(scores, _quantile_level(n, alpha)))

    per_period: dict[int, float] = {}
    if periods is not None:
        frame = pd.DataFrame({"period": np.asarray(periods), "score": scores})
        for period, group in frame.groupby("period"):
            # A period with too few observations gets the global offset rather
            # than a quantile estimated from a handful of points.
            if len(group) >= min_per_period:
                level = _quantile_level(len(group), alpha)
                # `period` is a groupby key, which pandas types as a union wide
                # enough to include datetimes. It is an int here by construction
                # — the column was built from settlement periods above.
                per_period[int(cast(int, period))] = float(
                    np.quantile(group["score"], level)
                )

    return Conformaliser(
        alpha=alpha,
        global_offset=global_offset,
        per_period=per_period,
        n_calibration=n,
    )


def split_calibration(
    train: pd.DataFrame,
    calibration_days: int = CALIBRATION_DAYS,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training window into fit and calibration parts, by time.

    The calibration part is the *most recent* slice, not a random sample. A
    random split would satisfy exchangeability on paper and be the wrong choice
    in practice: it would calibrate the interval against the average of the
    whole training window, when what the model is about to forecast is the
    period immediately after its end.
    """
    dates = pd.to_datetime(train[date_column])
    boundary = dates.max() - pd.Timedelta(days=calibration_days)
    fit_part = train[dates <= boundary]
    calibration_part = train[dates > boundary]

    if len(fit_part) == 0 or len(calibration_part) == 0:
        # Not enough history to hold anything back. Returning the whole window
        # for both would calibrate on the fitting data and produce an offset of
        # roughly zero — silently giving back the uncalibrated interval.
        log.warning(
            "training window too short to hold out %d calibration days; "
            "interval will not be conformalised",
            calibration_days,
        )
        return train, train.iloc[:0]

    return fit_part, calibration_part
