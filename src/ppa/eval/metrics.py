"""Forecast error metrics.

MAE is the headline. Electricity prices spike hard and go negative, and both
break the usual alternatives:

- **RMSE** is dominated by a handful of scarcity hours. A model can win on RMSE
  by being better at four hours a year and worse the rest of the time.
- **MAPE** divides by price. GB prices touch zero and go negative, so MAPE is
  undefined or explosive exactly where it matters most. It is reported here only
  on a filtered subset, with the exclusions counted.

The headline improvement figure is therefore **MAE reduction versus the declared
seasonal-naive baseline**, computed on the same rows for both.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _aligned(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows either series cannot score.

    Both models must be judged on identical rows, or a model that quietly
    predicts nothing on hard days wins by abstention.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def mae(y_true: Any, y_pred: Any) -> float:
    t, p = _aligned(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if len(t) else float("nan")


def rmse(y_true: Any, y_pred: Any) -> float:
    t, p = _aligned(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2))) if len(t) else float("nan")


def smape(y_true: Any, y_pred: Any) -> float:
    """Symmetric MAPE — bounded, and defined at zero unlike plain MAPE."""
    t, p = _aligned(y_true, y_pred)
    denominator = (np.abs(t) + np.abs(p)) / 2
    mask = denominator > 1e-6
    return float(np.mean(np.abs(t[mask] - p[mask]) / denominator[mask]) * 100) if mask.any() else float("nan")


def mape_filtered(y_true: Any, y_pred: Any, floor: float = 10.0) -> dict[str, float]:
    """MAPE on |price| >= floor, reporting how much was excluded."""
    t, p = _aligned(y_true, y_pred)
    mask = np.abs(t) >= floor
    return {
        "mape": float(np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100) if mask.any() else float("nan"),
        "excluded_rows": int((~mask).sum()),
        "excluded_share": float((~mask).mean()) if len(t) else float("nan"),
    }


def improvement(baseline_mae: float, model_mae: float) -> float:
    """Fractional MAE reduction. 0.18 means an 18% improvement."""
    if not np.isfinite(baseline_mae) or baseline_mae == 0:
        return float("nan")
    return float((baseline_mae - model_mae) / baseline_mae)


def diebold_mariano(
    y_true: Any, pred_a: Any, pred_b: Any, power: int = 1
) -> dict[str, Any]:
    """Is A's error genuinely smaller than B's, or is it sampling noise?

    The standard test for comparing forecast accuracy. Its statistic is the mean
    loss differential over its standard error, using a Newey-West variance to
    account for the serial correlation that half-hourly forecast errors always
    have — ignoring that correlation would overstate significance badly.

    Negative statistic => A has lower loss. |stat| > ~1.96 => significant at 5%.
    """
    t, a = _aligned(y_true, pred_a)
    _, b = _aligned(y_true, pred_b)
    n = min(len(a), len(b))
    t, a, b = t[:n], a[:n], b[:n]

    loss = np.abs(t - a) ** power - np.abs(t - b) ** power
    mean_loss = float(np.mean(loss))

    # Newey-West with a rule-of-thumb bandwidth.
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    variance = float(np.var(loss, ddof=1))
    for k in range(1, max(lag, 1) + 1):
        cov = float(np.cov(loss[k:], loss[:-k])[0, 1])
        variance += 2 * (1 - k / (lag + 1)) * cov

    stderr = np.sqrt(max(variance, 1e-12) / n)
    statistic = mean_loss / stderr

    # Normal approximation; n is in the tens of thousands here.
    from math import erfc, sqrt

    p_value = erfc(abs(statistic) / sqrt(2))

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_loss_differential": mean_loss,
        "n": int(n),
        "favours": "a" if statistic < 0 else "b",
    }


def summary(y_true: Any, y_pred: Any) -> dict[str, float]:
    t, p = _aligned(y_true, y_pred)
    return {
        "n": int(len(t)),
        "mae": mae(t, p),
        "rmse": rmse(t, p),
        "smape": smape(t, p),
        "bias": float(np.mean(p - t)) if len(t) else float("nan"),
    }


def by_period_of_day(
    frame: pd.DataFrame, truth: str, prediction: str
) -> pd.DataFrame:
    """MAE per settlement period — where in the day the model struggles."""
    rows = [
        {
            "settlement_period": int(period),  # type: ignore[arg-type]
            "mae": mae(group[truth], group[prediction]),
            "n": int(len(group)),
        }
        for period, group in frame.groupby("settlement_period", sort=True)
    ]
    return pd.DataFrame(rows)


def bootstrap_mae_ci(
    y_true: Any,
    y_pred: Any,
    n_boot: int = 1000,
    block: int = 48,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Block bootstrap CI for MAE.

    Blocks of one day, because half-hourly forecast errors are strongly
    autocorrelated — an i.i.d. bootstrap would treat 48 correlated errors as 48
    independent observations and report an interval several times too narrow.
    """
    t, p = _aligned(y_true, y_pred)
    errors = np.abs(t - p)
    n_blocks = len(errors) // block
    if n_blocks < 2:
        return float("nan"), float("nan")

    blocks = errors[: n_blocks * block].reshape(n_blocks, block)
    rng = np.random.default_rng(seed)
    draws = [
        float(blocks[rng.integers(0, n_blocks, n_blocks)].mean()) for _ in range(n_boot)
    ]
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
