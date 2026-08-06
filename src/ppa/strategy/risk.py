"""Risk and performance statistics.

The annualisation factor is the detail people get wrong and nobody checks.

Our P&L series is **daily** — the battery completes one cycle per day, so a day
is one decision. There are 365.25 trading days a year (power trades every day,
unlike equities), giving a Sharpe scaling of sqrt(365.25) ~ 19.1.

Two wrong answers that are easy to reach for:

- **sqrt(252)** — the equity convention. Power markets have no weekends off.
- **sqrt(17532)** — treating the underlying half-hourly periods as independent
  observations. The four charge periods of a day are a single decision made at
  one gate closure; counting them as four independent bets inflates the Sharpe
  by roughly seven times. An earlier version of this file did exactly that and
  produced a Sharpe of 50, which is how the error was caught.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ppa.eval.regimes import label_regime

# Re-exported: regime labelling moved to `eval/regimes.py` when the forecast
# layer needed it too, but `risk.label_regime` is what the tests and notebooks
# already call.
__all__ = ["label_regime"]

DAYS_PER_YEAR = 365.25
ANNUALISATION = float(np.sqrt(DAYS_PER_YEAR))


def sharpe(pnl: pd.Series | np.ndarray) -> float:
    """Annualised Sharpe of a daily P&L series, zero risk-free rate.

    A risk-free rate makes no sense for a self-financing spread position with no
    capital base, so the numerator is the raw mean.
    """
    series = pd.Series(pnl).dropna()
    if len(series) < 2 or series.std(ddof=1) == 0:
        return float("nan")
    return float(series.mean() / series.std(ddof=1) * ANNUALISATION)


def max_drawdown(pnl: pd.Series | np.ndarray) -> float:
    """Largest peak-to-trough fall of the cumulative P&L, in P&L units.

    Additive, not percentage: there is no capital base to express a percentage
    against, and inventing one would be the same fabrication as compounding.
    """
    cumulative = pd.Series(pnl).fillna(0).cumsum()
    peak = cumulative.cummax()
    return float((cumulative - peak).min())


def hit_rate(pnl: pd.Series | np.ndarray) -> float:
    """Share of *traded* periods that made money.

    Periods with no position are excluded. Including them would inflate the
    number towards whatever fraction of the time the strategy sits flat.
    """
    series = pd.Series(pnl).dropna()
    traded = series[series != 0]
    return float((traded > 0).mean()) if len(traded) else float("nan")


def summary(pnl: pd.Series | np.ndarray) -> dict[str, Any]:
    series = pd.Series(pnl).dropna()
    return {
        "n": int(len(series)),
        "total": float(series.sum()),
        "mean_per_period": float(series.mean()) if len(series) else float("nan"),
        "std_per_period": float(series.std(ddof=1)) if len(series) > 1 else float("nan"),
        "sharpe": sharpe(series),
        "hit_rate": hit_rate(series),
        "max_drawdown": max_drawdown(series),
        "best_period": float(series.max()) if len(series) else float("nan"),
        "worst_period": float(series.min()) if len(series) else float("nan"),
    }


def bootstrap_sharpe_ci(
    pnl: pd.Series | np.ndarray,
    n_boot: int = 1000,
    block: int = 7,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[float]:
    """Block-bootstrap CI for the Sharpe ratio.

    One-week blocks. Daily battery P&L is far less autocorrelated than the
    half-hourly series was, but weather and price regimes persist for days, so
    an i.i.d. bootstrap would still understate the interval.
    """
    series = pd.Series(pnl).dropna().to_numpy()
    n_blocks = len(series) // block
    if n_blocks < 2:
        return [float("nan"), float("nan")]

    blocks = series[: n_blocks * block].reshape(n_blocks, block)
    rng = np.random.default_rng(seed)
    draws = [
        sharpe(blocks[rng.integers(0, n_blocks, n_blocks)].ravel()) for _ in range(n_boot)
    ]
    clean = [d for d in draws if np.isfinite(d)]
    if not clean:
        return [float("nan"), float("nan")]
    lo, hi = np.quantile(clean, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def by_regime(frame: pd.DataFrame, pnl_col: str = "pnl") -> dict[str, dict[str, Any]]:
    labelled = frame.copy()
    labelled["regime"] = label_regime(labelled["start_time"])
    return {
        str(name): summary(group[pnl_col])
        for name, group in labelled.groupby("regime", sort=False)
    }


def by_year(frame: pd.DataFrame, pnl_col: str = "pnl") -> dict[str, dict[str, Any]]:
    labelled = frame.copy()
    labelled["year"] = pd.to_datetime(labelled["start_time"]).dt.year
    return {
        str(year): summary(group[pnl_col])
        for year, group in labelled.groupby("year", sort=True)
    }
