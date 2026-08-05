"""Turning a price forecast into a position: day-ahead storage arbitrage.

**A design that was tried first and thrown away.** The obvious strategy is to
take a position on `forecast - reference` and settle against `realised -
reference`, where the reference is the seasonal-naive forecast. It backtests
beautifully — and it is meaningless, because *there is no instrument whose payoff
is "price minus last week's price at this time"*. It also produces an absurd
Sharpe (~50), which is the tell: a synthetic spread you cannot trade will happily
pay you a synthetic return.

**What this does instead.** A 1 MW / 2 h battery bidding into the day-ahead
auction. At gate closure on T-1 we have a forecast for all 48 periods of T. We
commit to:

- **charge** (buy) during the 4 cheapest forecast periods,
- **discharge** (sell) during the 4 most expensive forecast periods.

The schedule is fixed at gate closure and settles against the *realised* prices.
Every leg is a real trade at a real price, so the P&L is money.

This is the canonical commercial use of a day-ahead price forecast in power, and
it makes the forecast's value directly measurable: run the identical battery on
the naive forecast and on perfect foresight, and the model's edge is the gap
between them.

Physical constraints that make it honest:

- **Round-trip efficiency** (default 85%) — you get back less than you put in.
- **Energy balance** — charge and discharge periods are equal in number, so the
  battery starts and ends each day empty. No borrowing energy across days.
- **One cycle per day** — battery degradation makes more cycles uneconomic, and
  it stops the strategy from manufacturing trades out of noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatteryConfig:
    """A 1 MW / 2 MWh battery — a small grid-scale unit."""

    power_mw: float = 1.0
    # Half-hourly periods spent charging (and, equally, discharging). Four
    # periods x 0.5 h x 1 MW = 2 MWh, so this is a 2-hour battery.
    periods_per_leg: int = 4
    round_trip_efficiency: float = 0.85
    # Only trade a day if the forecast spread clears this, in GBP/MWh. Stops the
    # battery cycling on days where the forecast sees no real opportunity.
    min_forecast_spread: float = 5.0


DEFAULT = BatteryConfig()

PERIOD_HOURS = 0.5


def build(
    predictions: pd.DataFrame,
    forecast_col: str = "pred_xgb",
    config: BatteryConfig = DEFAULT,
) -> pd.DataFrame:
    """Assign each settlement period a schedule: charge, discharge, or idle.

    `position` is in MW: negative while charging (buying), positive while
    discharging (selling). The schedule for day T is decided using only the
    forecast, which was itself built from information available at gate closure
    on T-1.
    """
    out = predictions.copy()
    out["position"] = 0.0

    for _, day in out.groupby("settlement_date", sort=False):
        forecast = day[forecast_col]
        if forecast.isna().all():
            continue

        n = config.periods_per_leg
        if len(day) < 2 * n:
            continue

        ranked = forecast.rank(method="first")
        cheapest = ranked.nsmallest(n).index
        dearest = ranked.nlargest(n).index

        # Skip days where the forecast sees no worthwhile spread.
        spread = float(forecast.loc[dearest].mean() - forecast.loc[cheapest].mean())
        if not np.isfinite(spread) or spread < config.min_forecast_spread:
            continue

        out.loc[cheapest, "position"] = -config.power_mw
        out.loc[dearest, "position"] = config.power_mw

    return out


def summarise(signals: pd.DataFrame) -> dict[str, float]:
    position = signals["position"]
    active = position != 0
    traded_days = signals.loc[active, "settlement_date"].nunique()
    total_days = signals["settlement_date"].nunique()

    return {
        "n_periods": int(len(signals)),
        "n_active": int(active.sum()),
        "participation_rate": float(active.mean()),
        "days_traded": int(traded_days),
        "total_days": int(total_days),
        "day_participation_rate": float(traded_days / total_days) if total_days else 0.0,
        "mwh_charged": float(-position[position < 0].sum() * PERIOD_HOURS),
        "mwh_discharged": float(position[position > 0].sum() * PERIOD_HOURS),
    }
