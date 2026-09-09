"""Edge case tests for power market pricing and battery arbitrage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppa.strategy import backtest, risk, signals
from ppa.strategy.signals import BatteryConfig
from ppa.data.calendar import periods_in_day


def make_test_day(prices: list[float], forecast: list[float] | None = None, date: str = "2024-03-31") -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame(
        {
            "start_time": pd.date_range(f"{date} 00:00", periods=n, freq="30min", tz="UTC"),
            "settlement_date": date,
            "settlement_period": range(1, n + 1),
            "price": prices,
            "pred_xgb": forecast if forecast is not None else prices,
            "pred_same_period_last_week": prices,
        }
    )


CONFIG = BatteryConfig(periods_per_leg=2, round_trip_efficiency=0.90, min_forecast_spread=10.0)


def test_extreme_negative_spot_prices_arbitrage() -> None:
    """When prices go negative, charging earns revenue rather than paying cost."""
    # Periods 1 & 2 have negative prices (-£60 and -£40)
    prices = [-60.0, -40.0, 50.0, 120.0, 30.0, 30.0]
    frame = backtest.run_schedule(make_test_day(prices), "pred_xgb", 0.0, 0.0, CONFIG)

    # Position in negative periods must be charging (position < 0)
    assert frame.loc[0, "position"] < 0
    assert frame.loc[1, "position"] < 0
    # Discharging in highest periods (50, 120)
    assert frame.loc[3, "position"] > 0
    assert frame["cashflow"].sum() > 0


def test_zero_volatility_flat_regime_no_trades() -> None:
    """If forecast spread does not clear min_forecast_spread hurdle, zero trades are executed."""
    # Flat £50 across all periods
    prices = [50.0] * 48
    frame = backtest.run_schedule(make_test_day(prices), "pred_xgb", 0.0, 0.0, CONFIG)

    assert (frame["position"] == 0).all()
    assert frame["cashflow"].sum() == pytest.approx(0.0)


def test_clock_change_march_short_day_boundary() -> None:
    """March clock change day has exactly 46 settlement periods."""
    n_periods = periods_in_day("2024-03-31")
    assert n_periods == 46

    # Test that battery scheduler handles 46-period day cleanly
    prices = [40.0 + 10.0 * np.sin(i / 5.0) for i in range(46)]
    frame = signals.build(make_test_day(prices, date="2024-03-31"), config=CONFIG)
    assert len(frame) == 46
    assert frame["position"].sum() == pytest.approx(0.0)


def test_clock_change_october_long_day_boundary() -> None:
    """October clock change day has exactly 50 settlement periods."""
    n_periods = periods_in_day("2024-10-27")
    assert n_periods == 50

    prices = [40.0 + 10.0 * np.sin(i / 5.0) for i in range(50)]
    frame = signals.build(make_test_day(prices, date="2024-10-27"), config=CONFIG)
    assert len(frame) == 50
    assert frame["position"].sum() == pytest.approx(0.0)
