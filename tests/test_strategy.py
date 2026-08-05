"""Storage-arbitrage backtest mechanics.

Tested against hand-built price series where the right answer is known by
inspection. "The backtest agrees with itself" proves nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppa.strategy import backtest, risk, signals
from ppa.strategy.signals import PERIOD_HOURS, BatteryConfig


def make_day(prices: list[float], forecast: list[float] | None = None, date: str = "2024-01-01") -> pd.DataFrame:
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


SMALL = BatteryConfig(periods_per_leg=2, round_trip_efficiency=1.0, min_forecast_spread=0.0)


# --- scheduling --------------------------------------------------------------


def test_charges_at_the_cheapest_and_discharges_at_the_dearest() -> None:
    prices = [10.0, 90.0, 20.0, 80.0, 50.0, 50.0]
    frame = signals.build(make_day(prices), config=SMALL)

    # Two cheapest are periods 1 (10) and 3 (20); two dearest are 2 (90), 4 (80).
    assert frame.loc[0, "position"] < 0
    assert frame.loc[2, "position"] < 0
    assert frame.loc[1, "position"] > 0
    assert frame.loc[3, "position"] > 0
    assert frame.loc[4, "position"] == 0


def test_energy_balances_within_the_day() -> None:
    """The battery must start and end each day empty."""
    frame = signals.build(make_day([10.0, 90.0, 20.0, 80.0, 50.0, 50.0]), config=SMALL)
    assert frame["position"].sum() == pytest.approx(0.0)


def test_schedule_follows_the_forecast_not_the_outturn() -> None:
    """The whole point: the decision is made from the forecast at gate closure."""
    prices = [10.0, 90.0, 20.0, 80.0]
    misleading = [90.0, 10.0, 80.0, 20.0]  # forecast is exactly inverted
    frame = signals.build(make_day(prices, misleading), config=SMALL)

    # It charges where the FORECAST is low (periods 2 and 4), not where price is.
    assert frame.loc[1, "position"] < 0
    assert frame.loc[3, "position"] < 0


def test_flat_day_is_skipped() -> None:
    config = BatteryConfig(periods_per_leg=2, min_forecast_spread=5.0)
    frame = signals.build(make_day([50.0, 51.0, 50.5, 50.2]), config=config)
    assert (frame["position"] == 0).all()


def test_a_day_too_short_to_cycle_is_skipped() -> None:
    frame = signals.build(make_day([10.0, 90.0]), config=BatteryConfig(periods_per_leg=4))
    assert (frame["position"] == 0).all()


# --- P&L ---------------------------------------------------------------------


def test_pnl_is_hand_computable_without_frictions() -> None:
    """Buy 2 x 0.5 MWh at 10 and 20; sell 2 x 0.5 MWh at 90 and 80."""
    prices = [10.0, 90.0, 20.0, 80.0]
    frame = backtest.run_schedule(
        make_day(prices), "pred_xgb", transaction_cost=0.0, slippage=0.0, config=SMALL
    )
    expected = (90 + 80) * PERIOD_HOURS - (10 + 20) * PERIOD_HOURS
    assert frame["cashflow"].sum() == pytest.approx(expected)


def test_efficiency_reduces_revenue_not_cost() -> None:
    prices = [10.0, 90.0, 20.0, 80.0]
    lossy = BatteryConfig(periods_per_leg=2, round_trip_efficiency=0.5, min_forecast_spread=0.0)
    frame = backtest.run_schedule(make_day(prices), "pred_xgb", 0.0, 0.0, lossy)

    charge_cost = -frame.loc[frame["position"] < 0, "cashflow"].sum()
    revenue = frame.loc[frame["position"] > 0, "cashflow"].sum()

    assert charge_cost == pytest.approx((10 + 20) * PERIOD_HOURS)
    assert revenue == pytest.approx((90 + 80) * PERIOD_HOURS * 0.5)


def test_frictions_are_charged_on_both_legs() -> None:
    prices = [10.0, 90.0, 20.0, 80.0]
    free = backtest.run_schedule(make_day(prices), "pred_xgb", 0.0, 0.0, SMALL)
    charged = backtest.run_schedule(make_day(prices), "pred_xgb", 1.0, 0.0, SMALL)

    total_mwh = 4 * PERIOD_HOURS  # 2 charge + 2 discharge periods
    assert charged["friction_cost"].sum() == pytest.approx(total_mwh * 1.0)
    assert charged["cashflow"].sum() < free["cashflow"].sum()


def test_a_bad_forecast_loses_money() -> None:
    """Inverted forecast: buy at the peak, sell at the trough."""
    prices = [10.0, 90.0, 20.0, 80.0]
    inverted = [90.0, 10.0, 80.0, 20.0]
    frame = backtest.run_schedule(make_day(prices, inverted), "pred_xgb", 0.0, 0.0, SMALL)
    assert frame["cashflow"].sum() < 0


def test_oracle_beats_any_forecast_on_the_same_day() -> None:
    prices = [10.0, 90.0, 20.0, 80.0, 45.0, 55.0]
    noisy = [12.0, 60.0, 55.0, 80.0, 44.0, 58.0]

    oracle = backtest.run_schedule(make_day(prices), "price", 0.0, 0.0, SMALL)
    model = backtest.run_schedule(make_day(prices, noisy), "pred_xgb", 0.0, 0.0, SMALL)
    assert oracle["cashflow"].sum() >= model["cashflow"].sum()


def test_daily_aggregation_gives_one_row_per_day() -> None:
    days = pd.concat(
        [make_day([10.0, 90.0, 20.0, 80.0], date=d) for d in ("2024-01-01", "2024-01-02")],
        ignore_index=True,
    )
    frame = backtest.run_schedule(days, "pred_xgb", 0.0, 0.0, SMALL)
    daily = backtest.daily_pnl(frame)

    assert len(daily) == 2
    assert daily["pnl"].sum() == pytest.approx(frame["cashflow"].sum())


# --- risk statistics ---------------------------------------------------------


def test_annualisation_is_daily_not_half_hourly() -> None:
    """sqrt(365.25) ~ 19.1. Treating the 48 periods as independent gave ~132."""
    assert pytest.approx(np.sqrt(365.25)) == risk.ANNUALISATION
    assert pytest.approx(19.1, abs=0.1) == risk.ANNUALISATION
    assert pytest.approx(np.sqrt(252), abs=0.5) != risk.ANNUALISATION


def test_sharpe_of_a_constant_series_is_undefined() -> None:
    assert np.isnan(risk.sharpe(pd.Series([1.0] * 100)))


def test_sharpe_scales_as_expected() -> None:
    rng = np.random.default_rng(0)
    pnl = pd.Series(rng.normal(0.1, 1.0, 20000))
    assert risk.sharpe(pnl) == pytest.approx(0.1 * risk.ANNUALISATION, rel=0.15)


def test_max_drawdown_on_a_hand_built_curve() -> None:
    # cumulative: 10, 5, 15, 5, 10 -> worst peak-to-trough is 15 -> 5 = -10
    assert risk.max_drawdown(pd.Series([10.0, -5.0, 10.0, -10.0, 5.0])) == pytest.approx(-10.0)


def test_max_drawdown_of_a_monotone_curve_is_zero() -> None:
    assert risk.max_drawdown(pd.Series([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_hit_rate_excludes_untraded_days() -> None:
    assert risk.hit_rate(pd.Series([1.0, -1.0, 0.0, 0.0, 1.0])) == pytest.approx(2 / 3)


def test_regime_labels_use_declared_dates() -> None:
    times = pd.Series(
        pd.to_datetime(["2020-06-01", "2022-01-01", "2024-06-01"]).tz_localize("UTC")
    )
    assert risk.label_regime(times).tolist() == [
        "calm_pre_crisis",
        "volatile_gas_crisis",
        "post_crisis",
    ]


def test_bootstrap_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(1)
    pnl = pd.Series(rng.normal(0.05, 1.0, 5000))
    lo, hi = risk.bootstrap_sharpe_ci(pnl, n_boot=200)
    assert lo < risk.sharpe(pnl) < hi


def test_block_bootstrap_is_wider_than_iid_on_correlated_pnl() -> None:
    rng = np.random.default_rng(2)
    shocks = rng.normal(0, 1, 200)
    correlated = pd.Series(np.repeat(shocks, 7) + 0.05)  # one shock per week

    wide = risk.bootstrap_sharpe_ci(correlated, n_boot=200, block=7)
    narrow = risk.bootstrap_sharpe_ci(correlated, n_boot=200, block=1)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
