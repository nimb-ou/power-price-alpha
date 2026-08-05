"""The storage-arbitrage backtest, with frictions.

Per settlement period, for a schedule fixed at gate closure:

    charging  (position < 0):  cash out = price x MWh x (1 + friction)
    discharging (position > 0): cash in  = price x MWh x efficiency - friction

P&L is aggregated to **daily** figures, because the battery completes one cycle
per day and a half-hourly P&L series would be autocorrelated by construction —
the four charge periods of a day are one decision, not four.

Three schedules are always run side by side:

- **xgb** — the model's forecast,
- **naive** — the seasonal-naive forecast, the same battery on a worse signal,
- **oracle** — perfect foresight, the ceiling on what any forecast could earn.

The model's value is the gap between the first two; the oracle says how much of
the available money the forecast actually captures. Reporting the model alone
would tell you nothing about whether the forecast mattered.

    python -m ppa.strategy.backtest
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import pandas as pd

from ppa.config import DATA_PROCESSED, REPORTS, STUDY
from ppa.strategy import risk, signals
from ppa.strategy.signals import PERIOD_HOURS, BatteryConfig

log = logging.getLogger(__name__)

BACKTEST_NAME = "backtest.parquet"
PREDICTIONS_NAME = "walkforward_predictions.parquet"

SCHEDULES = {
    "xgb": "pred_xgb",
    "naive": "pred_same_period_last_week",
    "oracle": "price",  # perfect foresight — the ceiling, not a strategy
}


def run_schedule(
    predictions: pd.DataFrame,
    forecast_col: str,
    transaction_cost: float = STUDY.transaction_cost,
    slippage: float = STUDY.slippage,
    config: BatteryConfig = signals.DEFAULT,
) -> pd.DataFrame:
    """Period-level cashflows for one schedule."""
    frame = signals.build(predictions, forecast_col=forecast_col, config=config)
    frame = frame.sort_values("start_time").reset_index(drop=True)

    friction = transaction_cost + slippage
    energy = frame["position"].abs() * PERIOD_HOURS  # MWh moved this period

    charging = frame["position"] < 0
    discharging = frame["position"] > 0

    cash = pd.Series(0.0, index=frame.index)
    # Buying: pay the price plus friction on every MWh taken in.
    cash[charging] = -energy[charging] * (frame.loc[charging, "price"] + friction)
    # Selling: receive the price less friction, on the energy that survives the
    # round trip. Applying efficiency on the way out is the standard convention.
    cash[discharging] = (
        energy[discharging]
        * config.round_trip_efficiency
        * (frame.loc[discharging, "price"] - friction)
    )

    frame["energy_mwh"] = energy
    frame["cashflow"] = cash
    frame["friction_cost"] = energy * friction
    return frame


def daily_pnl(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per day: the battery's realised profit."""
    daily = (
        frame.groupby("settlement_date")
        .agg(
            pnl=("cashflow", "sum"),
            friction=("friction_cost", "sum"),
            mwh=("energy_mwh", "sum"),
            start_time=("start_time", "min"),
        )
        .reset_index()
    )
    daily["cum_pnl"] = daily["pnl"].cumsum()
    return daily


def evaluate(
    predictions: pd.DataFrame,
    transaction_cost: float = STUDY.transaction_cost,
    slippage: float = STUDY.slippage,
    config: BatteryConfig = signals.DEFAULT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run all three schedules and summarise."""
    results: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    dailies: dict[str, pd.DataFrame] = {}

    for name, column in SCHEDULES.items():
        frame = run_schedule(predictions, column, transaction_cost, slippage, config)
        daily = daily_pnl(frame)
        frames[name] = frame
        dailies[name] = daily

        block = risk.summary(daily["pnl"])
        block["total_friction"] = float(frame["friction_cost"].sum())
        block["mwh_cycled"] = float(frame["energy_mwh"].sum())
        block["signal"] = signals.summarise(frame)
        results[name] = block

    xgb_daily = dailies["xgb"]["pnl"]
    naive_total = float(dailies["naive"]["pnl"].sum())
    oracle_total = float(dailies["oracle"]["pnl"].sum())
    xgb_total = float(xgb_daily.sum())

    report: dict[str, Any] = {
        "battery": {
            "power_mw": config.power_mw,
            "energy_mwh": config.power_mw * config.periods_per_leg * PERIOD_HOURS,
            "round_trip_efficiency": config.round_trip_efficiency,
            "min_forecast_spread": config.min_forecast_spread,
        },
        "frictions": {
            "transaction_cost": transaction_cost,
            "slippage": slippage,
            "total_per_mwh": transaction_cost + slippage,
        },
        "schedules": results,
        "uplift_vs_naive": xgb_total - naive_total,
        "uplift_vs_naive_pct": (
            (xgb_total - naive_total) / abs(naive_total) if naive_total else float("nan")
        ),
        "share_of_oracle": xgb_total / oracle_total if oracle_total else float("nan"),
        "naive_share_of_oracle": naive_total / oracle_total if oracle_total else float("nan"),
    }

    net = results["xgb"]
    net["sharpe_ci95"] = risk.bootstrap_sharpe_ci(xgb_daily)
    net["by_regime"] = risk.by_regime(dailies["xgb"], pnl_col="pnl")
    net["by_year"] = risk.by_year(dailies["xgb"], pnl_col="pnl")

    # A frictionless run, so the reader can see what the costs consumed.
    frictionless = daily_pnl(run_schedule(predictions, "pred_xgb", 0.0, 0.0, config))
    report["frictionless_reference"] = risk.summary(frictionless["pnl"])

    return report, dailies["xgb"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction-cost", type=float, default=STUDY.transaction_cost)
    parser.add_argument("--slippage", type=float, default=STUDY.slippage)
    parser.add_argument("--efficiency", type=float, default=signals.DEFAULT.round_trip_efficiency)
    args = parser.parse_args()

    predictions = pd.read_parquet(DATA_PROCESSED / PREDICTIONS_NAME)
    config = BatteryConfig(round_trip_efficiency=args.efficiency)

    report, daily = evaluate(predictions, args.transaction_cost, args.slippage, config)
    daily.to_parquet(DATA_PROCESSED / BACKTEST_NAME, index=False)
    (REPORTS / "strategy_metrics.json").write_text(json.dumps(report, indent=2, default=str))

    xgb = report["schedules"]["xgb"]
    print()
    print("=" * 72)
    print(f"  1 MW / {report['battery']['energy_mwh']:.0f} MWh battery, "
          f"{config.round_trip_efficiency:.0%} round trip, "
          f"{report['frictions']['total_per_mwh']:.2f} GBP/MWh friction")
    print(f"  days traded      {xgb['signal']['days_traded']:,} of {xgb['signal']['total_days']:,}")
    print()
    print(f"  {'schedule':<10}{'total P&L':>14}{'GBP/day':>10}{'Sharpe':>9}{'hit':>8}")
    for name, block in report["schedules"].items():
        print(f"  {name:<10}{block['total']:>14,.0f}{block['mean_per_period']:>10.1f}"
              f"{block['sharpe']:>9.2f}{block['hit_rate']:>8.1%}")
    print()
    print(f"  uplift vs naive  {report['uplift_vs_naive']:,.0f} GBP "
          f"({report['uplift_vs_naive_pct']:+.1%})")
    print(f"  share of oracle  {report['share_of_oracle']:.1%}  "
          f"(naive captures {report['naive_share_of_oracle']:.1%})")
    print(f"  Sharpe 95% CI    {[round(v, 2) for v in xgb['sharpe_ci95']]}")
    print(f"  max drawdown     {xgb['max_drawdown']:,.0f} GBP")
    print()
    print("  by regime:")
    for name, block in xgb["by_regime"].items():
        print(f"    {name:<22} Sharpe {block['sharpe']:>6.2f}  "
              f"GBP/day {block['mean_per_period']:>7.1f}  n {block['n']:>5,}")
    print("=" * 72)


if __name__ == "__main__":
    main()
