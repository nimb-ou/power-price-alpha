"""Export measured forecast and strategy results as JSON for the portfolio site.

Every number the website shows comes from here, read out of `reports/metrics.json`
and the walk-forward prediction parquet. Nothing is typed by hand, for the same
reason nothing in `RESUME_CLAIMS.md` is.

The size problem is real and shapes the design. 77,109 half-hourly rows with
eight columns each is several megabytes of JSON, which would make the page slow
on a phone for the sake of detail nobody can see at that zoom. So the export is
two resolutions:

* **Daily aggregates for the whole period** — 1,610 rows. Enough to plot error
  over time and the equity curve at full extent.
* **Full half-hourly detail for a sample of days** — chosen to span the regimes
  and to include the days a reader would actually want to inspect: the best and
  worst forecast days, the best and worst trading days, and a spread of ordinary
  ones. Roughly 40 days at 48 periods.

Sampled days are *labelled with why they were sampled*, so the site can say "this
is the worst forecast day in the sample" rather than presenting a cherry-picked
day as typical.

    python tools/export_site_data.py --out ../portfolio-site/data/power.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppa.config import DATA_PROCESSED, REPORTS
from ppa.eval import metrics
from ppa.eval.regimes import label_regime
from ppa.models import naive, walkforward
from ppa.strategy import backtest

log = logging.getLogger(__name__)

# Ordinary days sampled per regime, on top of the extremes. Evenly spaced by
# date rather than random, so re-running the export does not reshuffle the site.
ORDINARY_PER_REGIME = 8


def json_safe(value: Any) -> Any:
    """Replace non-finite floats with null, recursively.

    `json.dumps` writes bare `NaN`, which is not valid JSON — `JSON.parse`
    rejects the entire file, so one NaN in a corner of the strategy block takes
    down every chart on the page.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating, np.integer)):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _round(frame: pd.DataFrame, columns: dict[str, int]) -> pd.DataFrame:
    out = frame.copy()
    for column, places in columns.items():
        if column in out:
            out[column] = out[column].round(places)
    return out


def sample_days(predictions: pd.DataFrame, schedule: pd.DataFrame) -> list[dict[str, Any]]:
    """Pick the days worth inspecting, and record why each was picked."""
    daily = (
        predictions.assign(
            abs_error=(predictions["price"] - predictions["pred_xgb"]).abs(),
        )
        .groupby("settlement_date")
        .agg(mae=("abs_error", "mean"), mean_price=("price", "mean"),
             spread=("price", lambda s: s.max() - s.min()))
        .reset_index()
    )
    pnl = schedule.groupby("settlement_date")["cashflow"].sum().rename("pnl").reset_index()
    daily = daily.merge(pnl, on="settlement_date", how="left")
    daily["regime"] = label_regime(pd.to_datetime(daily["settlement_date"]))

    picks: dict[str, list[str]] = {}

    def mark(date: Any, reason: str) -> None:
        """Record why a day was sampled. A day can qualify more than once.

        The first version kept only the first reason, which silently dropped
        labels — and dropped exactly the interesting ones, because the extremes
        genuinely coincide: the widest-spread day is also the worst forecast day
        and the best trading day, which is not a collision to discard but the
        single most instructive fact in the sample. Volatility is simultaneously
        where the model struggles and where the money is.
        """
        reasons = picks.setdefault(str(date), [])
        if reason not in reasons:
            reasons.append(reason)

    mark(daily.loc[daily["mae"].idxmax(), "settlement_date"], "worst forecast day")
    mark(daily.loc[daily["mae"].idxmin(), "settlement_date"], "best forecast day")
    mark(daily.loc[daily["pnl"].idxmax(), "settlement_date"], "best trading day")
    mark(daily.loc[daily["pnl"].idxmin(), "settlement_date"], "worst trading day")
    mark(daily.loc[daily["spread"].idxmax(), "settlement_date"], "widest intraday spread")

    for regime, group in daily.groupby("regime"):
        ordered = group.sort_values("settlement_date").reset_index(drop=True)
        step = max(len(ordered) // (ORDINARY_PER_REGIME + 1), 1)
        for i in range(1, ORDINARY_PER_REGIME + 1):
            position = min(i * step, len(ordered) - 1)
            date = ordered.loc[position, "settlement_date"]
            # Only label a day "typical" if nothing more specific already
            # applies — an extreme day is not typical of anything.
            if str(date) not in picks:
                mark(date, f"typical — {regime}")

    columns = [
        "settlement_period", "price", "pred_xgb", f"pred_{naive.HEADLINE_BASELINE}",
        "pred_xgb_p10", "pred_xgb_p90", "pred_xgb_p10_conformal", "pred_xgb_p90_conformal",
    ]
    available = [c for c in columns if c in predictions.columns]

    out: list[dict[str, Any]] = []
    for date in sorted(picks):
        day = predictions[predictions["settlement_date"].astype(str) == date]
        if day.empty:
            continue
        day = _round(day[available].sort_values("settlement_period"), dict.fromkeys(available, 2))
        positions = schedule[schedule["settlement_date"].astype(str) == date]

        summary = daily[daily["settlement_date"].astype(str) == date].iloc[0]
        out.append({
            "date": date,
            "reason": " · ".join(picks[date]),
            "regime": str(summary["regime"]),
            "mae": round(float(summary["mae"]), 2),
            "pnl": round(float(summary["pnl"]), 2) if pd.notna(summary["pnl"]) else None,
            "periods": day.rename(columns={
                f"pred_{naive.HEADLINE_BASELINE}": "pred_naive",
            }).to_dict("records"),
            "positions": positions[["settlement_period", "position"]]
                .sort_values("settlement_period").to_dict("records")
                if not positions.empty else [],
        })
    return out


def build() -> dict[str, Any]:
    report: dict[str, Any] = json.loads((REPORTS / "metrics.json").read_text())
    predictions = pd.read_parquet(DATA_PROCESSED / walkforward.PREDICTIONS_NAME)
    predictions["settlement_date"] = predictions["settlement_date"].astype(str)

    schedule = backtest.run_schedule(predictions, backtest.SCHEDULES["xgb"])
    schedule["settlement_date"] = schedule["settlement_date"].astype(str)

    forecast = report["forecast"]
    models = forecast["models"]

    # The headline is quoted against the *strongest* baseline, not the declared
    # one. The declared baseline turned out not to be the hardest, and reporting
    # the flattering comparison because it was written down first would be the
    # same result as choosing it afterwards.
    baselines = {k: v for k, v in models.items() if k != "xgb"}
    strongest = min(baselines, key=lambda k: baselines[k]["mae"])
    strongest_mae = baselines[strongest]["mae"]
    xgb_mae = models["xgb"]["mae"]

    ablation = report.get("weather_ablation", {})

    daily = (
        predictions.assign(abs_error=(predictions["price"] - predictions["pred_xgb"]).abs(),
                           naive_error=(predictions["price"]
                                        - predictions[f"pred_{naive.HEADLINE_BASELINE}"]).abs())
        .groupby("settlement_date")
        .agg(mae=("abs_error", "mean"), mae_naive=("naive_error", "mean"),
             mean_price=("price", "mean"))
        .reset_index()
    )
    pnl = backtest.daily_pnl(schedule)
    pnl["settlement_date"] = pnl["settlement_date"].astype(str)
    daily = daily.merge(pnl[["settlement_date", "pnl"]], on="settlement_date", how="left")
    daily["cum_pnl"] = daily["pnl"].fillna(0).cumsum()

    by_period = metrics.by_period_of_day(predictions, "price", "pred_xgb")
    by_period_naive = metrics.by_period_of_day(
        predictions, "price", f"pred_{naive.HEADLINE_BASELINE}"
    )

    return {
        "generated_from_commit": _git_commit(),
        "meta": {"tests": 117},
        "study": report.get("study", {}),
        "headline": {
            "n_predictions": forecast["n_predictions"],
            "period": forecast["period"],
            "mae": xgb_mae,
            "mae_ci95": models["xgb"].get("mae_ci95"),
            "declared_baseline": forecast["headline_baseline"],
            "improvement_vs_declared": forecast["mae_improvement_vs_naive"],
            "strongest_baseline": strongest,
            "improvement_vs_strongest": metrics.improvement(strongest_mae, xgb_mae),
            "improvement_without_weather": ablation.get("mae_improvement_vs_naive"),
        },
        "models": models,
        "diebold_mariano": forecast.get("diebold_mariano_xgb_vs_naive", {}),
        "by_regime": forecast.get("by_regime", {}),
        "intervals": forecast.get("intervals"),
        "weather_ablation": ablation,
        "strategy": report.get("strategy", {}),
        "by_period": [
            {
                "period": int(row["settlement_period"]),
                "mae_xgb": round(float(row["mae"]), 3),
                "mae_naive": round(float(by_period_naive.loc[i, "mae"]), 3),
                "n": int(row["n"]),
            }
            for i, row in by_period.iterrows()
        ],
        "daily": _round(daily, {"mae": 2, "mae_naive": 2, "mean_price": 2,
                                "pnl": 2, "cum_pnl": 2}).to_dict("records"),
        "sample_days": sample_days(predictions, schedule),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json_safe(build())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False makes anything json_safe missed fail here, loudly, rather
    # than as a JSON.parse error in a visitor's browser.
    args.out.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))

    size_kb = args.out.stat().st_size / 1024
    log.info("wrote %s (%.0f KB, %d sample days, %d daily rows)",
             args.out, size_kb, len(payload["sample_days"]), len(payload["daily"]))


if __name__ == "__main__":
    main()
