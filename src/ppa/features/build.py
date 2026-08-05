"""Day-ahead features, built against an explicit information set.

**The rule.** To forecast every settlement period of day T, the decision is made
at day-ahead gate closure: 11:00 local on **T-1**. A feature may only use data
with a timestamp strictly before that instant.

That rule is stricter than it first looks, and it rules out the feature almost
everyone reaches for first:

> `price_lag_1d` — "the price at this same settlement period yesterday"

For a target period at 14:30 on day T, the same period on T-1 is 14:30 on T-1 —
which is **three and a half hours after gate closure**. It has not happened yet
when the forecast is made. Using it inflates accuracy substantially and is
invisible in any backtest that does not model the information set explicitly.

What *is* knowable at 11:00 on T-1:

- everything up to and including settlement period 22 of T-1 (SP22 ends 11:00);
- every complete day up to and including T-2;
- the calendar, forever;
- a weather *forecast* for T.

So the price features here use T-2 and earlier at period granularity, plus
morning-of-T-1 aggregates. `tests/test_no_leakage.py` asserts every feature's
source timestamp is strictly before its decision time.

    python -m ppa.features.build
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from ppa.config import DATA_PROCESSED, DAY_AHEAD_GATE_CLOSURE_LOCAL, STUDY
from ppa.data import calendar as cal
from ppa.data.assemble import load as load_panel

log = logging.getLogger(__name__)

FEATURES_NAME = "features.parquet"
TARGET = "price"

# SP22 covers 10:30-11:00 local, so periods 1..22 of T-1 have completed by gate
# closure and periods 23+ have not.
LAST_PERIOD_BEFORE_GATE = 22

# Same-period lags, in days. 1 is deliberately absent — see the module docstring.
PERIOD_LAGS_DAYS = [2, 3, 7, 14]


def decision_time(target_date: pd.Series) -> pd.Series:
    """Gate closure for each target date: 11:00 local on the previous day."""
    hour, minute = (int(x) for x in DAY_AHEAD_GATE_CLOSURE_LOCAL.split(":"))
    previous = pd.to_datetime(target_date) - pd.Timedelta(days=1)
    local = previous + pd.Timedelta(hours=hour, minutes=minute)
    return (
        local.dt.tz_localize(cal.TZ, nonexistent="shift_forward", ambiguous="NaT")
        .dt.tz_convert("UTC")
    )


def _period_lags(panel: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """Same-settlement-period values from N days earlier.

    Shifting by `48 * days` rows would be wrong: rows are not 48 per day on
    clock-change days, so a positional shift drifts by an hour twice a year and
    never recovers. Merging on (date - N days, period) is correct by
    construction.
    """
    out = pd.DataFrame(index=panel.index)
    base = panel[["settlement_date", "settlement_period", column]].copy()

    for days in PERIOD_LAGS_DAYS:
        shifted = base.copy()
        shifted["settlement_date"] = (
            pd.to_datetime(shifted["settlement_date"]) + pd.Timedelta(days=days)
        ).dt.date.astype(str)
        shifted = shifted.rename(columns={column: f"{prefix}_lag_{days}d"})

        merged = panel[["settlement_date", "settlement_period"]].merge(
            shifted, on=["settlement_date", "settlement_period"], how="left"
        )
        out[f"{prefix}_lag_{days}d"] = merged[f"{prefix}_lag_{days}d"].to_numpy()

    return out


def _morning_aggregates(panel: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """Aggregates of T-1's periods 1..22 — the freshest legitimate information.

    This is the feature that makes the honest setup competitive. The market's
    state on the morning of T-1 is genuinely known at gate closure and carries
    most of the short-horizon signal that `lag_1d` would have provided.
    """
    morning = panel[panel["settlement_period"] <= LAST_PERIOD_BEFORE_GATE]
    daily = morning.groupby("settlement_date")[column].agg(["mean", "max", "min", "std"])
    daily.columns = [f"{prefix}_am_{stat}" for stat in daily.columns]
    daily = daily.reset_index()

    # Attach T-1's morning to target date T.
    daily["settlement_date"] = (
        pd.to_datetime(daily["settlement_date"]) + pd.Timedelta(days=1)
    ).dt.date.astype(str)

    merged = panel[["settlement_date"]].merge(daily, on="settlement_date", how="left")
    return merged.drop(columns="settlement_date").set_index(panel.index)


def _rolling_daily(panel: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """Rolling means of complete days, ending at T-2.

    `shift(2)` before rolling is what enforces the information set: the window
    ends two days before the target, so T-1's incomplete day never enters.
    """
    daily = panel.groupby("settlement_date")[column].mean().reset_index()
    daily = daily.sort_values("settlement_date")

    for window in (7, 30):
        daily[f"{prefix}_roll_{window}d"] = (
            daily[column].shift(2).rolling(window, min_periods=max(2, window // 2)).mean()
        )

    daily = daily.drop(columns=column)
    merged = panel[["settlement_date"]].merge(daily, on="settlement_date", how="left")
    return merged.drop(columns="settlement_date").set_index(panel.index)


def _fourier(panel: pd.DataFrame) -> pd.DataFrame:
    """Smooth periodic terms for intraday, weekly and annual cycles.

    Trees can split on `settlement_period` directly, but they cannot express
    that period 48 is adjacent to period 1. Fourier terms make that wraparound
    explicit and let the model share information across neighbouring periods
    instead of learning 48 independent step functions.
    """
    out = pd.DataFrame(index=panel.index)
    period = panel["settlement_period"].to_numpy()
    doy = panel["day_of_year"].to_numpy()
    dow = panel["day_of_week"].to_numpy()

    for k in (1, 2, 3):
        out[f"sin_day_{k}"] = np.sin(2 * np.pi * k * period / 48)
        out[f"cos_day_{k}"] = np.cos(2 * np.pi * k * period / 48)
    for k in (1, 2):
        out[f"sin_year_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        out[f"cos_year_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    out["sin_week"] = np.sin(2 * np.pi * dow / 7)
    out["cos_week"] = np.cos(2 * np.pi * dow / 7)
    return out


def build(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = panel if panel is not None else load_panel()
    panel = panel.sort_values("start_time").reset_index(drop=True)

    parts = [
        panel[
            [
                "settlement_date",
                "settlement_period",
                "start_time",
                "day_of_week",
                "day_of_year",
                "month",
                "year",
                "is_weekend",
                TARGET,
            ]
        ],
        _period_lags(panel, "price", "price"),
        _period_lags(panel, "residual_load", "residual"),
        _morning_aggregates(panel, "price", "price"),
        _morning_aggregates(panel, "residual_load", "residual"),
        _rolling_daily(panel, "price", "price"),
        _rolling_daily(panel, "residual_load", "residual"),
        _fourier(panel),
    ]

    # Weather for the target day, degraded to daily aggregates. The archive API
    # returns outturn, not the forecast that would have been available at gate
    # closure; a daily summary is much closer to a real forecast's information
    # content than the half-hourly outturn would be. Stated in the README and
    # quantified by the --perfect-weather run.
    weather_columns = [
        "temperature_mean",
        "heating_degrees",
        "cooling_degrees",
        "wind_speed_mean",
        "wind_speed_std",
        "radiation_mean",
        "cloud_cover_mean",
    ]
    available = [c for c in weather_columns if c in panel.columns]
    daily_weather = panel.groupby("settlement_date")[available].mean().reset_index()
    daily_weather.columns = ["settlement_date"] + [f"{c}_fcst" for c in available]
    weather_part = (
        panel[["settlement_date"]]
        .merge(daily_weather, on="settlement_date", how="left")
        .drop(columns="settlement_date")
        .set_index(panel.index)
    )
    parts.append(weather_part)

    features = pd.concat(parts, axis=1)
    features["decision_time"] = decision_time(features["settlement_date"])

    before = len(features)
    features = features.dropna(subset=[TARGET]).reset_index(drop=True)
    log.info("dropped %d rows with no target price", before - len(features))

    return features


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Model inputs: everything except the target, the keys and audit columns.

    Datetime columns are excluded by dtype as well as by name. A helper column
    added downstream (the walk-forward harness adds `date`) would otherwise be
    handed to the model as a feature — which is both a crash and, if it happened
    to be numeric, a way to memorise regimes by calendar position rather than
    learn from market state.
    """
    excluded = {TARGET, "settlement_date", "start_time", "decision_time", "year", "date"}
    return [
        c
        for c in features.columns
        if c not in excluded and not pd.api.types.is_datetime64_any_dtype(features[c])
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=STUDY.start)
    parser.add_argument("--end", default=STUDY.end)
    parser.parse_args()

    features = build()
    path = DATA_PROCESSED / FEATURES_NAME
    features.to_parquet(path, index=False)

    columns = feature_columns(features)
    complete = features.dropna(subset=columns)

    print(f"wrote {path}  ({len(features):,} rows x {len(columns)} features)")
    print(f"rows with every feature present: {len(complete):,}")
    print(f"target: mean {features[TARGET].mean():.2f}  std {features[TARGET].std():.2f}")
    print(f"features: {columns}")


if __name__ == "__main__":
    main()
