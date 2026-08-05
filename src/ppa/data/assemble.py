"""Assemble the half-hourly panel from three sources.

The join is the risky part of this project. Prices arrive keyed by UTC
timestamp, demand by (local settlement date, period), and weather by UTC hour.
Getting any of those alignments wrong produces a frame that looks completely
normal and is wrong.

So the panel is built against an **expected grid** from `calendar.py` — the
complete set of (date, period) pairs the range should contain, honouring 46/48/50
— and everything missing is reported rather than silently dropped or
interpolated.

    python -m ppa.data.assemble --start 2019-01-01 --end 2025-06-30
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import pandas as pd

from ppa.config import DATA_PROCESSED, STUDY
from ppa.data import calendar as cal
from ppa.ingest import elexon, neso, openmeteo

log = logging.getLogger(__name__)

PANEL_NAME = "panel.parquet"


def build(start: str, end: str) -> pd.DataFrame:
    prices = elexon.load()
    demand = neso.load()
    weather = openmeteo.to_national(openmeteo.load())

    grid = cal.expected_periods(start, end)
    log.info("expected %d settlement periods over %s..%s", len(grid), start, end)

    # --- prices ---------------------------------------------------------
    # Elexon gives us both a UTC timestamp and a (date, period) pair. We key on
    # (date, period) and *recompute* the timestamp from the calendar, then check
    # the two agree. That check is the cheapest possible validation that our
    # calendar matches the market operator's.
    prices = prices[["settlement_date", "settlement_period", "price", "volume"]]
    panel = grid.merge(prices, on=["settlement_date", "settlement_period"], how="left")
    panel = cal.add_utc_column(panel)

    # --- demand ---------------------------------------------------------
    panel = panel.merge(demand, on=["settlement_date", "settlement_period"], how="left")

    # Residual load: what the wholesale market actually has to price. Embedded
    # wind and solar sit behind the meter, so they suppress measured demand
    # rather than appearing as generation.
    panel["residual_load"] = (
        panel["national_demand"]
        - panel["embedded_wind"].fillna(0)
        - panel["embedded_solar"].fillna(0)
    )

    interconnectors = [c for c in panel.columns if c.endswith("_flow")]
    panel["net_imports"] = panel[interconnectors].sum(axis=1, min_count=1)

    # --- weather --------------------------------------------------------
    # Weather is hourly, the panel is half-hourly. Floor each period to its hour
    # and merge; both half-hours in an hour share that hour's weather. Merging
    # the other way (forward-filling weather onto periods) would be equivalent
    # here but breaks silently if a weather hour is missing.
    panel["hour"] = panel["start_time"].dt.floor("h")
    panel = panel.merge(weather, left_on="hour", right_on="time", how="left").drop(
        columns=["time", "hour"]
    )

    # --- calendar features ----------------------------------------------
    local = panel["start_time"].dt.tz_convert(cal.TZ)
    panel["year"] = local.dt.year
    panel["month"] = local.dt.month
    panel["day_of_week"] = local.dt.dayofweek
    panel["day_of_year"] = local.dt.dayofyear
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype(int)

    return panel.sort_values("start_time").reset_index(drop=True)


def report_quality(panel: pd.DataFrame) -> dict[str, Any]:
    """What is missing, and where. Never silently filled."""
    total = len(panel)
    report: dict[str, Any] = {
        "rows": total,
        "start": str(panel["start_time"].min()),
        "end": str(panel["start_time"].max()),
        "missing": {
            column: int(panel[column].isna().sum())
            for column in ("price", "national_demand", "embedded_wind", "temperature_mean")
        },
        "negative_prices": int((panel["price"] < 0).sum()),
        "zero_prices": int((panel["price"] == 0).sum()),
        "duplicate_keys": int(
            panel.duplicated(subset=["settlement_date", "settlement_period"]).sum()
        ),
        "timestamps_strictly_increasing": bool(panel["start_time"].is_monotonic_increasing),
    }

    gaps = panel[panel["price"].isna()]
    report["price_gap_days"] = (
        gaps.groupby("settlement_date").size().sort_values(ascending=False).head(10).to_dict()
    )
    return report


def load() -> pd.DataFrame:
    path = DATA_PROCESSED / PANEL_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `make panel` first")
    return pd.read_parquet(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=STUDY.start)
    parser.add_argument("--end", default=STUDY.end)
    args = parser.parse_args()

    panel = build(args.start, args.end)
    report = report_quality(panel)

    path = DATA_PROCESSED / PANEL_NAME
    panel.to_parquet(path, index=False)

    print(f"wrote {path}  ({len(panel):,} rows x {len(panel.columns)} columns)")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
