"""Elexon BMRS Insights — half-hourly GB market index prices.

The MID (Market Index Data) dataset gives, per settlement period, a
volume-weighted reference price for GB short-term power. We use
`dataProvider=APXMIDP`, which is populated across our whole study window.

`N2EXMIDP` is the other option and it is a trap: it returns **zero prices before
roughly 2020**. Training on that would teach the model that electricity was free
for a year, and nothing about the API tells you — you only find out by looking at
the data.

    python -m ppa.ingest.elexon --start 2019-01-01 --end 2025-06-30
"""

from __future__ import annotations

import argparse
import logging
from functools import partial

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ppa.config import ELEXON_BASE, ELEXON_DATA_PROVIDER, STUDY
from ppa.ingest.cache import cached, fixed_chunks, load_all

log = logging.getLogger(__name__)

# Hard limit enforced by the endpoint, stated only in its 400 response.
ELEXON_MAX_RANGE_DAYS = 7

SOURCE = "elexon_mid"
ENDPOINT = f"{ELEXON_BASE}/balancing/pricing/market-index"


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def _fetch_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One request. Retried with exponential backoff on transient failure.

    The `to` bound is exclusive of the next day's first period, so we ask for
    end + 1 day and filter, rather than losing the last day of each chunk.
    """
    params = {
        "from": start.strftime("%Y-%m-%dT00:00Z"),
        "to": (end + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00Z"),
        "dataProviders": ELEXON_DATA_PROVIDER,
        "format": "json",
    }
    response = requests.get(ENDPOINT, params=params, timeout=90)
    response.raise_for_status()
    records = response.json().get("data", [])

    if not records:
        log.warning("no Elexon data for %s..%s", start.date(), end.date())
        return pd.DataFrame(
            columns=["start_time", "settlement_date", "settlement_period", "price", "volume"]
        )

    frame = pd.DataFrame(records)
    return pd.DataFrame(
        {
            "start_time": pd.to_datetime(frame["startTime"], utc=True),
            "settlement_date": pd.to_datetime(frame["settlementDate"]).dt.date.astype(str),
            "settlement_period": frame["settlementPeriod"].astype(int),
            "price": pd.to_numeric(frame["price"], errors="coerce"),
            "volume": pd.to_numeric(frame["volume"], errors="coerce"),
        }
    )


def ingest(start: str, end: str, force: bool = False) -> pd.DataFrame:
    """Fetch (or reuse cached) prices in 7-day windows.

    Seven days is the maximum the endpoint accepts; anything longer returns a
    400 whose body is the only place that limit is documented.
    """
    frames = []
    for chunk_start, chunk_end in fixed_chunks(start, end, days=ELEXON_MAX_RANGE_DAYS):
        key = chunk_start.strftime("%Y-%m-%d")
        frames.append(
            cached(SOURCE, key, partial(_fetch_window, chunk_start, chunk_end), force)
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["price"])
    combined = combined.drop_duplicates(subset=["settlement_date", "settlement_period"])
    return combined.sort_values("start_time").reset_index(drop=True)


def load() -> pd.DataFrame:
    """Read every cached chunk without touching the network."""
    frame = load_all(SOURCE)
    frame = frame.dropna(subset=["price"]).drop_duplicates(
        subset=["settlement_date", "settlement_period"]
    )
    return frame.sort_values("start_time").reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=STUDY.start)
    parser.add_argument("--end", default=STUDY.end)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    frame = ingest(args.start, args.end, force=args.force)

    print(f"{len(frame):,} settlement periods  {frame['start_time'].min()} .. {frame['start_time'].max()}")
    print(f"price  mean {frame['price'].mean():.2f}  min {frame['price'].min():.2f}  max {frame['price'].max():.2f}")
    print(f"negative prices: {(frame['price'] < 0).sum():,}  zero prices: {(frame['price'] == 0).sum():,}")


if __name__ == "__main__":
    main()
