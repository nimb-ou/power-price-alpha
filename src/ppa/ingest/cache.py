"""Parquet caching for API responses.

Six years of half-hourly GB data is roughly 115,000 rows per series, fetched in
chunks from three public APIs. Refetching on every run would be slow, rude to
free services, and would make results non-reproducible when a provider restates
history.

So every chunk is cached to parquet under `data/raw/`, keyed by source and date
range. `make ingest` is idempotent and offline after the first run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from ppa.config import DATA_RAW

log = logging.getLogger(__name__)


def cache_path(source: str, key: str) -> Path:
    directory = DATA_RAW / source
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.parquet"


def cached(
    source: str,
    key: str,
    fetch: Callable[[], pd.DataFrame],
    force: bool = False,
) -> pd.DataFrame:
    """Return a cached frame, fetching and storing it if absent.

    An empty result is cached too. A date range with genuinely no data (a gap in
    a provider's history) would otherwise be refetched on every run forever.
    """
    path = cache_path(source, key)

    if path.exists() and not force:
        return pd.read_parquet(path)

    frame = fetch()
    frame.to_parquet(path, index=False)
    log.debug("cached %s rows -> %s", len(frame), path)
    return frame


def month_chunks(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a date range into calendar months."""
    begin = pd.Timestamp(start).normalize()
    finish = pd.Timestamp(end).normalize()

    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = begin
    while cursor <= finish:
        month_end = (cursor + pd.offsets.MonthEnd(1)).normalize()
        chunks.append((cursor, min(month_end, finish)))
        cursor = month_end + pd.Timedelta(days=1)
    return chunks


def fixed_chunks(start: str, end: str, days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a date range into fixed-length windows, anchored at `start`.

    Elexon's market-index endpoint rejects any range longer than 7 days — a
    limit that is not in the documentation, only in the 400 response body. Six
    and a half years is therefore ~340 requests.

    Anchoring at `start` rather than at calendar boundaries keeps the cache keys
    stable: re-running with a later `end` reuses every window already fetched
    instead of re-partitioning the whole history.
    """
    begin = pd.Timestamp(start).normalize()
    finish = pd.Timestamp(end).normalize()

    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = begin
    while cursor <= finish:
        window_end = cursor + pd.Timedelta(days=days - 1)
        chunks.append((cursor, min(window_end, finish)))
        cursor = window_end + pd.Timedelta(days=1)
    return chunks


def load_all(source: str) -> pd.DataFrame:
    """Concatenate every cached chunk for a source."""
    directory = DATA_RAW / source
    if not directory.exists():
        raise FileNotFoundError(f"no cache at {directory} — run `make ingest` first")

    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"cache at {directory} is empty — run `make ingest` first")

    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)
