"""Settlement-period arithmetic — the thing that silently corrupts GB power data.

A GB day is divided into half-hourly **settlement periods**, numbered from 1,
where period 1 starts at 00:00 **local time**. Local, not UTC. So:

| Day | Periods |
|---|---|
| Normal | 48 |
| Last Sunday in March (clocks forward) | **46** — 01:00 jumps to 02:00 |
| Last Sunday in October (clocks back) | **50** — 01:00-02:00 happens twice |

Our two sources disagree about how they express time:

- **Elexon** returns a UTC `startTime` per record.
- **NESO** returns `SETTLEMENT_DATE` (a local calendar date) + `SETTLEMENT_PERIOD`.

The tempting join is `timestamp = date + (period - 1) x 30min`, treating the
date as UTC. That is wrong for six months of every year (BST is UTC+1) and
catastrophically wrong on the two transition days. Worse, it fails *silently*:
you get a plausible-looking frame with misaligned rows, and every lag feature
computed as "48 periods ago" is wrong for a week either side of each transition.

Everything in this module exists to convert correctly in both directions, and
`tests/test_calendar.py` pins the behaviour on the transition days in our window.
"""

from __future__ import annotations

import pandas as pd

from ppa.config import MARKET_TIMEZONE, PERIODS_PER_NORMAL_DAY

TZ = MARKET_TIMEZONE


def periods_in_day(date: str | pd.Timestamp) -> int:
    """How many settlement periods a given local date contains: 46, 48 or 50.

    Computed from the actual UTC offset at midnight and at the following
    midnight rather than from a table of transition dates, so it stays correct
    if the UK ever changes its DST rules.
    """
    day = pd.Timestamp(date).normalize()
    start = day.tz_localize(TZ, nonexistent="shift_forward", ambiguous=True)
    end = (day + pd.Timedelta(days=1)).tz_localize(
        TZ, nonexistent="shift_forward", ambiguous=True
    )
    hours = (end - start).total_seconds() / 3600
    return int(round(hours * 2))


def is_transition_day(date: str | pd.Timestamp) -> bool:
    return periods_in_day(date) != PERIODS_PER_NORMAL_DAY


def to_utc(date: str | pd.Timestamp, period: int) -> pd.Timestamp:
    """(local settlement date, period) -> UTC timestamp of the period's start.

    Built by offsetting from local midnight in *absolute* time. Adding a
    timedelta to a tz-aware instant crosses a DST boundary correctly, whereas
    constructing a local wall-clock time and then localising it hits the
    nonexistent hour in spring and the ambiguous hour in autumn.
    """
    day = pd.Timestamp(date).normalize()
    midnight = day.tz_localize(TZ, nonexistent="shift_forward", ambiguous=True)
    return (midnight + pd.Timedelta(minutes=30 * (period - 1))).tz_convert("UTC")


def from_utc(timestamp: pd.Timestamp) -> tuple[str, int]:
    """UTC timestamp -> (local settlement date, settlement period).

    The inverse of `to_utc`. Period is derived from elapsed absolute time since
    local midnight, so it counts 1..50 on the autumn transition day rather than
    repeating 3..4.
    """
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")

    local = stamp.tz_convert(TZ)
    day = local.normalize()
    midnight = day.tz_localize(None).tz_localize(
        TZ, nonexistent="shift_forward", ambiguous=True
    )
    elapsed = (stamp - midnight).total_seconds()
    period = int(elapsed // 1800) + 1
    return str(day.date()), period


def add_utc_column(
    frame: pd.DataFrame,
    date_col: str = "settlement_date",
    period_col: str = "settlement_period",
    out_col: str = "start_time",
) -> pd.DataFrame:
    """Vectorised (date, period) -> UTC for a whole frame.

    Loops per distinct date rather than per row: 2,400 dates instead of 115,000
    rows, and the per-date midnight is the only expensive part.
    """
    out = frame.copy()
    midnights = {
        date: pd.Timestamp(date)
        .normalize()
        .tz_localize(TZ, nonexistent="shift_forward", ambiguous=True)
        for date in out[date_col].unique()
    }
    # A Series of tz-aware Timestamps, all in Europe/London but with different
    # UTC offsets across the year. Converting via `.dt` (not `.tz_convert`,
    # which would operate on the Series *index*) preserves that.
    base = pd.to_datetime(out[date_col].map(midnights))
    offsets = pd.to_timedelta((out[period_col].astype(int) - 1) * 30, unit="m")
    out[out_col] = (base + offsets).dt.tz_convert("UTC")
    return out


def expected_periods(start: str, end: str) -> pd.DataFrame:
    """The complete (date, period) grid for a range, honouring 46/48/50.

    This is the reference the assembled panel is checked against — it is how we
    detect a missing period rather than quietly interpolating over it.
    """
    rows: list[dict[str, object]] = []
    for day in pd.date_range(start, end, freq="D"):
        date = str(day.date())
        for period in range(1, periods_in_day(day) + 1):
            rows.append({"settlement_date": date, "settlement_period": period})
    return pd.DataFrame(rows)


def transition_days(start: str, end: str) -> pd.DataFrame:
    """Every clock-change day in a range, with its period count."""
    rows = [
        {"date": str(day.date()), "periods": periods_in_day(day)}
        for day in pd.date_range(start, end, freq="D")
        if is_transition_day(day)
    ]
    return pd.DataFrame(rows)
