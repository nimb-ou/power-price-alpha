"""Panel and ingest tests.

The panel tests are the ones that would catch a silently misaligned join — the
failure mode that produces a normal-looking frame and a meaningless model.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ppa.config import DATA_PROCESSED, PERIODS_PER_NORMAL_DAY
from ppa.data import calendar as cal
from ppa.ingest.cache import fixed_chunks, month_chunks

# --- chunking (no data needed) -----------------------------------------------


def test_fixed_chunks_respect_the_window_length() -> None:
    chunks = fixed_chunks("2024-01-01", "2024-01-31", days=7)
    for start, end in chunks:
        assert (end - start).days <= 6  # inclusive range of 7 days


def test_fixed_chunks_cover_the_range_without_gaps() -> None:
    chunks = fixed_chunks("2024-01-01", "2024-03-15", days=7)
    assert chunks[0][0] == pd.Timestamp("2024-01-01")
    assert chunks[-1][1] == pd.Timestamp("2024-03-15")
    for (_, end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
        assert next_start - end == pd.Timedelta(days=1)


def test_fixed_chunks_are_stable_when_the_end_extends() -> None:
    """Cache keys must not re-partition when re-run with a later end date."""
    short = fixed_chunks("2024-01-01", "2024-02-01", days=7)
    long = fixed_chunks("2024-01-01", "2024-06-01", days=7)
    assert [c[0] for c in short] == [c[0] for c in long[: len(short)]]


def test_month_chunks_align_to_calendar_months() -> None:
    chunks = month_chunks("2024-01-15", "2024-03-10")
    assert chunks[0] == (pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-31"))
    assert chunks[-1][1] == pd.Timestamp("2024-03-10")


# --- the assembled panel -----------------------------------------------------

needs_panel = pytest.mark.skipif(
    not (DATA_PROCESSED / "panel.parquet").exists(),
    reason="no panel; run `make panel` first",
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "panel.parquet")


@needs_panel
def test_panel_matches_the_expected_settlement_grid(panel: pd.DataFrame) -> None:
    """The whole point of building against calendar.expected_periods."""
    start = panel["settlement_date"].min()
    end = panel["settlement_date"].max()
    expected = cal.expected_periods(start, end)

    assert len(panel) == len(expected)
    merged = expected.merge(
        panel[["settlement_date", "settlement_period"]],
        on=["settlement_date", "settlement_period"],
        how="left",
        indicator=True,
    )
    assert (merged["_merge"] == "both").all()


@needs_panel
def test_no_duplicate_settlement_periods(panel: pd.DataFrame) -> None:
    assert not panel.duplicated(subset=["settlement_date", "settlement_period"]).any()


@needs_panel
def test_timestamps_are_strictly_increasing_and_half_hourly(panel: pd.DataFrame) -> None:
    times = panel["start_time"].sort_values()
    gaps = times.diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(minutes=30)]


@needs_panel
def test_clock_change_days_have_the_right_period_counts(panel: pd.DataFrame) -> None:
    counts = panel.groupby("settlement_date").size()
    for date in ("2024-03-31", "2023-03-26", "2022-03-27"):
        if date in counts.index:
            assert counts[date] == 46, date
    for date in ("2024-10-27", "2023-10-29", "2022-10-30"):
        if date in counts.index:
            assert counts[date] == 50, date


@needs_panel
def test_ordinary_days_have_48_periods(panel: pd.DataFrame) -> None:
    counts = panel.groupby("settlement_date").size()
    ordinary = counts[~counts.index.map(cal.is_transition_day)]
    assert (ordinary == PERIODS_PER_NORMAL_DAY).all()


@needs_panel
def test_demand_and_weather_joined_completely(panel: pd.DataFrame) -> None:
    """A join failure would show up as a block of NaNs, not an exception."""
    assert panel["national_demand"].isna().sum() == 0
    assert panel["temperature_mean"].isna().sum() == 0


@needs_panel
def test_residual_load_is_demand_net_of_embedded_generation(panel: pd.DataFrame) -> None:
    expected = (
        panel["national_demand"]
        - panel["embedded_wind"].fillna(0)
        - panel["embedded_solar"].fillna(0)
    )
    pd.testing.assert_series_equal(
        panel["residual_load"], expected, check_names=False, rtol=1e-9
    )


@needs_panel
def test_residual_load_never_exceeds_demand(panel: pd.DataFrame) -> None:
    assert (panel["residual_load"] <= panel["national_demand"] + 1e-6).all()


@needs_panel
def test_prices_include_negatives_and_are_not_clipped(panel: pd.DataFrame) -> None:
    """GB prices genuinely go negative; a pipeline that clipped them is broken."""
    prices = panel["price"].dropna()
    assert (prices < 0).sum() > 0
    assert prices.min() < -5


@needs_panel
def test_price_gaps_have_the_expected_two_part_structure(panel: pd.DataFrame) -> None:
    """Missing prices come from two distinct, benign causes — not a join bug.

    Measured over 2019-2025: 2,060 of 113,902 periods (1.8%) have no price.

    1. **Whole-day outages** — 36 days where Elexon published nothing (a run in
       Jan 2019, several in Nov 2020). 1,722 rows, **83.6% of all missing**.
    2. **Partial-day outages** — 22 days losing 4-45 periods, 283 rows.
    3. **Isolated single periods** — 55 rows across 36 other days. MID is a
       volume-weighted average of real trades, so a period with no trades has
       no index price.

    A hypothesis worth recording *because it was checked and rejected*: that the
    isolated gaps would cluster in thin overnight periods. They do not. They are
    spread across the day (29% fall in SP1-6, barely above the 12.5% you would
    get by chance), and median volume in the affected periods is no lower than
    elsewhere. So the assertions below deliberately do **not** claim a pattern
    in those 55 rows — 0.05% of the panel is too little to characterise, and
    asserting a shape on it would be exactly the small-sample overreach this
    project warns about elsewhere.

    What a genuine join bug would look like, and what these assertions catch:
    missing rows spread evenly over dates rather than concentrated in outage
    days, and a much larger overall loss.
    """
    gaps = panel[panel["price"].isna()]
    if gaps.empty:
        return

    per_day = gaps.groupby("settlement_date").size()
    whole_days = per_day[per_day >= 46]

    # Missing data is dominated by full-day outages, not spread thinly.
    assert whole_days.sum() / len(gaps) > 0.75

    # Everything that is not a full-day outage is negligible.
    residual = len(gaps) - whole_days.sum()
    assert residual / len(panel) < 0.005

    # Overall loss stays small; a real join failure would be far larger.
    assert len(gaps) / len(panel) < 0.05


@needs_panel
def test_solar_generation_is_zero_at_night(panel: pd.DataFrame) -> None:
    """A sanity check that the weather/demand join is aligned to real local time."""
    local_hour = panel["start_time"].dt.tz_convert(cal.TZ).dt.hour
    night = panel[(local_hour >= 23) | (local_hour <= 2)]
    assert night["embedded_solar"].fillna(0).abs().max() < 1.0
