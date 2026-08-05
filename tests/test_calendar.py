"""Settlement-period calendar tests.

These are the highest-value tests in the repo. A calendar bug does not raise —
it produces a plausible frame with silently misaligned rows, and it poisons
every lag feature for a week either side of each clock change. There is no
downstream metric that would catch it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ppa.data import calendar as cal

# The four spring/autumn transitions we exercise directly, plus their period counts.
SPRING_2024 = "2024-03-31"  # clocks forward: 46 periods
AUTUMN_2024 = "2024-10-27"  # clocks back:   50 periods
NORMAL_SUMMER = "2024-06-15"  # BST, 48 periods
NORMAL_WINTER = "2024-01-15"  # GMT, 48 periods


# --- period counts -----------------------------------------------------------


@pytest.mark.parametrize(
    "date,expected",
    [
        (NORMAL_WINTER, 48),
        (NORMAL_SUMMER, 48),
        (SPRING_2024, 46),
        (AUTUMN_2024, 50),
        ("2019-03-31", 46),
        ("2019-10-27", 50),
        ("2021-10-31", 50),
        ("2025-03-30", 46),
    ],
)
def test_periods_in_day(date: str, expected: int) -> None:
    assert cal.periods_in_day(date) == expected


def test_every_year_has_exactly_two_transitions() -> None:
    for year in range(2019, 2025):
        days = cal.transition_days(f"{year}-01-01", f"{year}-12-31")
        assert len(days) == 2, f"{year}: {days.to_dict('records')}"
        assert sorted(days["periods"]) == [46, 50]


def test_transitions_are_always_sundays() -> None:
    days = cal.transition_days("2019-01-01", "2025-06-30")
    assert (pd.to_datetime(days["date"]).dt.dayofweek == 6).all()


def test_transition_period_counts_net_out_over_a_year() -> None:
    """46 + 50 = 96 = 2 x 48, which is why a naive yearly row count looks fine."""
    days = cal.transition_days("2024-01-01", "2024-12-31")
    assert days["periods"].sum() == 2 * 48


# --- conversion --------------------------------------------------------------


def test_period_one_is_local_midnight_in_winter() -> None:
    """GMT: local midnight is UTC midnight."""
    assert cal.to_utc(NORMAL_WINTER, 1) == pd.Timestamp("2024-01-15 00:00", tz="UTC")


def test_period_one_is_an_hour_earlier_in_summer() -> None:
    """BST is UTC+1, so local midnight is 23:00 UTC the previous day."""
    assert cal.to_utc(NORMAL_SUMMER, 1) == pd.Timestamp("2024-06-14 23:00", tz="UTC")


def test_spring_day_skips_the_missing_hour() -> None:
    """01:00 local jumps to 02:00. SP3 onwards must not stall."""
    assert cal.to_utc(SPRING_2024, 1) == pd.Timestamp("2024-03-31 00:00", tz="UTC")
    assert cal.to_utc(SPRING_2024, 2) == pd.Timestamp("2024-03-31 00:30", tz="UTC")
    # SP3 is 02:00 BST, which is 01:00 UTC — no gap in absolute time.
    assert cal.to_utc(SPRING_2024, 3) == pd.Timestamp("2024-03-31 01:00", tz="UTC")
    assert cal.to_utc(SPRING_2024, 46) == pd.Timestamp("2024-03-31 22:30", tz="UTC")


def test_autumn_day_covers_the_repeated_hour() -> None:
    """01:00-02:00 happens twice; periods run 1..50 without repeating."""
    assert cal.to_utc(AUTUMN_2024, 1) == pd.Timestamp("2024-10-26 23:00", tz="UTC")
    assert cal.to_utc(AUTUMN_2024, 50) == pd.Timestamp("2024-10-27 23:30", tz="UTC")


@pytest.mark.parametrize(
    "date,period",
    [
        (NORMAL_WINTER, 1), (NORMAL_WINTER, 24), (NORMAL_WINTER, 48),
        (NORMAL_SUMMER, 1), (NORMAL_SUMMER, 25), (NORMAL_SUMMER, 48),
        (SPRING_2024, 1), (SPRING_2024, 3), (SPRING_2024, 46),
        (AUTUMN_2024, 1), (AUTUMN_2024, 6), (AUTUMN_2024, 50),
    ],
)
def test_round_trip(date: str, period: int) -> None:
    assert cal.from_utc(cal.to_utc(date, period)) == (date, period)


def test_periods_are_strictly_increasing_in_absolute_time() -> None:
    """The property a naive local-wall-clock implementation breaks in autumn."""
    for date in (SPRING_2024, AUTUMN_2024, NORMAL_SUMMER):
        stamps = [cal.to_utc(date, p) for p in range(1, cal.periods_in_day(date) + 1)]
        assert all(b - a == pd.Timedelta(minutes=30) for a, b in zip(stamps, stamps[1:], strict=False))


def test_consecutive_days_join_without_gap_or_overlap() -> None:
    """The autumn boundary is where an off-by-one hour would show up."""
    for date, following in [(AUTUMN_2024, "2024-10-28"), (SPRING_2024, "2024-04-01")]:
        last = cal.to_utc(date, cal.periods_in_day(date))
        first = cal.to_utc(following, 1)
        assert first - last == pd.Timedelta(minutes=30)


# --- frame helpers -----------------------------------------------------------


def test_add_utc_column_matches_scalar_conversion() -> None:
    frame = pd.DataFrame(
        {
            "settlement_date": [NORMAL_WINTER, NORMAL_SUMMER, SPRING_2024, AUTUMN_2024],
            "settlement_period": [1, 1, 3, 50],
        }
    )
    out = cal.add_utc_column(frame)
    for _, row in out.iterrows():
        assert row["start_time"] == cal.to_utc(
            row["settlement_date"], row["settlement_period"]
        )


def test_expected_grid_has_the_right_shape() -> None:
    grid = cal.expected_periods("2024-03-30", "2024-04-01")
    counts = grid.groupby("settlement_date").size().to_dict()
    assert counts == {"2024-03-30": 48, "2024-03-31": 46, "2024-04-01": 48}


def test_expected_grid_for_a_full_year() -> None:
    """365 x 48 - 2 + 2 = 17,520. The transitions cancel, which is the trap."""
    grid = cal.expected_periods("2023-01-01", "2023-12-31")
    assert len(grid) == 365 * 48
    assert grid.groupby("settlement_date").size().min() == 46
    assert grid.groupby("settlement_date").size().max() == 50
