"""No-look-ahead tests.

The single question a quant interviewer will ask about this project. These tests
answer it in code rather than in prose.

The rule under test: to forecast day T, every feature must be derived from data
timestamped strictly before **11:00 local on T-1** (day-ahead gate closure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppa.config import DATA_PROCESSED
from ppa.data import calendar as cal
from ppa.features.build import (
    LAST_PERIOD_BEFORE_GATE,
    PERIOD_LAGS_DAYS,
    decision_time,
    feature_columns,
)
from ppa.models import naive
from ppa.models.walkforward import make_folds

pytestmark = pytest.mark.skipif(
    not (DATA_PROCESSED / "features.parquet").exists(),
    reason="no feature matrix; run `make features` first",
)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "features.parquet")


# --- the information set -----------------------------------------------------


def test_decision_time_is_gate_closure_on_the_previous_day(features: pd.DataFrame) -> None:
    sample = features.head(2000)
    for _, row in sample.iterrows():
        local = row["decision_time"].tz_convert(cal.TZ)
        target = pd.Timestamp(row["settlement_date"])
        assert local.hour == 11 and local.minute == 0
        assert (target.date() - local.date()).days == 1


def test_every_target_is_after_its_own_decision_time(features: pd.DataFrame) -> None:
    """A forecast must be for something that has not happened yet."""
    assert (features["start_time"] > features["decision_time"]).all()


def test_no_one_day_period_lag_exists(features: pd.DataFrame) -> None:
    """price_lag_1d is the tempting feature and it is not knowable at gate closure.

    The same settlement period on T-1 is up to 13 hours after 11:00 on T-1.
    """
    assert 1 not in PERIOD_LAGS_DAYS
    assert "price_lag_1d" not in features.columns
    assert "residual_lag_1d" not in features.columns


@pytest.mark.parametrize("days", PERIOD_LAGS_DAYS)
def test_period_lags_predate_gate_closure(features: pd.DataFrame, days: int) -> None:
    """Every same-period lag must resolve to an instant before the decision."""
    sample = features.dropna(subset=[f"price_lag_{days}d"]).head(500)
    for _, row in sample.iterrows():
        source_date = (pd.Timestamp(row["settlement_date"]) - pd.Timedelta(days=days)).date()
        source_time = cal.to_utc(str(source_date), int(row["settlement_period"]))
        assert source_time < row["decision_time"], (
            f"lag_{days}d for {row['settlement_date']} SP{row['settlement_period']} "
            f"resolves to {source_time}, at or after gate closure {row['decision_time']}"
        )


def test_morning_aggregates_stop_at_the_gate(features: pd.DataFrame) -> None:
    """SP22 ends at 11:00 local. SP23 has not happened at gate closure."""
    assert LAST_PERIOD_BEFORE_GATE == 22
    last_included = cal.to_utc("2024-01-15", LAST_PERIOD_BEFORE_GATE)
    first_excluded = cal.to_utc("2024-01-15", LAST_PERIOD_BEFORE_GATE + 1)
    gate = decision_time(pd.Series(["2024-01-16"])).iloc[0]

    # SP22 covers 10:30-11:00: it starts before the gate and ends exactly on it.
    assert last_included < gate
    assert last_included + pd.Timedelta(minutes=30) == gate
    assert first_excluded >= gate


def test_morning_aggregate_matches_a_hand_computed_value(features: pd.DataFrame) -> None:
    """Recompute price_am_mean for one day from the raw panel."""
    panel = pd.read_parquet(DATA_PROCESSED / "panel.parquet")
    target_date = "2024-06-15"
    previous = "2024-06-14"

    expected = panel[
        (panel["settlement_date"] == previous)
        & (panel["settlement_period"] <= LAST_PERIOD_BEFORE_GATE)
    ]["price"].mean()

    actual = features[features["settlement_date"] == target_date]["price_am_mean"].iloc[0]
    assert actual == pytest.approx(expected, rel=1e-6)


def test_rolling_windows_end_two_days_before_target(features: pd.DataFrame) -> None:
    """shift(2) before rolling is what keeps T-1's incomplete day out."""
    panel = pd.read_parquet(DATA_PROCESSED / "panel.parquet")
    daily = panel.groupby("settlement_date")["price"].mean()

    target = "2024-06-15"
    window_dates = [
        str((pd.Timestamp(target) - pd.Timedelta(days=d)).date()) for d in range(2, 9)
    ]
    expected = float(np.mean([daily[d] for d in window_dates]))

    actual = features[features["settlement_date"] == target]["price_roll_7d"].iloc[0]
    assert actual == pytest.approx(expected, rel=1e-6)


def test_no_feature_column_is_the_target(features: pd.DataFrame) -> None:
    columns = feature_columns(features)
    assert "price" not in columns
    assert "start_time" not in columns
    assert "decision_time" not in columns
    # `year` is excluded too: it would let a tree memorise regimes by date
    # rather than learn from market state.
    assert "year" not in columns


def test_no_datetime_column_reaches_the_model(features: pd.DataFrame) -> None:
    """Any timestamp column is a calendar-position shortcut, and a crash.

    The walk-forward harness adds a `date` helper to the frame it passes
    through; this asserts such columns can never be selected as features.
    """
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["settlement_date"])

    columns = feature_columns(frame)
    assert "date" not in columns
    for column in columns:
        assert not pd.api.types.is_datetime64_any_dtype(frame[column]), column


def test_no_feature_correlates_suspiciously_with_the_target(features: pd.DataFrame) -> None:
    """A near-perfect correlation would mean the target leaked in under a new name."""
    columns = feature_columns(features)
    sample = features[columns + ["price"]].dropna().sample(
        min(20000, len(features)), random_state=0
    )
    correlations = sample.corr()["price"].drop("price").abs()
    worst = correlations.idxmax()
    assert correlations.max() < 0.97, f"{worst} correlates {correlations.max():.4f} with price"


# --- baselines ---------------------------------------------------------------


def test_headline_baseline_is_declared_and_legitimate() -> None:
    assert naive.HEADLINE_BASELINE == "same_period_last_week"
    assert "same_period_yesterday" not in naive.BASELINES


def test_baselines_use_only_legitimate_lags(features: pd.DataFrame) -> None:
    week = naive.predict(features, "same_period_last_week")
    np.testing.assert_array_equal(week, features["price_lag_7d"].to_numpy())


# --- walk-forward ------------------------------------------------------------


def test_folds_never_train_on_the_future(features: pd.DataFrame) -> None:
    dates = pd.to_datetime(features["settlement_date"])
    for fold in make_folds(dates):
        assert fold.train_end < fold.test_start
        assert fold.test_start <= fold.test_end


def test_folds_are_contiguous_and_non_overlapping(features: pd.DataFrame) -> None:
    dates = pd.to_datetime(features["settlement_date"])
    folds = make_folds(dates)
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later.test_start > earlier.test_end
        # Expanding window: each fold trains on strictly more data.
        assert later.n_train > earlier.n_train


def test_folds_cover_the_period_after_the_initial_window(features: pd.DataFrame) -> None:
    dates = pd.to_datetime(features["settlement_date"])
    folds = make_folds(dates)
    unique = sorted(dates.unique())
    assert folds[0].test_start == unique[len(unique) - len(unique) + folds[0].n_train]
    assert folds[-1].test_end == unique[-1]
