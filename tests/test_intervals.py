"""Tests for quantile forecasts, interval scoring, and conformal calibration.

The conformal guarantee is a *finite-sample* one, so it can be tested directly
rather than approximately: build data that satisfies exchangeability, calibrate,
and assert the empirical coverage lands at nominal. Several tests below do
exactly that, which is why they use large synthetic samples instead of the real
panel — on real prices exchangeability fails, and a test that asserted nominal
coverage there would be asserting something untrue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ppa.eval import metrics
from ppa.models import conformal

# --- pinball loss ------------------------------------------------------------


def test_pinball_at_the_median_is_half_the_absolute_error() -> None:
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 30.0])

    assert metrics.pinball(y, p, 0.5) == pytest.approx(np.mean(np.abs(y - p)) / 2)


def test_pinball_penalises_asymmetrically_at_a_tail_quantile() -> None:
    """At q=0.1, over-predicting must cost 9x under-predicting."""
    truth = np.array([100.0])
    over = metrics.pinball(truth, np.array([110.0]), 0.1)
    under = metrics.pinball(truth, np.array([90.0]), 0.1)

    assert over == pytest.approx(9 * under)


def test_pinball_is_minimised_at_the_true_quantile() -> None:
    """The property that makes it a proper scoring rule."""
    rng = np.random.default_rng(0)
    sample = rng.normal(50, 10, 20_000)
    candidates = np.arange(30, 70, 0.5)

    for level in (0.1, 0.5, 0.9):
        losses = [metrics.pinball(sample, np.full_like(sample, c), level) for c in candidates]
        best = candidates[int(np.argmin(losses))]
        assert best == pytest.approx(np.quantile(sample, level), abs=1.0)


# --- interval scoring --------------------------------------------------------


def test_coverage_counts_points_inside_the_band() -> None:
    y = np.array([1.0, 5.0, 9.0, 15.0])
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([10.0, 10.0, 10.0, 10.0])

    result = metrics.interval_score(y, lo, hi, alpha=0.2)

    assert result["coverage"] == pytest.approx(0.75)
    assert result["breaches_high"] == 1
    assert result["breaches_low"] == 0
    assert result["mean_width"] == pytest.approx(10.0)


def test_winkler_punishes_widening_so_coverage_cannot_be_bought() -> None:
    """The reason coverage is never reported without width beside it."""
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 5_000)

    tight = metrics.interval_score(y, np.full(5_000, -1.28), np.full(5_000, 1.28), alpha=0.2)
    absurd = metrics.interval_score(y, np.full(5_000, -100.0), np.full(5_000, 100.0), alpha=0.2)

    # The absurd band has perfect coverage...
    assert absurd["coverage"] == 1.0
    assert absurd["coverage"] > tight["coverage"]
    # ...and a far worse Winkler score, which is the point.
    assert absurd["winkler"] > tight["winkler"]


def test_crossings_are_counted_not_assumed_away() -> None:
    y = np.array([5.0, 5.0])
    lo = np.array([0.0, 8.0])
    hi = np.array([10.0, 2.0])

    assert metrics.interval_score(y, lo, hi)["crossings"] == 1


def test_empty_input_returns_nan_rather_than_raising() -> None:
    result = metrics.interval_score(np.array([]), np.array([]), np.array([]))

    assert result["n"] == 0
    assert np.isnan(result["coverage"])


# --- conformal calibration ---------------------------------------------------


def test_conformity_score_is_negative_inside_the_interval() -> None:
    """So that an over-wide interval can be narrowed, not only widened."""
    scores = conformal.conformity_scores(
        np.array([5.0]), np.array([0.0]), np.array([10.0])
    )

    assert scores[0] == pytest.approx(-5.0)


def test_conformity_score_measures_distance_outside() -> None:
    below = conformal.conformity_scores(np.array([-3.0]), np.array([0.0]), np.array([10.0]))
    above = conformal.conformity_scores(np.array([14.0]), np.array([0.0]), np.array([10.0]))

    assert below[0] == pytest.approx(3.0)
    assert above[0] == pytest.approx(4.0)


def test_calibration_restores_nominal_coverage_when_exchangeable() -> None:
    """The finite-sample guarantee, asserted directly.

    A deliberately over-confident interval — half the width it should be — is
    calibrated on one exchangeable sample and evaluated on another.
    """
    rng = np.random.default_rng(2)
    calibration = rng.normal(0, 10, 5_000)
    test = rng.normal(0, 10, 5_000)

    # Nominal 80% would need +/- 12.8; use +/- 6.
    too_narrow = 6.0
    adjuster = conformal.fit(
        calibration, np.full(5_000, -too_narrow), np.full(5_000, too_narrow), alpha=0.2
    )

    lo, hi = adjuster.apply(np.full(5_000, -too_narrow), np.full(5_000, too_narrow))
    coverage = float(np.mean((test >= lo) & (test <= hi)))

    assert adjuster.global_offset > 0
    assert coverage == pytest.approx(0.80, abs=0.02)


def test_calibration_narrows_an_over_wide_interval() -> None:
    """Conformal prediction is a two-way correction, not a widening rule."""
    rng = np.random.default_rng(3)
    sample = rng.normal(0, 1, 5_000)
    far_too_wide = 50.0

    adjuster = conformal.fit(
        sample, np.full(5_000, -far_too_wide), np.full(5_000, far_too_wide), alpha=0.2
    )

    assert adjuster.global_offset < 0
    lo, hi = adjuster.apply(np.array([-far_too_wide]), np.array([far_too_wide]))
    assert (hi - lo)[0] < 2 * far_too_wide


def test_finite_sample_correction_is_applied() -> None:
    """(n+1)(1-alpha)/n, not the plain empirical quantile."""
    n = 100
    assert conformal._quantile_level(n, 0.2) == pytest.approx(np.ceil(101 * 0.8) / 100)
    # And it is capped at 1.0, so a tiny calibration set does not ask for a
    # quantile above the maximum observed score.
    assert conformal._quantile_level(3, 0.01) == 1.0


def test_per_period_offsets_differ_when_uncertainty_differs() -> None:
    """One scalar cannot express that 04:00 and the evening peak differ."""
    rng = np.random.default_rng(4)
    n = 4_000
    periods = np.tile([1, 35], n // 2)
    # Period 35 (the evening peak) is four times as volatile.
    noise = np.where(periods == 35, rng.normal(0, 40, n), rng.normal(0, 10, n))

    adjuster = conformal.fit(
        noise, np.full(n, -12.0), np.full(n, 12.0), alpha=0.2, periods=periods
    )

    assert adjuster.per_period[35] > adjuster.per_period[1]


def test_a_thin_period_falls_back_to_the_global_offset() -> None:
    rng = np.random.default_rng(5)
    n = 1_000
    periods = np.array([1] * (n - 5) + [48] * 5)
    noise = rng.normal(0, 10, n)

    adjuster = conformal.fit(
        noise, np.full(n, -5.0), np.full(n, 5.0), alpha=0.2, periods=periods
    )

    assert 48 not in adjuster.per_period
    lo, hi = adjuster.apply(np.array([-5.0]), np.array([5.0]), np.array([48]))
    assert (hi - lo)[0] == pytest.approx(10 + 2 * adjuster.global_offset)


def test_calibration_split_takes_the_most_recent_window() -> None:
    """Not a random sample — the model is about to forecast what comes next."""
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    train = pd.DataFrame({"date": dates, "price": range(200)})

    fit_part, calibration = conformal.split_calibration(train, calibration_days=30)

    assert len(calibration) == 30
    assert calibration["date"].min() > fit_part["date"].max()
    assert calibration["date"].max() == dates.max()


def test_a_window_too_short_to_split_does_not_silently_self_calibrate() -> None:
    """Calibrating on the fitting data would give offset ~0 and a false result."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    train = pd.DataFrame({"date": dates, "price": range(10)})

    fit_part, calibration = conformal.split_calibration(train, calibration_days=60)

    assert len(calibration) == 0
    assert len(fit_part) == len(train)


def test_empty_calibration_set_yields_a_no_op_rather_than_a_crash() -> None:
    adjuster = conformal.fit(np.array([]), np.array([]), np.array([]), alpha=0.2)

    assert adjuster.global_offset == 0.0
    lo, hi = adjuster.apply(np.array([1.0]), np.array([2.0]))
    assert (lo[0], hi[0]) == (1.0, 2.0)
