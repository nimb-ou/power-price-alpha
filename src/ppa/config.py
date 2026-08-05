"""Paths, market constants and study parameters.

Market constants live here rather than in the modules that use them because
several of them (gate closure, settlement-period length) are the kind of thing
that silently invalidates a backtest if two modules disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ARTIFACTS = REPO_ROOT / "artifacts"
REPORTS = REPO_ROOT / "reports"

for _p in (DATA_RAW, DATA_PROCESSED, ARTIFACTS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)


# --- Market structure -------------------------------------------------------

# GB settles electricity in half-hour blocks. A normal day has 48; the spring
# clock change has 46 and the autumn one has 50. Assuming 48 is the most common
# way to corrupt a GB power dataset — see ppa/data/calendar.py.
SETTLEMENT_PERIOD_MINUTES = 30
PERIODS_PER_NORMAL_DAY = 48

# The GB day-ahead auction clears around 11:00 local on day D for delivery on
# D+1. Everything used to forecast D+1 must be knowable at this instant; the
# leakage test in tests/test_no_leakage.py enforces it.
DAY_AHEAD_GATE_CLOSURE_LOCAL = "11:00"

MARKET_TIMEZONE = "Europe/London"


# --- Data sources -----------------------------------------------------------

ELEXON_BASE = "https://data.elexon.co.uk/bmrs/api/v1"
# APXMIDP is populated across the whole study window; N2EXMIDP returns zeros
# before ~2020, which is why it is not the default.
ELEXON_DATA_PROVIDER = "APXMIDP"

NESO_BASE = "https://api.neso.energy/api/3/action"
NESO_DEMAND_PACKAGE = "historic-demand-data"

OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# Population/demand-weighted sample of GB. Weather drives both demand (heating,
# lighting) and supply (wind, solar), so we take a spread rather than one point.
WEATHER_SITES: dict[str, tuple[float, float]] = {
    "london": (51.51, -0.13),
    "manchester": (53.48, -2.24),
    "birmingham": (52.49, -1.89),
    "glasgow": (55.86, -4.25),
    "cardiff": (51.48, -3.18),
    # Offshore-ish, for wind
    "dogger_bank": (54.75, 2.00),
}


# --- Study parameters -------------------------------------------------------


@dataclass(frozen=True)
class StudyConfig:
    start: str = "2019-01-01"
    end: str = "2025-06-30"

    # Walk-forward: train on an expanding window, re-fit every `refit_days`,
    # always predicting one day ahead.
    initial_train_days: int = 365 * 2
    refit_days: int = 30
    random_seed: int = 42

    # Trading frictions, in GBP/MWh. Deliberately conservative: a retail-scale
    # participant would not do better than this.
    transaction_cost: float = 0.50
    slippage: float = 0.25

    # Regime split. The GB gas crisis begins in earnest in Sep 2021 and prices
    # normalise through 2023. Used for the calm-vs-volatile breakdown.
    volatile_start: str = "2021-09-01"
    volatile_end: str = "2023-06-30"


STUDY = StudyConfig()
