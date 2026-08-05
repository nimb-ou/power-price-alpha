"""Build the committed test fixture from the local API cache.

CI must not depend on Elexon or NESO being up — a red build should mean our code
broke, never that a public API had a bad morning. So a short window of real data
is committed under `tests/fixtures/`, and the CI pipeline job runs the entire
chain (panel -> features -> walk-forward -> backtest -> report) against it.

Real data, not synthetic: the whole point of the pipeline job is to catch things
like a clock-change misalignment, and generated data would have whatever
structure we thought to put in it. The window deliberately **spans the October
clock change**, so CI exercises a 50-period day on every push.

Run this after a full `make ingest`:

    python tools/make_fixtures.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppa.ingest import elexon, neso, openmeteo  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "raw"

# Six months around the 2024 autumn transition. Long enough for a scaled-down
# walk-forward (90-day initial train, 30-day refits) to produce several folds.
START = "2024-07-01"
END = "2024-12-31"


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    prices = elexon.load()
    prices = prices[
        prices["settlement_date"].between(START, END)
    ].reset_index(drop=True)
    (FIXTURES / "elexon_mid").mkdir(exist_ok=True)
    prices.to_parquet(FIXTURES / "elexon_mid" / "fixture.parquet", index=False)

    demand = neso.load()
    demand = demand[demand["settlement_date"].between(START, END)].reset_index(drop=True)
    (FIXTURES / "neso_demand").mkdir(exist_ok=True)
    demand.to_parquet(FIXTURES / "neso_demand" / "fixture.parquet", index=False)

    weather = openmeteo.load()
    mask = weather["time"].dt.tz_convert("UTC").dt.date.astype(str).between(START, END)
    weather = weather[mask].reset_index(drop=True)
    (FIXTURES / "openmeteo").mkdir(exist_ok=True)
    weather.to_parquet(FIXTURES / "openmeteo" / "fixture.parquet", index=False)

    total = sum(f.stat().st_size for f in FIXTURES.rglob("*.parquet"))
    print(f"wrote fixtures to {FIXTURES.relative_to(ROOT)}  ({total / 1e6:.1f} MB)")
    print(f"  prices  {len(prices):,} periods")
    print(f"  demand  {len(demand):,} periods")
    print(f"  weather {len(weather):,} site-hours")

    days = prices.groupby("settlement_date").size()
    odd = days[days != 48]
    print(f"  clock-change days included: {odd.to_dict()}")
    if odd.empty:
        raise SystemExit("fixture window contains no clock change — widen it")


if __name__ == "__main__":
    main()
