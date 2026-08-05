"""NESO Data Portal — half-hourly GB demand and embedded renewables.

The single most useful series in this project is not a price lag. It is
**residual load**: national demand minus embedded wind and solar. Embedded
generation sits behind the meter on the distribution network, so it is invisible
to the transmission operator as *generation* and instead shows up as *suppressed
demand*. What the wholesale market has to price is what is left over — and that
is what sets the marginal generator, and therefore the price.

NESO publishes one CKAN resource per year, so the client resolves resource ids
from the package rather than hardcoding them (they change, and a hardcoded id
that 404s in 2027 is a bad legacy).

    python -m ppa.ingest.neso --start 2019-01-01 --end 2025-06-30
"""

from __future__ import annotations

import argparse
import logging
from functools import lru_cache, partial
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ppa.config import NESO_BASE, NESO_DEMAND_PACKAGE, STUDY
from ppa.ingest.cache import cached, load_all

log = logging.getLogger(__name__)

SOURCE = "neso_demand"
PAGE_SIZE = 20_000

# Columns we keep. Interconnector flows matter because imports displace domestic
# generation and therefore change which plant is marginal.
COLUMNS = {
    "SETTLEMENT_DATE": "settlement_date",
    "SETTLEMENT_PERIOD": "settlement_period",
    "ND": "national_demand",
    "TSD": "transmission_system_demand",
    "ENGLAND_WALES_DEMAND": "england_wales_demand",
    "EMBEDDED_WIND_GENERATION": "embedded_wind",
    "EMBEDDED_SOLAR_GENERATION": "embedded_solar",
    "EMBEDDED_WIND_CAPACITY": "embedded_wind_capacity",
    "EMBEDDED_SOLAR_CAPACITY": "embedded_solar_capacity",
    "PUMP_STORAGE_PUMPING": "pump_storage_pumping",
    "IFA_FLOW": "ifa_flow",
    "IFA2_FLOW": "ifa2_flow",
    "BRITNED_FLOW": "britned_flow",
    "NEMO_FLOW": "nemo_flow",
    "NSL_FLOW": "nsl_flow",
    "ELECLINK_FLOW": "eleclink_flow",
    "VIKING_FLOW": "viking_flow",
    "MOYLE_FLOW": "moyle_flow",
    "EAST_WEST_FLOW": "east_west_flow",
}


@lru_cache(maxsize=1)
def _resource_ids() -> dict[int, str]:
    """Map year -> CKAN resource id, read from the package metadata."""
    response = requests.get(
        f"{NESO_BASE}/package_show", params={"id": NESO_DEMAND_PACKAGE}, timeout=60
    )
    response.raise_for_status()

    ids: dict[int, str] = {}
    for resource in response.json()["result"]["resources"]:
        name = resource.get("name", "")
        for token in name.split():
            if token.isdigit() and len(token) == 4:
                ids[int(token)] = resource["id"]
                break
    log.info("resolved %d yearly NESO resources", len(ids))
    return ids


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def _fetch_page(resource_id: str, offset: int) -> list[dict]:
    params: dict[str, Any] = {
        "resource_id": resource_id,
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    response = requests.get(
        f"{NESO_BASE}/datastore_search",
        params=params,
        timeout=120,
    )
    response.raise_for_status()
    records: list[dict] = response.json()["result"]["records"]
    return records


def _fetch_year(year: int) -> pd.DataFrame:
    """One year, paged. A year is ~17,520 rows, so this is one or two pages."""
    resource_id = _resource_ids().get(year)
    if resource_id is None:
        log.warning("no NESO resource for %d", year)
        return pd.DataFrame(columns=list(COLUMNS.values()))

    records: list[dict] = []
    offset = 0
    while True:
        page = _fetch_page(resource_id, offset)
        records.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=list(COLUMNS.values()))

    # Not every year carries every interconnector — Viking and Greenlink only
    # exist from 2023/24 — so select what is present and leave the rest absent
    # for the assembler to fill.
    present = {src: dst for src, dst in COLUMNS.items() if src in frame.columns}
    out = frame[list(present)].rename(columns=present)

    out["settlement_date"] = pd.to_datetime(out["settlement_date"]).dt.date.astype(str)
    out["settlement_period"] = out["settlement_period"].astype(int)
    for column in out.columns:
        if column not in ("settlement_date", "settlement_period"):
            out[column] = pd.to_numeric(out[column], errors="coerce")

    return out


def ingest(start: str, end: str, force: bool = False) -> pd.DataFrame:
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    frames = [
        cached(SOURCE, str(year), partial(_fetch_year, year), force) for year in years
    ]

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["settlement_date", "settlement_period"])

    mask = (combined["settlement_date"] >= str(pd.Timestamp(start).date())) & (
        combined["settlement_date"] <= str(pd.Timestamp(end).date())
    )
    return combined[mask].sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)


def load() -> pd.DataFrame:
    frame = load_all(SOURCE).drop_duplicates(subset=["settlement_date", "settlement_period"])
    return frame.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=STUDY.start)
    parser.add_argument("--end", default=STUDY.end)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    frame = ingest(args.start, args.end, force=args.force)
    residual = frame["national_demand"] - frame["embedded_wind"] - frame["embedded_solar"]

    print(f"{len(frame):,} rows  {frame['settlement_date'].min()} .. {frame['settlement_date'].max()}")
    print(f"national demand  mean {frame['national_demand'].mean():,.0f} MW")
    print(f"residual load    mean {residual.mean():,.0f} MW  min {residual.min():,.0f}  max {residual.max():,.0f}")
    print(f"columns: {list(frame.columns)}")


if __name__ == "__main__":
    main()
