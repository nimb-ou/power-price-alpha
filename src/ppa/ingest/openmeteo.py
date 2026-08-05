"""Open-Meteo archive — hourly weather at six GB sites.

Weather drives both sides of the market: temperature drives heating and lighting
demand, wind speed drives supply, and solar radiation drives embedded generation.
One location will not do — GB weather is regionally heterogeneous and a windy
North Sea with a calm south-east is a completely different market state from the
reverse. So we take five population centres plus an offshore point near the main
wind farms.

**An honesty note that belongs in the code, not just the README.** This is the
*archive* API: it returns weather that actually happened. A genuine day-ahead
forecast would use the weather forecast available at gate closure, which is less
accurate. Using outturn weather therefore makes the model look better than a
real deployment would be.

We do this because free historical *forecast* archives are not available for the
whole window, and we handle it explicitly:

- `features/build.py` degrades the weather signal to a daily aggregate rather
  than a half-hourly outturn, which is much closer to the information content of
  a real day-ahead forecast;
- the limitation is stated in the README and in RESUME_CLAIMS.md;
- `--perfect-weather` in the walk-forward run quantifies the gap, so the size of
  the advantage is measured rather than waved away.

    python -m ppa.ingest.openmeteo --start 2019-01-01 --end 2025-06-30
"""

from __future__ import annotations

import argparse
import logging
from functools import partial
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ppa.config import OPENMETEO_ARCHIVE, STUDY, WEATHER_SITES
from ppa.ingest.cache import cached, load_all

log = logging.getLogger(__name__)

SOURCE = "openmeteo"

HOURLY_VARIABLES = [
    "temperature_2m",
    "wind_speed_100m",  # 100m, not 10m: turbine hub height
    "shortwave_radiation",
    "cloud_cover",
]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def _fetch_site(site: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }
    response = requests.get(OPENMETEO_ARCHIVE, params=params, timeout=120)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    frame = pd.DataFrame(
        {"time": pd.to_datetime(hourly["time"], utc=True)}
        | {var: hourly[var] for var in HOURLY_VARIABLES}
    )
    frame["site"] = site
    return frame


def ingest(start: str, end: str, force: bool = False) -> pd.DataFrame:
    """One cached file per site — the API accepts the full range in one request."""
    frames = []
    for site, (lat, lon) in WEATHER_SITES.items():
        key = f"{site}_{start}_{end}"
        frames.append(
            cached(
                SOURCE,
                key,
                partial(_fetch_site, site, lat, lon, start, end),
                force,
            )
        )
    return pd.concat(frames, ignore_index=True)


def load() -> pd.DataFrame:
    return load_all(SOURCE)


def to_national(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-site hourly weather into national hourly aggregates.

    Mean temperature and mean radiation are the demand-relevant summaries. For
    wind we keep the mean *and* the spread: a uniformly windy GB and a GB with
    one windy corner produce very different total output for the same average.
    """
    grouped = frame.groupby("time")

    national = pd.DataFrame(
        {
            "temperature_mean": grouped["temperature_2m"].mean(),
            "temperature_min": grouped["temperature_2m"].min(),
            "temperature_max": grouped["temperature_2m"].max(),
            "wind_speed_mean": grouped["wind_speed_100m"].mean(),
            "wind_speed_std": grouped["wind_speed_100m"].std(),
            "wind_speed_max": grouped["wind_speed_100m"].max(),
            "radiation_mean": grouped["shortwave_radiation"].mean(),
            "cloud_cover_mean": grouped["cloud_cover"].mean(),
        }
    ).reset_index()

    # Heating and cooling degree hours, base 15.5C — the UK convention. The
    # relationship between temperature and demand is V-shaped, and a linear
    # temperature term cannot represent that; these two make it linear again.
    national["heating_degrees"] = (15.5 - national["temperature_mean"]).clip(lower=0)
    national["cooling_degrees"] = (national["temperature_mean"] - 15.5).clip(lower=0)

    return national


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=STUDY.start)
    parser.add_argument("--end", default=STUDY.end)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    frame = ingest(args.start, args.end, force=args.force)
    national = to_national(frame)

    print(f"{len(frame):,} site-hours across {frame['site'].nunique()} sites")
    print(f"{len(national):,} national hours  {national['time'].min()} .. {national['time'].max()}")
    print(
        f"temperature mean {national['temperature_mean'].mean():.1f}C  "
        f"wind mean {national['wind_speed_mean'].mean():.1f} km/h  "
        f"heating degrees mean {national['heating_degrees'].mean():.1f}"
    )


if __name__ == "__main__":
    main()
