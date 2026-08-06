# Power Price Alpha

Half-hourly GB electricity price forecasting, and an honest backtest of a battery-arbitrage
strategy built on it.

**Measured over 77,109 out-of-sample half-hours (2021-01 → 2025-06), 54 expanding-window folds:**

| | MAE | XGBoost improvement |
|---|---|---|
| **XGBoost** | **27.23** | — |
| `same_period_two_days_ago` (strongest baseline) | 34.95 | **−22.1%** |
| `same_period_last_week` (declared baseline) | 38.80 | −29.8% |

Diebold-Mariano statistic −28.35 (p ≈ 7.6e-177, Newey-West).

**Quote 22%, not 30%.** The declared baseline is the literature-standard one and was fixed in
code before any model was fitted — but it is not the *hardest* baseline on this data. Against
the strongest one the improvement is 22.1%, and re-running with every weather feature removed
gives 22.1% as well. Two independent stress tests landing on the same number is the reason to
trust it.

A 1 MW / 2 MWh battery scheduled from those forecasts, settled against realised prices with
frictions on both legs, earns **130,603 GBP** over the period against **85,104** for the same
battery on the naive forecast — a **+53.5% uplift**, capturing **60.5%** of what perfect
foresight would have made.

## The parts that actually matter

Forecasting half-hourly power prices with XGBoost is not the interesting bit. These are:

- **The settlement-period calendar.** A GB day has 48 half-hourly settlement periods — except
  the two clock-change days a year, which have 46 and 50. Elexon reports UTC timestamps;
  NESO reports settlement date + period. Joining them naively corrupts those days and every
  lag feature that crosses them, silently. `src/ppa/data/calendar.py` handles it; 32 tests
  pin it across all 13 transitions in the window.

- **An honest information set.** Day-ahead means the forecast for every period of day T is
  made at gate closure on T-1. So `price_lag_1d` — "the price at this period yesterday" — is
  **not available**: for a 14:30 target it refers to 14:30 on T-1, three and a half hours
  after the decision. It is deliberately absent, and 18 tests assert that no feature's source
  timestamp reaches past its decision time.

- **A strategy you could actually trade.** The first design took a position on
  forecast-minus-naive and settled against realised-minus-naive. It backtested at a Sharpe of
  50 — which is the tell, because no instrument pays "price minus last week's price". It was
  replaced by the battery, where every leg is a real trade at a real price. Both versions are
  in the git history.

- **Saying what the number isn't.** The battery's Sharpe of 10 is *physical arbitrage*, not
  forecasting skill: the naive schedule alone scores 8.09, because the intraday spread is
  almost always positive. The forecast's contribution is the uplift, and the report says so.

## Quickstart

```bash
make venv
make all
```

The first run downloads ~6.5 years of half-hourly data from three keyless public APIs and
caches it to parquet (~350 requests, a few minutes). Subsequent runs are offline. It ends by
printing `reports/metrics.json` and writing `reports/report.html`.

The pipeline is incremental — re-running skips any stage whose inputs have not changed, which
matters because the walk-forward takes about ten minutes. `make rebuild` forces the lot.

```bash
open reports/report.html
```

## Data sources

All free, keyless, verified working:

| Source | What |
|---|---|
| [Elexon BMRS Insights](https://data.elexon.co.uk/bmrs/api/v1) | Half-hourly market index price and volume (`APXMIDP`) |
| [NESO Data Portal](https://api.neso.energy) | Half-hourly demand, embedded wind/solar, interconnector flows |
| [Open-Meteo Archive](https://open-meteo.com) | Hourly temperature, wind@100m, radiation, cloud at 6 GB sites |

Two things learned from the responses rather than the docs, both recorded in code: Elexon
rejects any range longer than 7 days, and `N2EXMIDP` returns **zero** prices before ~2020 —
training on it would teach the model that electricity was once free.

## Known limitation: weather

Features use the Open-Meteo **archive** (what the weather actually did), degraded to daily
aggregates. A real deployment would use the day-ahead *forecast*, which is less accurate — so
the headline number is optimistic by some amount.

Rather than wave that away, `make forecast-ablation` re-runs the entire walk-forward with
every weather feature removed. The result: **22.1%** improvement with no weather at all
(MAE 30.22, DM −22.90). So the archive advantage is worth at most 7.7 percentage points, and
a real deployment lands between 22.1% and 29.8% — nearer the top, since day-ahead temperature
forecasts are accurate and it is wind that is hard.

## What's where

| Path | What it is |
|---|---|
| `src/ppa/ingest/` | API clients with retry + parquet caching |
| `src/ppa/data/` | Settlement-period calendar, panel assembly, quality checks |
| `src/ppa/features/` | Lags, morning aggregates, residual load, Fourier terms |
| `src/ppa/models/` | Seasonal naive, OLS/SARIMAX, XGBoost, walk-forward harness |
| `src/ppa/eval/` | MAE/RMSE/sMAPE, Diebold-Mariano, block bootstrap |
| `src/ppa/strategy/` | Battery scheduling, backtest with frictions, risk |
| `src/ppa/report/` | Writes `reports/metrics.json` and a self-contained `report.html` |
| `RESUME_CLAIMS.md` | Every claim → the command that proves it |

## The course

Start at `course/00-gb-power-market.md` — settlement periods, the day-ahead auction, which
price series this forecasts and why, and the information set at gate closure. You cannot
model this market sensibly without them.
