# Power Price Alpha

Half-hourly GB electricity price forecasting, and an honest backtest of the trading strategy
that falls out of it.

The forecasting part is straightforward: lagged prices, national demand net of embedded wind
and solar, weather across six GB sites, and calendar seasonality, fed to XGBoost, scored
under walk-forward validation against a seasonal-naive baseline that is much harder to beat
than it looks.

The parts that actually matter, and that most versions of this project get wrong:

- **The settlement-period calendar.** A GB day has 48 half-hourly settlement periods — except
  the two clock-change days each year, which have 46 and 50. Elexon reports in UTC
  timestamps; NESO reports in settlement date + period. Joining them naively corrupts those
  days and every lag feature that crosses them. `src/ppa/data/calendar.py` handles it and
  `tests/test_calendar.py` asserts it.

- **An honest information set.** Day-ahead means every price for delivery on D+1 is forecast
  at gate closure on D. So lagged prices stop at D, and weather must be the *forecast*
  available then, not the outturn we can see now. `tests/test_no_leakage.py` asserts that no
  feature's source timestamp exceeds its decision time. This is the first question a
  quant interviewer asks, and the repo answers it in code rather than in prose.

- **Frictions.** The backtest charges transaction costs and slippage on every position
  change, and reports Sharpe, hit rate and maximum drawdown split into calm and volatile
  regimes — because a strategy that only works during the 2021–22 gas crisis is not a
  strategy, it is a coincidence.

## Quickstart

```bash
make venv
make all
```

The first `make all` downloads roughly six years of half-hourly data from three public APIs
(no keys required) and caches it under `data/raw/` as parquet; subsequent runs are offline
and fast. It ends by printing `reports/metrics.json` and writing `reports/report.html`.

## Data sources

All free, all keyless, all verified working:

| Source | What |
|---|---|
| [Elexon BMRS Insights](https://data.elexon.co.uk/bmrs/api/v1) | Half-hourly market index price and volume (`APXMIDP`) |
| [NESO Data Portal](https://api.neso.energy) | Half-hourly national demand, embedded wind/solar generation, interconnector flows |
| [Open-Meteo Archive](https://open-meteo.com) | Hourly historical temperature, wind speed, radiation at six GB sites |

## What's where

| Path | What it is |
|---|---|
| `src/ppa/ingest/` | API clients with retry + parquet caching |
| `src/ppa/data/` | Settlement-period calendar, panel assembly, quality checks |
| `src/ppa/features/` | Lags, rolling stats, residual load, Fourier seasonality |
| `src/ppa/models/` | Seasonal naive, SARIMAX, XGBoost, walk-forward harness |
| `src/ppa/eval/` | Error metrics, regime splits, diagnostics |
| `src/ppa/strategy/` | Signal construction, event-driven backtester, risk |
| `src/ppa/report/` | Writes `reports/metrics.json` and `reports/report.html` |
| `notebooks/` | The course — executable lessons, in order |
| `RESUME_CLAIMS.md` | Every claim I make about this project → the command that proves it |

## The course

Built as a taught course. Start at `course/00-gb-power-market.md` — you cannot model this
market without understanding settlement periods, the day-ahead auction and imbalance pricing
— then work through the numbered notebooks.

## Honesty note

The headline forecast-improvement figure is whatever `make all` measures on your machine, read
out of `reports/metrics.json`. If it isn't 18%, the docs get changed, not the pipeline.
