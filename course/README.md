# The course

This repository was built as a taught course, not published as finished code. Every lesson
states a problem, derives an approach, writes the code into `src/`, runs it against real GB
market data, shows the output, and ends at a commit.

It is roughly **25 hours** end to end, or about **8 hours** reading for understanding without
the exercises. It is longer than the credit course because the first four lessons are spent on
data that arrives wrong.

## Before you start

**Assumed.** Python and pandas, and comfort with time-series basics — lags, seasonality, the
idea that a random train/test split is illegal here.

**Not assumed.** Anything about electricity markets. Lesson 00 covers settlement periods, the
day-ahead auction and the merit order from scratch, and you should read it even if you plan to
skim everything else, because the rest of the course is unreadable without it.

**Setup.**

```bash
make venv
make all      # first run ~35 minutes; downloads 6.5 years and runs 54 folds
```

No API keys. Three public sources — Elexon, NESO, Open-Meteo — all cached to parquet, so the
first run is slow and every run after it is offline and fast.

In a hurry, or offline:

```bash
make fixtures    # loads a committed 6-month sample instead of the full cache
```

The fixture deliberately spans a clock-change day, because that is where the joins break.

## The order

Read `00` first, and do not skip it.

| # | Format | Lesson | You end with |
|---|---|---|---|
| 00 | md | [How the GB power market works](00-gb-power-market.md) | Settlement periods, gate closure, merit order, what price you are even forecasting |
| 01 | nb | `01-ingest-prices.ipynb` | Elexon market index prices, with retries and a parquet cache |
| 02 | nb | `02-ingest-demand-and-weather.ipynb` | NESO demand and Open-Meteo weather, joined on the right keys |
| 03 | nb | `03-settlement-calendar.ipynb` | Days with 46, 48 and 50 periods handled correctly |
| 04 | nb | `04-panel-assembly.ipynb` | One half-hourly panel, with the quality checks that caught the joins |
| 05 | nb | `05-eda.ipynb` | Intraday and weekly shape, negative prices, spikes, the 2021 regime break |
| 06 | nb | `06-baselines.ipynb` | Three seasonal naive baselines, and a declaration of which one you will be judged against |
| 07 | nb | `07-statsmodels.ipynb` | OLS with Fourier terms and SARIMAX, plus residual diagnostics |
| 08 | nb | `08-features.ipynb` | Lags, rolling stats, residual load, interconnectors — every one timestamped by when it was knowable |
| 09 | nb | `09-walkforward-and-leakage.ipynb` | An expanding-window harness and tests that assert no feature can see the future |
| 10 | nb | `10-honest-scoring.ipynb` | The headline improvement, measured against the *strongest* baseline rather than the flattering one |
| 11 | nb | `11-strategy.ipynb` | A battery arbitrage schedule — and the discarded design that scored a Sharpe of 50 |
| 12 | nb | `12-backtest-engine.ipynb` | Transaction costs, slippage, round-trip efficiency, no look-ahead |
| 13 | nb | `13-risk-and-reporting.ipynb` | Sharpe, hit rate, max drawdown, block-bootstrap intervals, per-regime splits |
| 14 | nb | `14-forecast-uncertainty.ipynb` | Quantile forecasts, why their coverage is a lie, and a conformal fix |
| 15 | md | [Reproducibility](15-reproducibility.md) | `make all`, CI against a committed fixture, publishing the report |

## If you only read four

- **03** — the settlement calendar. Naively joining Elexon's UTC timestamps to NESO's
  settlement-date-and-period silently corrupts two days a year and every lag that crosses them.
  This is the lesson that separates the project from a Kaggle notebook.
- **09** — walk-forward validation and the information set. "Would this feature have been
  knowable at gate closure?" is the single question a quant interviewer asks, and it is
  answered here in tests rather than in prose.
- **11** — the strategy that was thrown away. It reported a Sharpe of 50 by trading a spread no
  instrument pays. Working out why is more useful than the strategy that replaced it.
- **14** — the interval that does not cover what it claims, the diagnosis that separates a
  broken estimator from a moving market, and the conformal repair.

## Exercises

Every lesson ends with exercises. Worked solutions are in [`solutions.md`](solutions.md).

Several exercises ask you to measure something before deciding whether to keep it, and then to
write down in advance what result would make you throw it away. That habit is most of what
separates a backtest from a story, and it is the thing this course is really trying to teach.

## How the notebooks are built

Lesson sources live in `notebooks/_src/*.py` in percent format; the `.ipynb` files are generated
with no stored outputs. Edit the `.py`, then:

```bash
make notebooks
```

`tests/test_course.py` enforces that sources and notebooks stay in sync and that every
cross-reference resolves. `make notebooks-check` executes every lesson end to end.

## A note on the numbers

Every figure in the lessons is read from `reports/metrics.json`, regenerated by `make all`.
Where a number is uncomfortable it is still there:

- The headline improvement is quoted at **22%** against the strongest baseline, not the **30%**
  the declared baseline would allow.
- Weather features are built from Open-Meteo's *archive*, which is outturn rather than the
  forecast available at gate closure. That is optimistic by an unknown amount, so the whole
  walk-forward is re-run with weather removed to bracket it.
- The battery strategy's Sharpe of 10 is physical arbitrage, not skill. The naive schedule
  alone scores 8. The number worth quoting is the **uplift over naive**, and the report leads
  with that.
