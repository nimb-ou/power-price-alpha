# STATE — Power Price Alpha

Living status file. A new session reads this, does the next session, updates this, commits.

## Current position
Sessions 0, 9-15 complete. Remaining: session 16 (course notebooks 01-13, CI fixtures).

## Sessions
- [x] **0 — Scaffold.** Tree, git, requirements, Makefile, config, CI skeleton, lesson 00.
- [x] **9 — Ingest.** Elexon MID, NESO demand, Open-Meteo, retry + parquet cache.
- [x] **10 — Panel.** Settlement-period calendar, assembly, quality tests.
- [x] **11-12 — Baselines + statsmodels.** naive.py (3 variants), sarimax.py (OLS + SARIMAX).
- [x] **13 — ML forecaster.** Features, XGBoost, walk-forward, 18 leakage tests.
- [x] **14 — Scoring.** 29.8% MAE improvement measured, Diebold-Mariano, weather ablation.
- [x] **15 — Strategy.** Battery arbitrage, frictions, risk report.
- [ ] **16 — Ship.** Course notebooks 01-13, committed CI fixtures, README polish.

## Measured results (reports/metrics.json)
- Forecast: XGBoost MAE **27.230** vs seasonal-naive **38.800** = **29.8% improvement**,
  over 77,109 out-of-sample half-hours in 54 folds. DM statistic -28.35, p = 7.6e-177.
- Strategy: 1 MW / 2 MWh battery. xgb **130,603 GBP** (Sharpe 10.01, CI [8.36, 13.04]),
  naive 85,104, oracle 215,828. Uplift over naive **+53.5%**; captures 60.5% of oracle.

## Known open items
- Course notebooks 01-13 not yet written (lesson 00 is done).
- `tests/fixtures/` is empty — CI's pipeline job needs a committed short window so it can
  run end to end without hitting Elexon/NESO.
- No GitHub remote yet.

## Decisions made (don't relitigate)
- **`APXMIDP`, not `N2EXMIDP`** — the latter returns zeros before ~2020.
- **Elexon caps ranges at 7 days**, documented only in its 400 response body.
- **`price_lag_1d` is banned.** The same period on T-1 is up to 13 hours after gate closure.
  Its absence is asserted in `tests/test_no_leakage.py`.
- **Headline baseline declared before fitting** (`naive.HEADLINE_BASELINE`).
- **Battery arbitrage, not a synthetic spread.** The first design traded "forecast minus
  seasonal naive" against "realised minus seasonal naive" and scored Sharpe 50 — there is
  no instrument with that payoff. Do not reintroduce it.
- **Daily annualisation, sqrt(365.25).** The battery cycles once a day; treating its four
  charge periods as independent bets inflated Sharpe sevenfold.
- **Weather is archive outturn, not a forecast.** Stated everywhere and bracketed by
  `make forecast-ablation`.
