# STATE — Power Price Alpha

Living status file. A new session reads this, does the next session, updates this, commits.

## Current position
Complete. Pushed to https://github.com/nimb-ou/power-price-alpha (private).

## Sessions
- [x] **0 — Scaffold.** Tree, git, requirements, Makefile, config, CI skeleton, lesson 00.
- [x] **9 — Ingest.** Elexon MID, NESO demand, Open-Meteo, retry + parquet cache.
- [x] **10 — Panel.** Settlement-period calendar, assembly, quality tests.
- [x] **11-12 — Baselines + statsmodels.** naive.py (3 variants), sarimax.py (OLS + SARIMAX).
- [x] **13 — ML forecaster.** Features, XGBoost, walk-forward, 18 leakage tests.
- [x] **14 — Scoring.** 29.9% MAE improvement measured, Diebold-Mariano, weather ablation.
- [x] **15 — Strategy.** Battery arbitrage, frictions, risk report.
- [x] **16 — Ship.** Notebooks 01-13, lesson 14, CI fixtures, README, pushed.

## Measured results (reports/metrics.json)
- Forecast: XGBoost MAE **27.2** vs seasonal-naive **38.8** over 77,109 out-of-sample
  half-hours in 54 folds. DM statistic -28.35, p ~ 7.6e-177.
- **Quote 22%, not 30%.** Three honest comparisons:
  vs declared baseline 29.9% · vs strongest baseline 22.1% · with weather removed 22.1%.
  Two independent stress tests converge on 22%.
- Strategy: 1 MW / 2 MWh battery. xgb **130,909 GBP** (Sharpe 10.04, CI [8.39, 13.08]),
  naive 85,104, oracle 215,828. Uplift over naive **+53.8%**; captures ~60% of oracle.

## Known open items
- Repo is private; make it public when ready to share.

## Reproducibility note
XGBoost is seeded but not bit-identical across runs. A full re-run moved the headline from
29.8% to 29.9% and the Sharpe from 10.01 to 10.04 — well inside the confidence intervals, and
the reason RESUME_CLAIMS quotes 22% rather than a three-significant-figure number.

## Decisions made (don't relitigate)
- **`APXMIDP`, not `N2EXMIDP`** — the latter returns zeros before ~2020.
- **Elexon caps ranges at 7 days**, documented only in its 400 response body.
- **`price_lag_1d` is banned.** The same period on T-1 is up to 13 hours after gate closure.
  Its absence is asserted in `tests/test_no_leakage.py`.
- **Headline baseline declared before fitting** (`naive.HEADLINE_BASELINE`) — and it turned
  out not to be the hardest, so claims are quoted against the strongest instead. Do not swap
  the constant; that is what it exists to prevent.
- **Battery arbitrage, not a synthetic spread.** The first design traded "forecast minus
  seasonal naive" against "realised minus seasonal naive" and scored Sharpe 50 — there is
  no instrument with that payoff. Do not reintroduce it.
- **Daily annualisation, sqrt(365.25).** The battery cycles once a day; treating its four
  charge periods as independent bets inflated Sharpe sevenfold.
- **Weather is archive outturn, not a forecast.** Bracketed by `make forecast-ablation`.
- **CI fixtures are real data spanning the October clock change**, so every push exercises a
  50-period day. Synthetic data would only contain structure we already thought of.
