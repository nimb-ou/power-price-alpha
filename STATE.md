# STATE — Power Price Alpha

Living status file. A new session reads this, does the next session, updates this, commits.
Keep under 100 lines.

## Current position
Session 0 complete. Session 9 (repo B's first build session) is next.

## Sessions
- [x] **0 — Scaffold.** Tree, git, requirements, Makefile, config, CI skeleton, lesson 00.
- [ ] **9 — Ingest.** Lessons 01–02: Elexon MID, NESO demand, Open-Meteo, retry + parquet cache.
- [ ] **10 — Panel.** Lessons 03–04: settlement-period calendar, assembly, quality tests.
- [ ] **11 — EDA + baselines.** Lessons 05–06: seasonality, spikes, regimes; seasonal naive variants.
- [ ] **12 — Statsmodels.** Lesson 07: OLS + Fourier, SARIMAX, residual diagnostics.
- [ ] **13 — ML forecaster.** Lessons 08–09: features, XGBoost, walk-forward, leakage tests.
- [ ] **14 — Scoring.** Lesson 10: **measures the 18% MAE-vs-naive claim**, per regime and per period.
- [ ] **15 — Strategy.** Lessons 11–12: signals, event-driven backtester with costs and slippage.
- [ ] **16 — Ship.** Lessons 13–14: risk report, HTML report, CI, RESUME_CLAIMS filled in.

## Built and working
- `src/ppa/config.py` — paths, market constants (settlement periods, gate closure, timezone),
  data-source endpoints, six weather sites, `StudyConfig` (window, walk-forward, frictions,
  regime dates).
- Makefile targets: `venv ingest panel features forecast backtest report test lint all clean clean-cache`.
- `.github/workflows/ci.yml` — quality job; pipeline job runs end-to-end on a committed fixture window.

## Known open items
- `tests/fixtures/` is empty. Session 9 must commit a short real window (2024-01-01 →
  2024-03-31) so CI can run the pipeline without network.
- `reports/metrics.json` does not exist yet; every entry in RESUME_CLAIMS.md is UNVERIFIED.
- No GitHub remote yet.

## Decisions made (don't relitigate)
- **`APXMIDP`, not `N2EXMIDP`** — verified live: N2EX returns zero prices before ~2020, which
  would silently poison the early training window.
- **Study window 2019-01-01 → 2025-06-30** — deliberately spans calm, crisis and
  normalisation so the regime split is meaningful rather than decorative.
- **Gate closure is a single constant** (`DAY_AHEAD_GATE_CLOSURE_LOCAL`) imported everywhere.
  Two modules disagreeing about the information set is the classic silent backtest bug.
- **Network tests are marked and excluded from CI.** CI must not fail because Elexon is down.
- Raw API responses are cached to parquet under `data/raw/` and gitignored.
