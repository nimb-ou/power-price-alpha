# Lesson 14 — Reproducibility, CI, and the interview demo

**You will end with:** a pipeline that regenerates every claim from scratch, a CI job that
re-measures them on every push, and a demo you can run in front of someone.

---

## 1. One command

```bash
make all
```

Cold, on a fresh machine: `venv` → `ingest` (~350 requests, a few minutes) → `panel` →
`features` → `forecast` (54 folds, ~10 min) → `forecast-ablation` → `backtest` → `report` →
`test`.

Warm, it is offline and fast — the raw API responses are cached to parquet.

The final output is `reports/metrics.json` and `reports/report.html`. Every number in
`RESUME_CLAIMS.md` is read out of the first. Nothing is typed by hand.

## 2. What is deterministic and what is not

| Component | Deterministic? |
|---|---|
| Ingest | Yes — cached parquet. `--force` to refetch. |
| Calendar, panel, features | Yes, exactly. |
| XGBoost | Seeded, but **not bit-identical** across runs. |
| Bootstrap intervals | Yes — seeded RNG. |

That third row is worth knowing. Re-running the full pipeline moved the headline from 29.8% to
**29.9%**, and the Sharpe from 10.01 to 10.04. Histogram binning and thread scheduling
introduce small variation that a seed does not remove.

The variation is far smaller than the confidence intervals, which is the relevant comparison.
It is also why `RESUME_CLAIMS.md` quotes **22%** rather than a three-significant-figure number:
a figure that moves in the second decimal between runs should not be reported as though it
were exact.

## 3. CI: two jobs

`.github/workflows/ci.yml`.

**quality** — `ruff check`, `mypy`, `pytest -m "not network"`. Network-marked tests are
excluded so a red build always means *our* code broke, never that Elexon had a bad morning.

**pipeline** — runs the entire chain end to end and uploads `reports/` as an artifact.

### The fixture

The pipeline job cannot hit the APIs, so `tests/fixtures/raw/` holds **six months of real
data** (July–December 2024, 0.7 MB) committed to the repo. CI copies it into `data/raw/` and
runs a scaled-down but genuine walk-forward:

```yaml
make fixtures
make report START=2024-07-01 END=2024-12-31 TRAIN_DAYS=90 REFIT_DAYS=30
```

Two deliberate choices:

- **Real data, not synthetic.** The point of the pipeline job is to catch things like a
  clock-change misalignment. Generated data would only have whatever structure we thought to
  put in it.
- **The window spans the October transition.** Every push therefore exercises a 50-period day.
  That is the highest-value thing CI can check on this project.

Regenerate the fixture after a full ingest with:

```bash
python tools/make_fixtures.py
```

It refuses to write a window with no clock change in it.

## 4. What CI actually proves

On every push, in a clean Ubuntu container with no local state:

- the calendar arithmetic is right on a real 50-period day;
- the panel matches the expected settlement grid exactly;
- no feature reaches past its decision time (18 leakage assertions);
- the backtest mechanics match hand-computed cashflows;
- the whole chain runs from raw data to a report.

That is a much stronger statement than a number in a README, because it keeps being true.

## 5. The interview demo

```bash
make all          # or just `make report` if the cache is warm
open reports/report.html
```

Walk it in this order:

1. **The forecast table.** Point at the three baselines and say which one was declared in
   advance — then say that it is not the hardest one, and that you therefore quote 22% rather
   than 30%. That is the whole project in thirty seconds.
2. **The weather ablation.** "The one thing I could be accused of is using outturn weather. So
   I removed it entirely and got 22.1% — which happens to be the same as the strongest-baseline
   figure."
3. **MAE by settlement period.** Both models are worst in the evening peak, which is where the
   money is and where the physics is hardest.
4. **The battery table.** Lead with the *uplift over naive*, not the Sharpe. Then explain why
   the Sharpe is 10 and why that is not impressive on its own.
5. **The limitations box** at the bottom of the report. Read it out. It is the most credible
   part of the page.

If they want code, open `src/ppa/data/calendar.py` and `tests/test_calendar.py`. That pair
answers "how careful are you actually?" better than any model file.

## 6. Exercises

1. Add a CI step that fails if `mae_improvement_vs_naive` drops more than 3 percentage points
   below a recorded baseline. Then decide whether you would want it blocking merges, given the
   run-to-run variation in section 2.
2. Cache the pip install and the fixture-window panel between CI runs. How much wall-clock does
   it save, and what is the risk of a stale cache key?
3. Pin `xgboost` to an exact build and re-run twice. Does the 0.1pp variation disappear?

## 7. Final commit

```bash
git add -A
git commit -m "Lesson 14: reproducibility, CI fixtures, demo script"
```

---

That is the course. Before an interview, reread `RESUME_CLAIMS.md` — particularly the last
table, which is the list of questions you will actually be asked.
