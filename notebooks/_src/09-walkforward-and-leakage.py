# %% [markdown]
# # Lesson 09 — Walk-forward validation, and the information set
#
# **You will end with:** the headline forecasting result, and — more importantly — the ability
# to defend it against the question every quant interviewer asks: *"how do you know you aren't
# peeking?"*
#
# This lesson is mostly about one idea: **a feature is not defined by what it computes, but by
# when it was knowable.**

# %%
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from ppa.config import DATA_PROCESSED, REPORTS
from ppa.data import calendar as cal
from ppa.eval import metrics
from ppa.features.build import (
    LAST_PERIOD_BEFORE_GATE,
    PERIOD_LAGS_DAYS,
    decision_time,
    feature_columns,
)
from ppa.models import naive
from ppa.models.walkforward import make_folds

features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
print(f"{len(features):,} rows x {len(feature_columns(features))} features")

# %% [markdown]
# ## 1. The information set
#
# The GB day-ahead auction clears around 11:00 local on day D for delivery across all of D+1.
# So to forecast day **T**, the decision instant is **11:00 local on T-1**.
#
# At that moment you know:
#
# | Known | Not known |
# |---|---|
# | Prices and demand through SP22 of T-1 (SP22 ends 11:00) | SP23 onwards of T-1 |
# | Every complete day up to T-2 | Anything on T itself |
# | The calendar, forever | — |
# | A weather *forecast* for T | The weather *outturn* for T |

# %%
gate = decision_time(pd.Series(["2024-06-15"])).iloc[0]
print(f"target day      2024-06-15")
print(f"decision time   {gate}  ({gate.tz_convert(cal.TZ)} local)")
print()
for period in (22, 23):
    stamp = cal.to_utc("2024-06-14", period)
    verdict = "KNOWN" if stamp < gate else "NOT YET HAPPENED"
    print(f"  SP{period} on 2024-06-14 starts {stamp}  ->  {verdict}")

# %% [markdown]
# ## 2. The feature everybody uses, and why we cannot
#
# `price_lag_1d` — "the price at this same settlement period yesterday" — is the first feature
# anyone builds for day-ahead price forecasting. It is also **not available at gate closure**
# for most of the day.
#
# For a target at 14:30 on T, the same period on T-1 is 14:30 on T-1: three and a half hours
# *after* the 11:00 decision. It has not happened yet.

# %%
rows = []
for period in [1, 10, 22, 23, 30, 48]:
    target = "2024-06-15"
    source = cal.to_utc("2024-06-14", period)  # same period, one day earlier
    gate = decision_time(pd.Series([target])).iloc[0]
    rows.append({
        "settlement_period": period,
        "lag_1d_source": source,
        "decision_time": gate,
        "hours_after_gate": (source - gate).total_seconds() / 3600,
        "usable": source < gate,
    })
pd.DataFrame(rows)

# %% [markdown]
# Only periods 1–22 would be legitimate, and those are exactly the ones already captured by
# the morning aggregates. So `PERIOD_LAGS_DAYS` starts at **2**, and the absence of a 1-day
# lag is asserted in `tests/test_no_leakage.py`.

# %%
print(f"period lags used: {PERIOD_LAGS_DAYS}")
print(f"'price_lag_1d' in features: {'price_lag_1d' in features.columns}")
print(f"last period before gate: SP{LAST_PERIOD_BEFORE_GATE}")

# %% [markdown]
# ### How much does that cost us?
#
# Being honest is not free. Let us measure what the banned feature would have been worth, by
# constructing it and checking its correlation with the target against the ones we do use.

# %%
panel = pd.read_parquet(DATA_PROCESSED / "panel.parquet")
banned = panel[["settlement_date", "settlement_period", "price"]].copy()
banned["settlement_date"] = (
    pd.to_datetime(banned["settlement_date"]) + pd.Timedelta(days=1)
).dt.date.astype(str)
banned = banned.rename(columns={"price": "price_lag_1d_BANNED"})

merged = features.merge(banned, on=["settlement_date", "settlement_period"], how="left")

comparison = merged[
    ["price", "price_lag_1d_BANNED", "price_lag_2d", "price_lag_7d", "price_am_mean"]
].dropna().corr()["price"].drop("price")
comparison.sort_values(ascending=False).to_frame("correlation with target")

# %% [markdown]
# The banned lag is the most correlated feature available — which is precisely why it is
# tempting and precisely why using it would inflate the result. `price_am_mean` (T-1's morning,
# which *is* known) recovers a good deal of that signal legitimately.
#
# This is the trade the project makes explicitly: a slightly worse number that is actually
# achievable, rather than a better one that is not.

# %% [markdown]
# ## 3. Walk-forward folds
#
# A random train/test split would let the model learn from July to predict June. On a market
# with strong regime persistence that produces a spectacular, meaningless score.
#
# Instead: an expanding window, refit every 30 days, always predicting forward.

# %%
folds = make_folds(pd.to_datetime(features["settlement_date"]))
fold_table = pd.DataFrame([
    {"fold": f.index, "train_end": f.train_end.date(), "test_start": f.test_start.date(),
     "test_end": f.test_end.date(), "train_days": f.n_train}
    for f in folds
])
print(f"{len(folds)} folds")
fold_table.head(4)

# %%
fig, ax = plt.subplots(figsize=(11, 6))
for f in folds[::4]:
    ax.barh(f.index, (f.train_end - pd.Timestamp(features["settlement_date"].min())).days,
            left=0, color="#4C72B0", height=3)
    ax.barh(f.index, (f.test_end - f.test_start).days + 1,
            left=(f.test_start - pd.Timestamp(features["settlement_date"].min())).days,
            color="crimson", height=3)
ax.set(xlabel="days from start of data", ylabel="fold",
       title="Expanding window (blue = train, red = test)")
plt.show()

# %% [markdown]
# **Expanding, not rolling.** GB power had a structural break in 2021–22, but it also has
# stable seasonal structure. Discarding 2019 would throw away the only examples of a calm
# market the model ever sees — exactly what it needs when prices normalise again in 2024.
#
# **Refit every 30 days, not daily.** Daily refitting means ~1,600 fits for a difference well
# inside the noise. Thirty days is a stated compromise, not a tuned one.

# %% [markdown]
# ## 4. The baseline, declared in advance
#
# "Beats a seasonal naive baseline" is only meaningful if you say which one *before* you look
# at results. `models/naive.py` names it in code.

# %%
print(f"headline baseline: {naive.HEADLINE_BASELINE}")
print(f"all variants scored: {list(naive.BASELINES)}")

# %% [markdown]
# ## 5. The results

# %%
report = json.loads((REPORTS / "forecast_metrics.json").read_text())

table = pd.DataFrame(report["models"]).T[["mae", "rmse", "smape", "bias", "mae_improvement_vs_naive"]]
table = table.sort_values("mae")
table.round(3)

# %%
improvement = report["mae_improvement_vs_naive"]
dm = report["diebold_mariano_xgb_vs_naive"]

print(f"out-of-sample half-hours : {report['n_predictions']:,}")
print(f"period                   : {report['period']['start'][:10]} .. {report['period']['end'][:10]}")
print(f"MAE improvement vs naive : {improvement:.1%}")
print(f"Diebold-Mariano          : {dm['statistic']:.2f}, p = {dm['p_value']:.2e}")

# %% [markdown]
# ### Is it significant, or is it luck?
#
# The **Diebold-Mariano** test is the standard tool for comparing forecast accuracy. Its
# statistic is the mean loss differential divided by its standard error — but the standard
# error must account for the fact that half-hourly forecast errors are strongly
# autocorrelated. Using a plain i.i.d. variance would treat 77,000 correlated errors as 77,000
# independent observations and overstate significance dramatically.
#
# `eval/metrics.py` uses a Newey-West variance with a rule-of-thumb bandwidth. Even so the
# statistic is enormous — this is not a marginal result.
#
# Note also the third baseline, `last_known_morning_shape`, which is a genuinely competitive
# construction and is reported even though it does not flatter the story.

# %% [markdown]
# ## 6. Where does the model actually help?

# %%
predictions = pd.read_parquet(DATA_PROCESSED / "walkforward_predictions.parquet")

fig, ax = plt.subplots(figsize=(11, 5))
for column, label in [("pred_xgb", "XGBoost"),
                      (f"pred_{naive.HEADLINE_BASELINE}", "seasonal naive")]:
    per_period = metrics.by_period_of_day(predictions, "price", column)
    ax.plot(per_period["settlement_period"], per_period["mae"], lw=2, label=label)
ax.set(xlabel="settlement period (1 = 00:00 local)", ylabel="MAE (GBP/MWh)",
       title="Error through the day")
ax.legend()
plt.show()

# %% [markdown]
# Both models are worst in the evening peak (roughly periods 33–40, 16:00–20:00) — when the
# marginal generator changes fastest and scarcity pricing kicks in. That is where the money
# is and where forecasting is hardest, which is not a coincidence.

# %%
monthly = predictions.copy()
monthly["month"] = pd.to_datetime(monthly["start_time"]).dt.to_period("M").astype(str)
rows = [
    {"month": m,
     "xgb": metrics.mae(g["price"], g["pred_xgb"]),
     "naive": metrics.mae(g["price"], g[f"pred_{naive.HEADLINE_BASELINE}"])}
    for m, g in monthly.groupby("month")
]
table = pd.DataFrame(rows)
table["improvement"] = 1 - table["xgb"] / table["naive"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax1.plot(table["month"], table["naive"], lw=1.5, label="seasonal naive")
ax1.plot(table["month"], table["xgb"], lw=2, label="XGBoost")
ax1.set(ylabel="MAE (GBP/MWh)", title="Monthly out-of-sample error")
ax1.legend()

ax2.bar(table["month"], table["improvement"], color="#4C72B0")
ax2.axhline(improvement, color="crimson", ls="--", label=f"overall {improvement:.1%}")
ax2.set(ylabel="MAE improvement", xlabel="")
ax2.set_xticks(table["month"][::6])
ax2.tick_params(axis="x", rotation=45)
ax2.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The improvement is positive in essentially every month, including through the gas crisis
# when absolute errors were far larger. A model that only beat the baseline during the crisis
# would be a coincidence, not a result.

# %% [markdown]
# ## 7. The weather caveat, measured
#
# Weather features come from the Open-Meteo **archive** — what the weather actually did — not
# the day-ahead forecast that would have existed at gate closure. That makes the headline
# number optimistic by some amount.
#
# Rather than disclaim it, `make forecast-ablation` re-runs the entire walk-forward with every
# weather feature removed. The truth for a real deployment sits between the two: closer to the
# with-weather figure for temperature (day-ahead temperature forecasts are accurate) and
# closer to the without-weather figure for wind (which is much harder).

# %%
ablation_path = REPORTS / "forecast_metrics_no_weather.json"
if ablation_path.exists():
    ablation = json.loads(ablation_path.read_text())
    print(f"with archive weather : MAE {report['models']['xgb']['mae']:.3f}  "
          f"({report['mae_improvement_vs_naive']:.1%} vs naive)")
    print(f"no weather features  : MAE {ablation['models']['xgb']['mae']:.3f}  "
          f"({ablation['mae_improvement_vs_naive']:.1%} vs naive)")
else:
    print("run `make forecast-ablation` to produce this comparison")

# %% [markdown]
# ## 8. What the tests assert
#
# Everything above is a demonstration. `tests/test_no_leakage.py` turns it into an invariant
# that runs on every commit:
#
# - every target is strictly after its own decision time;
# - every period lag resolves to an instant before gate closure (checked per lag, per row);
# - morning aggregates stop at SP22, verified against a hand-recomputed value from the panel;
# - rolling windows end at T-2, also hand-recomputed;
# - no datetime column can reach the model;
# - no feature correlates above 0.97 with the target;
# - folds never train on the future and never overlap.
#
# That list is the answer to "how do you know you aren't peeking?"

# %% [markdown]
# ## 9. Exercises
#
# 1. Add `price_lag_1d_BANNED` from section 2 to the feature set and re-run one fold. How much
#    does MAE improve? That number is the size of the lie you would be telling.
# 2. Change the walk-forward to a **rolling** 2-year window and re-run. Does performance
#    improve during the crisis and degrade afterwards, as the expanding-window argument
#    predicts?
# 3. Refit every 7 days instead of 30. Measure the improvement and the wall-clock cost, then
#    decide whether you would ship it.
#
# ## 10. Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 09: walk-forward, information set, leakage tests"
# ```
#
# ---
#
# **Next:** `11-strategy.ipynb` — turning a forecast into money, and finding out that the first
# design was worth nothing.
