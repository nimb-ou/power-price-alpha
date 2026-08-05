# %% [markdown]
# # Lesson 08 — Features, and the information set made concrete
#
# **You will end with:** 44 features, every one of which you can point at and say when it became
# knowable.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from ppa.config import DATA_PROCESSED
from ppa.data import calendar as cal
from ppa.features.build import (
    LAST_PERIOD_BEFORE_GATE,
    PERIOD_LAGS_DAYS,
    build,
    feature_columns,
)

features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
columns = feature_columns(features)
print(f"{len(columns)} features on {len(features):,} rows")

# %% [markdown]
# ## 1. The five families

# %%
families = {
    "period lags": [c for c in columns if "_lag_" in c],
    "morning aggregates": [c for c in columns if "_am_" in c],
    "rolling daily": [c for c in columns if "_roll_" in c],
    "Fourier / calendar": [c for c in columns if c.startswith(("sin_", "cos_"))]
    + ["settlement_period", "day_of_week", "day_of_year", "month", "is_weekend"],
    "weather": [c for c in columns if c.endswith("_fcst")],
}
pd.DataFrame([
    {"family": name, "n": len(cols), "examples": ", ".join(cols[:3])}
    for name, cols in families.items()
])

# %% [markdown]
# ## 2. Period lags — why merging, not shifting
#
# The obvious way to build "the price at this period two days ago" is a positional shift of
# `48 × 2 = 96` rows. **That is wrong**, and it is wrong for the reason Lesson 03 established:
# days do not all have 48 rows.
#
# On the March transition a 96-row shift lands an hour off, and it never recovers — every
# subsequent lag is misaligned until the October transition shifts it back.
#
# `_period_lags` merges on `(settlement_date − N days, settlement_period)` instead, which is
# correct by construction.

# %%
panel = pd.read_parquet(DATA_PROCESSED / "panel.parquet").sort_values("start_time")
panel = panel.reset_index(drop=True)

shifted = panel["price"].shift(96)
correct = features.set_index(["settlement_date", "settlement_period"])["price_lag_2d"]
merged = panel.set_index(["settlement_date", "settlement_period"]).join(
    correct.rename("correct_lag")
)
merged["positional_lag"] = shifted.to_numpy()

around = merged.loc["2024-04-01":"2024-04-03"]
mismatch = (around["correct_lag"] - around["positional_lag"]).abs()
print(f"days just after the March 2024 transition:")
print(f"  rows where the two methods disagree: {(mismatch > 0.01).sum()} of {len(around)}")
print(f"  mean absolute disagreement         : {mismatch.mean():.2f} GBP/MWh")

# %% [markdown]
# A positional shift silently produces a different feature for days after every clock change.
# Nothing errors.

# %%
print(f"period lags used: {PERIOD_LAGS_DAYS} days")
print("1 is absent — see Lesson 09 for why it is not knowable at gate closure.")

# %% [markdown]
# ## 3. Morning aggregates — the freshest legitimate information
#
# The banned 1-day lag would have been the strongest feature available. What replaces it is
# **T−1's morning**: settlement periods 1 to 22, which end exactly at 11:00 local — the gate
# closure instant.

# %%
gate_local = cal.to_utc("2024-06-14", LAST_PERIOD_BEFORE_GATE) + pd.Timedelta(minutes=30)
print(f"SP{LAST_PERIOD_BEFORE_GATE} on 2024-06-14 ends at {gate_local.tz_convert(cal.TZ)} local")
print("which is exactly gate closure for 2024-06-15.")

# %%
day = panel[panel["settlement_date"] == "2024-06-14"]
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(day["settlement_period"], day["price"], lw=2, color="black")
ax.axvspan(0.5, LAST_PERIOD_BEFORE_GATE + 0.5, alpha=0.2, color="green",
           label=f"known at gate closure (SP1-{LAST_PERIOD_BEFORE_GATE})")
ax.axvspan(LAST_PERIOD_BEFORE_GATE + 0.5, 48.5, alpha=0.15, color="crimson",
           label="has not happened yet")
ax.set(xlabel="settlement period on T-1", ylabel="GBP/MWh",
       title="What we know about yesterday when we forecast tomorrow")
ax.legend()
plt.show()

# %% [markdown]
# Note how much of the day is unavailable — including the entire evening peak, which is the
# most informative part. That is the cost of doing this honestly, and it is why the morning
# aggregates carry mean, max, min *and* standard deviation: if you only get half a day, extract
# as much shape from it as you can.

# %%
features[[c for c in columns if c.startswith("price_am_")]].describe().T.round(2)

# %% [markdown]
# ## 4. Residual load, lagged the same way
#
# Everything said about price lags applies to demand. `residual_load` is the demand-side
# feature that matters (Lesson 02), and it gets the same lag structure and the same morning
# aggregates.
#
# One subtlety worth noting: NESO demand is an **outturn**, published after the fact. Treating
# T−1's morning demand as known at 11:00 on T−1 assumes near-real-time availability, which is
# broadly true for GB (NESO publishes with a short delay) but is an assumption. In a production
# system you would use the *demand forecast*, which NESO also publishes.

# %% [markdown]
# ## 5. Rolling windows, and the `shift(2)` that matters

# %%
daily_price = panel.groupby("settlement_date")["price"].mean().reset_index()
daily_price["roll_no_shift"] = daily_price["price"].rolling(7).mean()
daily_price["roll_shift_2"] = daily_price["price"].shift(2).rolling(7).mean()

window = daily_price[daily_price["settlement_date"].between("2024-06-10", "2024-06-20")]
window[["settlement_date", "price", "roll_no_shift", "roll_shift_2"]].round(2)

# %% [markdown]
# Without the shift, the window for target day T **includes T itself** — the answer is in the
# feature. With `shift(2)`, the window ends at T−2 and T−1's incomplete day never enters.
#
# This is a one-token difference that would turn a legitimate model into a leaking one, which
# is exactly why `tests/test_no_leakage.py` recomputes the value by hand from the raw panel
# rather than trusting the pipeline.

# %% [markdown]
# ## 6. Fourier terms — teaching the model that period 48 is next to period 1
#
# A tree can split on `settlement_period` directly, but every split is a hard boundary. It has
# no way to know that period 48 and period 1 are half an hour apart, so it learns 48 more or
# less independent step functions and needs a lot of data to do it.
#
# Fourier terms make the wraparound explicit and let the model share information between
# neighbouring periods.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
periods = np.arange(1, 49)
for k in (1, 2, 3):
    ax1.plot(periods, np.sin(2 * np.pi * k * periods / 48), label=f"sin, k={k}")
ax1.set(xlabel="settlement period", title="Daily harmonics")
ax1.legend()

sample = features.sample(4000, random_state=0)
ax2.scatter(sample["sin_day_1"], sample["cos_day_1"], c=sample["settlement_period"],
            cmap="twilight", s=8)
ax2.set(xlabel="sin_day_1", ylabel="cos_day_1",
        title="The day, as a circle — period 48 is adjacent to period 1")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Weather, degraded on purpose
#
# The weather columns end in `_fcst` and are **daily aggregates**, not half-hourly outturn.
#
# That is a deliberate downgrade. We have the archive, which is what the weather actually did —
# far more information than a real day-ahead forecast would have carried. Averaging to a daily
# value throws away the intraday precision we should not have, leaving roughly the information
# content a genuine forecast provides.
#
# It is a mitigation, not a fix. Lesson 10 measures the residual advantage by removing weather
# entirely.

# %%
weather_columns = families["weather"]
per_day = features.groupby("settlement_date")[weather_columns].nunique().max()
print("distinct values per day for each weather feature (1 = constant within the day):")
print(per_day.to_string())

# %% [markdown]
# ## 8. Missing values, and why we keep the rows

# %%
missing = features[columns].isna().sum().sort_values(ascending=False)
missing[missing > 0].to_frame("missing rows")

# %% [markdown]
# The early rows of the series have no 14-day lag, and rolling windows need warm-up. Dropping
# them would remove the first two weeks of every training window.
#
# XGBoost handles NaN natively — it learns a default direction per split — so those rows are
# kept. That is a real advantage of gradient boosting over the OLS benchmark in Lesson 07,
# which had to drop them.

# %%
print(f"rows total                    : {len(features):,}")
print(f"rows with every feature present: {len(features.dropna(subset=columns)):,}")
print(f"rows XGBoost can still use     : {len(features):,}  (all of them)")

# %% [markdown]
# ## 9. Which features does the model actually use?

# %%
import json

from ppa.models import xgb

sample = features.dropna(subset=["price"]).tail(40000)
model = xgb.build({"n_estimators": 200}).fit(sample[columns], sample["price"])
importance = xgb.importance(model, columns).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
top = importance.iloc[::-1]
ax.barh(top["feature"], top["gain"], color="#4C72B0")
ax.set(xlabel="gain", title="Feature importance (indicative — fit on the last 40k rows)")
plt.tight_layout()
plt.show()

importance.round(4)

# %% [markdown]
# The morning aggregates and the price lags dominate, as you would expect. What matters is that
# **residual load features appear at all** — that is the demand-side information doing work
# beyond price persistence, and it is the reason the model beats a pure-lag baseline.

# %% [markdown]
# ## Exercises
#
# 1. Remove all residual-load features and re-run Lesson 09. How much of the 22–30% improvement
#    was demand information rather than better use of price history?
# 2. Add `price_lag_1d` (the banned one) and measure the MAE improvement. That number is the
#    size of the overstatement you would be making.
# 3. Replace the daily weather aggregates with the half-hourly outturn. Compare against the
#    ablation in Lesson 10 — where does the honest number sit between the two extremes?
#
# ## Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 08: feature families, merge-not-shift lags, Fourier terms"
# ```
#
# ---
#
# **Next:** `09-walkforward-and-leakage.ipynb` — putting it all under a harness that cannot peek.
