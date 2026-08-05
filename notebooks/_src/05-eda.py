# %% [markdown]
# # Lesson 05 — What this market actually looks like
#
# **You will end with:** the four structural facts about GB power prices that determine every
# modelling choice in the rest of the project.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from ppa.config import STUDY
from ppa.data import calendar as cal
from ppa.data.assemble import load

panel = load().dropna(subset=["price"])
panel["date"] = pd.to_datetime(panel["settlement_date"])
print(f"{len(panel):,} priced half-hours, {panel['date'].min().date()} .. {panel['date'].max().date()}")

# %% [markdown]
# ## Fact 1: three cycles, superimposed
#
# Intraday, weekly and annual. All three are driven by demand, and all three are *deterministic*
# — they come from the clock and the calendar, not from market dynamics. That matters for
# Lesson 07: it is why deterministic Fourier terms work better here than a stochastic seasonal
# ARIMA.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

by_period = panel.groupby("settlement_period")["price"].agg(["mean", "median"])
axes[0].plot(by_period.index, by_period["mean"], lw=2, label="mean")
axes[0].plot(by_period.index, by_period["median"], lw=2, label="median")
axes[0].set(xlabel="settlement period", ylabel="GBP/MWh", title="Intraday")
axes[0].legend()

by_dow = panel.groupby("day_of_week")["price"].mean()
axes[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], by_dow, color="#4C72B0")
axes[1].set(title="Weekly")

by_month = panel.groupby("month")["price"].mean()
axes[2].plot(by_month.index, by_month, "o-", lw=2)
axes[2].set(xlabel="month", title="Annual")
plt.tight_layout()
plt.show()

# %% [markdown]
# The intraday shape has the classic **twin peaks**: a morning ramp and a much larger evening
# peak around periods 33–40 (16:00–20:00). Note how far apart mean and median are at the
# evening peak — that gap is the spike risk, and it is why MAE rather than RMSE is the headline
# metric.

# %%
heat = panel.pivot_table(index="settlement_period", columns="month", values="price", aggfunc="median")
fig, ax = plt.subplots(figsize=(11, 6))
sns.heatmap(heat, cmap="RdYlBu_r", ax=ax, cbar_kws={"label": "median GBP/MWh"})
ax.set(xlabel="month", ylabel="settlement period", title="Median price by period and month")
plt.show()

# %% [markdown]
# ## Fact 2: prices go negative, and increasingly so

# %%
negative = panel[panel["price"] < 0]
by_year = panel.groupby(panel["date"].dt.year).agg(
    negative_periods=("price", lambda s: (s < 0).sum()),
    negative_share=("price", lambda s: (s < 0).mean()),
    min_price=("price", "min"),
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
ax1.bar(by_year.index.astype(str), by_year["negative_share"] * 100, color="#4C72B0")
ax1.set(ylabel="% of periods", title="Negative prices are becoming more common")

ax2.hist(negative["settlement_period"], bins=range(1, 50, 2), color="crimson")
ax2.set(xlabel="settlement period", title="When negative prices happen")
plt.tight_layout()
plt.show()

by_year

# %% [markdown]
# Negative prices cluster **overnight and around midday** — overnight when demand collapses and
# wind keeps blowing, midday when solar peaks. Both are the low-residual-load conditions from
# Lesson 02.
#
# The upward trend is the renewables build-out, and it has a direct consequence for the
# strategy: a battery's daily spread is widening, not narrowing.
#
# **What this rules out:** `log(price)`, MAPE, and any model with positive support. It is why
# `eval/metrics.py` reports sMAPE, and reports plain MAPE only on a filtered subset with the
# exclusions counted.

# %% [markdown]
# ## Fact 3: the distribution is heavy-tailed and asymmetric

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
ax1.hist(panel["price"], bins=200, color="#4C72B0")
ax1.set(xlabel="GBP/MWh", ylabel="periods", title="Full distribution")
ax1.axvline(0, color="black", lw=0.8)

ax2.hist(panel["price"].clip(-50, 300), bins=120, color="#4C72B0", log=True)
ax2.set(xlabel="GBP/MWh (clipped)", title="Log scale — note the right tail")
plt.tight_layout()
plt.show()

quantiles = panel["price"].quantile([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
quantiles.to_frame("GBP/MWh").round(2)

# %% [markdown]
# The top 0.1% of periods reach many times the median. **This is why the model optimises
# absolute error, not squared error.**
#
# Under squared loss, one 1,000 GBP/MWh scarcity hour contributes as much as several hundred
# ordinary hours. A model minimising RMSE will happily get worse at 99% of the day in exchange
# for being slightly better at the four hours a year that dominate the loss. That is why
# `models/xgb.py` sets `objective="reg:absoluteerror"` — the metric we report is the metric we
# optimise.

# %% [markdown]
# ## Fact 4: the level is not stationary

# %%
monthly = panel.groupby(panel["date"].dt.to_period("M"))["price"].agg(["mean", "std"])
monthly.index = monthly.index.to_timestamp()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(monthly.index, monthly["mean"], lw=2, label="monthly mean")
ax.fill_between(monthly.index, monthly["mean"] - monthly["std"],
                monthly["mean"] + monthly["std"], alpha=0.25, label="+/- 1 sd")
ax.axvspan(pd.Timestamp(STUDY.volatile_start), pd.Timestamp(STUDY.volatile_end),
           alpha=0.12, color="crimson", label="declared 'volatile' regime")
ax.set(ylabel="GBP/MWh", title="Level and volatility both shift by an order of magnitude")
ax.legend()
plt.show()

regimes = panel.copy()
regimes["regime"] = np.select(
    [regimes["date"] < STUDY.volatile_start,
     (regimes["date"] >= STUDY.volatile_start) & (regimes["date"] <= STUDY.volatile_end)],
    ["calm_pre_crisis", "volatile_gas_crisis"], default="post_crisis")
regimes.groupby("regime")["price"].agg(["mean", "std", "min", "max", "count"]).round(1)

# %% [markdown]
# Three consequences, each of which appears later:
#
# 1. **The regime dates are fixed from market history**, in `config.py`, *before* any strategy
#    result is seen. Choosing regime boundaries after noticing where a strategy worked is a
#    well-known way to manufacture a story.
# 2. **The walk-forward window is expanding, not rolling.** A rolling 2-year window in 2024
#    would have seen only crisis prices and none of the calm market it was about to re-enter.
# 3. **Absolute error is not comparable across regimes.** A MAE of 27 means something very
#    different at a mean price of 40 than at 200 — which is why Lesson 10 reports the
#    improvement *ratio* per month, not just the level.

# %% [markdown]
# ## The spikes
#
# Worth looking at the extremes directly, because they are where a forecaster fails and where
# a battery earns.

# %%
top = panel.nlargest(10, "price")[
    ["settlement_date", "settlement_period", "price", "national_demand", "residual_load",
     "embedded_wind"]
]
top.round(1)

# %% [markdown]
# High residual load and low wind, essentially every time. The market is short and the last
# available generator is expensive. Nothing mysterious — which is encouraging, because it means
# the features we have (residual load, wind) carry the relevant information.

# %% [markdown]
# ## Autocorrelation
#
# One last thing to check before building lag features: how far back does the price remember?

# %%
series = panel.set_index("start_time")["price"].asfreq("30min").interpolate()
lags = [1, 2, 24, 48, 96, 336, 672]
acf = {lag: series.autocorr(lag) for lag in lags}

pd.DataFrame({
    "lag_periods": lags,
    "lag_description": ["30 min", "1 hour", "12 hours", "1 day", "2 days", "1 week", "2 weeks"],
    "autocorrelation": [acf[lag] for lag in lags],
}).round(3)

# %% [markdown]
# Strong at every horizon, with a clear bump at 48 (one day) and 336 (one week) — the
# seasonality showing through. That is what makes the seasonal-naive baselines in Lesson 06
# genuinely hard to beat, and it is why the lag set is `[2, 3, 7, 14]` days rather than a dense
# sweep.

# %% [markdown]
# ## Exercises
#
# 1. Recompute the intraday profile separately for each regime. Does the evening peak move?
# 2. Find every period above the 99.9th percentile. What fraction occur in the evening peak,
#    and what fraction are in winter?
# 3. Compute the average daily peak-to-trough spread by year. That number is the battery's
#    revenue opportunity — is it growing?
#
# ## Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 05: EDA, seasonality, negative prices, regimes"
# ```
#
# ---
#
# **Next:** `06-baselines.ipynb` — declaring what we have to beat, before we try to beat it.
