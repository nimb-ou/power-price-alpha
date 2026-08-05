# %% [markdown]
# # Lesson 06 — Declaring the baseline before trying to beat it
#
# **You will end with:** a baseline fixed in code, and an understanding of why "beats a
# seasonal naive baseline" is a meaningless claim unless you say which one.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from ppa.config import DATA_PROCESSED
from ppa.eval import metrics
from ppa.models import naive

features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
print(f"declared headline baseline: {naive.HEADLINE_BASELINE}")
print(f"variants scored: {list(naive.BASELINES)}")

# %% [markdown]
# ## 1. The problem with "seasonal naive"
#
# It is not one thing. For a half-hourly series with daily and weekly cycles, at least these
# are all "seasonal naive":
#
# | Variant | Prediction for period *p* on day *T* |
# |---|---|
# | same period yesterday | price at *p* on T−1 |
# | same period two days ago | price at *p* on T−2 |
# | same period last week | price at *p* on T−7 |
# | yesterday's daily mean | mean price on T−1 |
# | last week's same weekday, shape-adjusted | T−7 rescaled by recent level |
#
# They differ by a lot. Pick the weakest after seeing the results and you can manufacture an
# impressive improvement without doing anything.
#
# So `models/naive.py` names one in code — `HEADLINE_BASELINE` — before any model is fitted.
# That is the discipline. Section 5 covers what to do when it turns out not to be the hardest.

# %% [markdown]
# ## 2. One variant is not allowed at all
#
# `same_period_yesterday` is the most obvious choice and it is **not implementable**. For a
# target at 14:30 on day T, the same period on T−1 is 14:30 on T−1 — three and a half hours
# after the 11:00 gate closure at which the forecast must be made.
#
# Including it would make our model look *worse* against an opponent that could not exist.

# %%
print(f"'same_period_yesterday' in BASELINES: "
      f"{'same_period_yesterday' in naive.BASELINES}")
print(f"'price_lag_1d' in the feature matrix: {'price_lag_1d' in features.columns}")

# %% [markdown]
# ## 3. The three legitimate variants

# %%
predictions = naive.predict_all(features)
truth = features["price"].to_numpy()

rows = []
for name in predictions.columns:
    summary = metrics.summary(truth, predictions[name].to_numpy())
    summary["variant"] = name
    summary["is_headline"] = name == naive.HEADLINE_BASELINE
    rows.append(summary)

table = pd.DataFrame(rows).set_index("variant")[
    ["n", "mae", "rmse", "smape", "bias", "is_headline"]
]
table.sort_values("mae").round(3)

# %% [markdown]
# ### Why `same_period_last_week` is the standard choice
#
# It preserves **both** cycles at once: the intraday shape *and* the weekday/weekend
# distinction. `same_period_two_days_ago` gets the intraday shape right but maps Saturday onto
# Thursday, which is a materially different demand profile.
#
# That is the argument in the electricity price forecasting literature (Weron's 2014 review and
# after), and it is why it was declared.

# %%
weekday_error = features.copy()
weekday_error["naive_week"] = predictions["same_period_last_week"]
weekday_error["naive_2d"] = predictions["same_period_two_days_ago"]

by_dow = []
for dow, group in weekday_error.dropna(subset=["naive_week", "naive_2d"]).groupby("day_of_week"):
    by_dow.append({
        "day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow],
        "last_week_mae": metrics.mae(group["price"], group["naive_week"]),
        "two_days_mae": metrics.mae(group["price"], group["naive_2d"]),
    })
dow_table = pd.DataFrame(by_dow)

fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(dow_table))
ax.bar(x - 0.2, dow_table["last_week_mae"], 0.4, label="same period last week")
ax.bar(x + 0.2, dow_table["two_days_mae"], 0.4, label="same period two days ago")
ax.set_xticks(x, dow_table["day"])
ax.set(ylabel="MAE (GBP/MWh)", title="Where each baseline struggles")
ax.legend()
plt.show()

dow_table.round(2)

# %% [markdown]
# The two-day lag is worst on **Saturday and Monday** — precisely the days where it crosses the
# weekday/weekend boundary and maps a working day onto a weekend or vice versa. The week lag
# has no such failure mode.
#
# So the theoretical argument for the week lag is right about *why*. It just does not win
# overall on this data, because the week lag pays a level penalty during volatile periods that
# outweighs its weekday advantage.

# %% [markdown]
# ## 4. A baseline that is not naive at all
#
# `last_known_morning_shape` is a genuinely competitive construction: take T−2's intraday
# shape and rescale it by how T−1's morning compared to T−2's morning. It uses the freshest
# legitimate level information available at gate closure.
#
# It is included precisely because a model should have to beat something thoughtful, not just a
# raw lag.

# %%
sample_date = "2024-02-14"
day = features[features["settlement_date"] == sample_date]
day_predictions = naive.predict_all(day)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(day["settlement_period"], day["price"], lw=2.5, color="black", label="realised")
for name in day_predictions.columns:
    ax.plot(day["settlement_period"], day_predictions[name], lw=1.5, ls="--", label=name)
ax.set(xlabel="settlement period", ylabel="GBP/MWh", title=f"Baselines on {sample_date}")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## 5. When your declared baseline is not the hardest one
#
# From the table in section 3: `same_period_two_days_ago` **beats** the declared headline. That
# is an awkward result and the honest handling matters.
#
# You have two obligations that pull in opposite directions:
#
# 1. **Declare the baseline in advance**, so you cannot shop for a flattering comparison after
#    seeing results. That is why `HEADLINE_BASELINE` is a constant in source control.
# 2. **Report against the toughest opponent**, so the number means something.
#
# The resolution: keep the declared baseline as the *stated* comparison — it is the literature
# standard and it was fixed in advance — but **quote the improvement against the strongest
# one** in any headline claim. `RESUME_CLAIMS.md` does exactly that, and shows both.
#
# Swapping `HEADLINE_BASELINE` now, after seeing the numbers, would be the thing the constant
# exists to prevent.

# %%
best_baseline = table["mae"].idxmin()
print(f"declared baseline : {naive.HEADLINE_BASELINE}  (MAE {table.loc[naive.HEADLINE_BASELINE, 'mae']:.3f})")
print(f"strongest baseline: {best_baseline}  (MAE {table.loc[best_baseline, 'mae']:.3f})")
print()
print("Any model improvement should be quoted against the second.")

# %% [markdown]
# ## 6. Why these are hard to beat
#
# From Lesson 05: autocorrelation at lag 336 (one week) is high. The price at this period last
# week is genuinely informative — it encodes the intraday shape, the weekday, roughly the
# season, and roughly the fuel-price level, all for free and with no model.
#
# A model has to add information *beyond* that. Demand, wind and temperature are how it does
# so, and Lesson 08 builds those features.

# %%
fig, ax = plt.subplots(figsize=(9, 6))
valid = features.dropna(subset=["price_lag_7d"]).sample(8000, random_state=0)
ax.scatter(valid["price_lag_7d"], valid["price"], s=4, alpha=0.15)
lims = [valid["price"].quantile(0.001), valid["price"].quantile(0.995)]
ax.plot(lims, lims, color="crimson", lw=1.5, label="perfect")
ax.set(xlim=lims, ylim=lims, xlabel="price one week ago", ylabel="price now",
       title="The baseline is already doing a lot of work")
ax.legend()
plt.show()

# %% [markdown]
# ## Exercises
#
# 1. Add a fourth variant: the mean of T−7 and T−14 at the same period. Does averaging two
#    noisy estimates beat either alone?
# 2. Build a variant that uses T−7 for weekdays and T−1's *weekend* equivalent for Saturdays
#    and Sundays. Does fixing the weekday problem beat the two-day lag?
# 3. Compute each baseline's MAE separately per regime. Does the ranking change between the
#    calm and crisis periods? What would that imply about quoting a single number?
#
# ## Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 06: declared baselines, and what to do when yours isn't hardest"
# ```
#
# ---
#
# **Next:** `07-statsmodels.ipynb` — a linear benchmark you can read.
