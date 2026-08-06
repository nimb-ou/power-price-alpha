# %% [markdown]
# # Lesson 14 — Forecasting the uncertainty, not just the number
#
# **You will end with:** quantile forecasts fitted under pinball loss, a demonstration that
# their nominal coverage is a lie out-of-sample, a conformal calibration that mostly fixes it,
# and a clear statement of what it still does not fix.
#
# Every number this course has produced so far is a point forecast. Lesson 10 established that
# the XGBoost model beats the strongest naive baseline by about 22% on MAE. A trading desk
# reading that has an obvious follow-up which the project could not answer: *how wrong is it
# likely to be tomorrow at 17:30?*
#
# "MAE is 27 GBP/MWh" is not the answer. That is an average over four and a half years, most
# of which was calm, and the evening peak in a volatile week behaves nothing like it. A
# position sized off the average error will be far too large exactly when it matters.

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
from ppa.models import conformal, walkforward, xgb

predictions = pd.read_parquet(DATA_PROCESSED / walkforward.PREDICTIONS_NAME)
print(f"{len(predictions):,} out-of-sample half-hours")
print([c for c in predictions.columns if c.startswith("pred_")])

# %% [markdown]
# ## 1. Pinball loss, and why it produces a quantile
#
# To fit the 10th percentile you need a loss that is minimised there. Squared error is
# minimised at the mean, absolute error at the median. Pinball loss is the generalisation:
#
# $$L_q(y, \hat{y}) = \max\big(q(y - \hat{y}),\; (q-1)(y - \hat{y})\big)$$
#
# At $q = 0.1$ an over-prediction costs nine times an under-prediction of the same size. That
# asymmetry is the entire mechanism — it drags the estimate down until only 10% of
# observations remain below it, because at that point the marginal cost of moving further down
# (paid on 90% of points) equals the marginal saving (collected on 10%).
#
# Let's confirm that empirically rather than take it on faith.

# %%
rng = np.random.default_rng(0)
sample = rng.normal(50, 15, 50_000)
candidates = np.arange(10, 90, 0.5)

fig, ax = plt.subplots(figsize=(9, 3.8))
for level, colour in zip((0.1, 0.5, 0.9), ("#2563eb", "#0f172a", "#dc2626")):
    losses = [metrics.pinball(sample, np.full_like(sample, c), level) for c in candidates]
    best = candidates[int(np.argmin(losses))]
    ax.plot(candidates, losses, color=colour, label=f"q={level}  minimised at {best:.1f}")
    ax.axvline(np.quantile(sample, level), color=colour, ls=":", lw=1)

ax.set_xlabel("constant prediction")
ax.set_ylabel("pinball loss")
ax.set_title("Pinball loss is minimised at the true quantile (dotted lines)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. One model, three quantiles
#
# `xgb.fit_predict_quantiles` fits P10, P50 and P90 in a single model using XGBoost's
# multi-quantile objective, rather than three separate models.
#
# The reason is not efficiency. Three independent fits have nothing coupling them, so on a
# volatile evening the P10 model and the P90 model can **cross** — the lower bound ends up
# above the upper bound. An interval like that is not a wide interval, it is a nonsensical one,
# and it will silently produce negative position sizes downstream. A shared tree structure
# does not make crossing impossible, so `walkforward.score` counts the crossings that remain
# instead of assuming there are none.
#
# Note also what the P50 is *not*: it is not the headline forecast. The headline comes from a
# separate model fitted under `reg:absoluteerror`. They estimate similar quantities and land
# close, but reporting whichever happened to score better would be exactly the quiet
# metric-shopping the rest of this repo is built to prevent.

# %%
intervals = walkforward.score_intervals(predictions)

print("pinball loss by quantile")
for name, value in intervals["pinball"].items():
    print(f"  {name}   {value:.3f}")
print(f"\nmean pinball (a coarse CRPS): {intervals['mean_pinball']:.3f}")
print(f"P50 MAE {intervals['median_mae']:.3f}  vs headline MAE "
      f"{metrics.mae(predictions['price'], predictions['pred_xgb']):.3f}")

# %% [markdown]
# ## 3. The interval does not cover what it claims
#
# A P10-P90 band should contain the outturn 80% of the time. Here is what it actually does.

# %%
band = intervals["interval"]
print(f"nominal coverage   {band['nominal_coverage']:.0%}")
print(f"empirical coverage {band['coverage']:.1%}")
print(f"mean width         {band['mean_width']:.1f} GBP/MWh")
print(f"breaches: {band['breaches_low']:,} below, {band['breaches_high']:,} above")
print(f"quantile crossings: {band['crossings']}")

# %% [markdown]
# That is a serious miss, and the first thing to establish is whether the *estimator* is broken
# or the *world* moved. The way to tell them apart is to look at coverage on the data the model
# was fitted on. If in-sample coverage is also wrong, the fitting is wrong. If in-sample is
# right and out-of-sample is wrong, the distribution shifted.

# %%
from ppa.features.build import build as build_features
from ppa.features.build import feature_columns

features = build_features()
window = features[features["start_time"] < "2021-01-01"]
columns = feature_columns(window)
cut = int(len(window) * 0.8)
train, test = window.iloc[:cut], window.iloc[cut:]

in_sample = xgb.fit_predict_quantiles(train, train, columns)
out_sample = xgb.fit_predict_quantiles(train, test, columns)

cov_in = np.mean((train["price"] >= in_sample[:, 0]) & (train["price"] <= in_sample[:, 2]))
cov_out = np.mean((test["price"] >= out_sample[:, 0]) & (test["price"] <= out_sample[:, 2]))

print(f"in-sample coverage   {cov_in:.3f}")
print(f"out-of-sample        {cov_out:.3f}")
print()
print(f"train prices: mean {train['price'].mean():.1f}  std {train['price'].std():.1f}")
print(f"test prices:  mean {test['price'].mean():.1f}  std {test['price'].std():.1f}")

# %% [markdown]
# In-sample coverage is essentially exactly nominal. The estimator is fine. The standard
# deviation of prices rose by around 45% across the split, and an interval calibrated on a
# calm window is simply too narrow for a volatile one.
#
# This is the same failure as lesson 09's leakage tests, seen from the other side. A model
# fitted on the past and evaluated on the future is not evaluated under exchangeability, and
# every guarantee that assumes exchangeability weakens accordingly.

# %% [markdown]
# ## 4. Conformal prediction
#
# Conformal prediction (Vovk and co-authors; the quantile-regression form is Romano, Patterson
# & Candès 2019) repairs coverage without touching the model. The recipe:
#
# 1. Hold out a **calibration window** the model was not fitted on.
# 2. On that window, compute the *conformity score* for each observation:
#    $E_i = \max(\hat{q}_{lo} - y_i,\; y_i - \hat{q}_{hi})$ — how far outside its own interval
#    the outturn fell, negative when comfortably inside.
# 3. Take the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile of those scores.
# 4. Widen the interval by that amount, both sides.
#
# Two details in that recipe do real work.
#
# **The score can be negative**, so the correction can *narrow* an over-wide interval. A
# one-sided score clipped at zero would only ever widen, and would leave a too-wide interval
# too wide forever.
#
# **The $n+1$** is not a rounding convenience. It is what turns an asymptotic statement into a
# finite-sample guarantee, by accounting for the test point itself being one of the $n+1$
# exchangeable observations. With it, marginal coverage is at least $1-\alpha$ for *any*
# underlying model, at any sample size.

# %%
rng = np.random.default_rng(2)
calibration_draw = rng.normal(0, 10, 5_000)
test_draw = rng.normal(0, 10, 5_000)

too_narrow = 6.0  # nominal 80% would need about +/- 12.8
adjuster = conformal.fit(
    calibration_draw, np.full(5_000, -too_narrow), np.full(5_000, too_narrow), alpha=0.2
)
lo, hi = adjuster.apply(np.full(5_000, -too_narrow), np.full(5_000, too_narrow))

print(f"offset learned:    {adjuster.global_offset:+.2f}")
print(f"coverage before:   {np.mean(np.abs(test_draw) <= too_narrow):.3f}")
print(f"coverage after:    {np.mean((test_draw >= lo) & (test_draw <= hi)):.3f}")

# %% [markdown]
# Exactly nominal, on synthetic data that genuinely is exchangeable. Now the hard case.

# %% [markdown]
# ## 5. On real prices, where exchangeability fails
#
# The walk-forward run holds back the last `CALIBRATION_DAYS` of each training window and
# calibrates on it. That window is the most *recent* slice, not a random sample — a random
# split would be exchangeable on paper and wrong in practice, because what the model is about
# to forecast is the period immediately following the window's end, not its average.

# %%
conformalised = intervals["interval_conformal"]

rows = pd.DataFrame(
    [
        {"band": "raw quantile fit", **{k: band[k] for k in
            ("coverage", "mean_width", "winkler", "crossings")}},
        {"band": "conformalised", **{k: conformalised[k] for k in
            ("coverage", "mean_width", "winkler", "crossings")}},
    ]
)
print(rows.to_string(index=False))
print(f"\nmean offset applied: {conformalised['mean_offset']:.1f} GBP/MWh "
      f"over a {conformalised['calibration_days']}-day calibration window")

# %% [markdown]
# Coverage improves substantially but does not reach 80%, and that shortfall is the honest
# result: exchangeability does not hold, so the guarantee does not either. What the guarantee
# buys you is that the failure is now bounded and diagnosable rather than arbitrary.
#
# Watch the **Winkler score**, not the coverage. Winkler is mean width plus a penalty
# proportional to how far outside a miss landed, and it is the one number here that cannot be
# gamed: any interval can reach 100% coverage by becoming infinitely wide, and doing so makes
# Winkler worse. The conformalised band is wider *and* scores better, which is what tells you
# the extra width was earned rather than bought.

# %% [markdown]
# ## 6. Uncertainty is not constant across the day
#
# A single scalar offset widens 04:00 and the evening peak by the same amount, and those are
# not the same quantity. `conformal.fit` also computes a per-settlement-period offset.

# %%
by_period = metrics.by_period_of_day(predictions, "price", "pred_xgb")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(by_period["settlement_period"], by_period["mae"], color="#0f172a", lw=2, label="MAE")
if "pred_xgb_p10_conformal" in predictions:
    width = (
        predictions.assign(
            w=predictions["pred_xgb_p90_conformal"] - predictions["pred_xgb_p10_conformal"]
        )
        .groupby("settlement_period")["w"]
        .mean()
    )
    ax.plot(width.index, width.to_numpy() / 2, color="#2563eb", lw=2, ls="--",
            label="half the conformalised interval width")

ax.set_xlabel("settlement period (1 = 00:00-00:30 local)")
ax.set_ylabel("GBP/MWh")
ax.set_title("Error and interval width both peak in the evening")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Both curves rise into the evening peak, which is reassuring — the interval is wide where the
# model is bad, not uniformly wide. Periods 33-38 (16:00-19:00) are where the merit order
# steepens, where the marginal plant is most likely to be a peaker, and where a battery makes
# most of its money. Being uncertain there is expensive and being *honestly* uncertain there is
# what lets you size the position correctly.

# %% [markdown]
# ## 7. Why this matters for the strategy
#
# Lesson 11's battery schedule trades whenever the forecast spread beats a fixed threshold.
# With a calibrated interval you can do something better: require the *interval* to clear the
# threshold, so the strategy stands down when the model itself says it does not know.
#
# That is a genuine improvement in principle and it is not implemented here — it belongs in a
# version of the strategy that was designed around it rather than retrofitted, and retrofitting
# it after seeing the backtest would be exactly the sin lesson 11 warns about. It is written up
# in the limitations instead.

# %% [markdown]
# ## 8. Exercises
#
# 1. `CALIBRATION_DAYS` is 60. Re-run with 15, 30, 120 and 240. Plot coverage and Winkler
#    against it. Is there a value that dominates, and does it depend on the regime?
# 2. Conformal coverage degrades as the test block gets further from the calibration window.
#    Measure that: split the predictions by how many days elapsed since the fold's calibration
#    window ended, and plot coverage against it. How stale is too stale?
# 3. Implement *adaptive* conformal inference (Gibbs & Candès 2021), which updates $\alpha$
#    online based on recent coverage. Does it beat the fixed offset on the gas-crisis regime?
# 4. The offset is symmetric. Prices spike upward far more than downward — measure the two
#    breach rates separately, then fit asymmetric offsets. Does Winkler improve?
# 5. Add the interval-aware trading rule from section 7 to `strategy/signals.py`. Before you
#    look at the P&L, write down what result would make you keep it and what would make you
#    throw it away.
#
# ## 9. Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 14: quantile forecasts and conformal calibration"
# ```
#
# ---
#
# **Next:** `course/15-reproducibility.md` — `make all`, CI, and publishing the report.
