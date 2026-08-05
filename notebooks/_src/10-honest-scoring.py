# %% [markdown]
# # Lesson 10 — Honest scoring: which number do you put on the resume?
#
# **You will end with:** three defensible figures for the same result, and a principled reason
# for choosing between them.
#
# This is the lesson where the project's central discipline gets applied to itself.

# %%
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from ppa.config import DATA_PROCESSED, REPORTS, STUDY
from ppa.eval import metrics
from ppa.models import naive
from ppa.strategy import risk

report = json.loads((REPORTS / "forecast_metrics.json").read_text())
predictions = pd.read_parquet(DATA_PROCESSED / "walkforward_predictions.parquet")
print(f"{report['n_predictions']:,} out-of-sample half-hours")

# %% [markdown]
# ## 1. Why MAE, and not the alternatives
#
# **RMSE** is dominated by a handful of scarcity hours. From Lesson 05, the top 0.1% of prices
# are many times the median; under squared loss a model can win by being better at four hours a
# year and worse the rest of the time.
#
# **MAPE** divides by price. GB prices touch zero and go negative — so MAPE is undefined or
# explosive exactly where it matters most.

# %%
truth = predictions["price"].to_numpy()
xgb = predictions["pred_xgb"].to_numpy()

mape = metrics.mape_filtered(truth, xgb, floor=10.0)
print(f"sMAPE (bounded, defined at zero)  : {metrics.smape(truth, xgb):.2f}%")
print(f"MAPE on |price| >= 10             : {mape['mape']:.2f}%")
print(f"  ...rows excluded to compute it  : {mape['excluded_rows']:,} ({mape['excluded_share']:.2%})")

# %% [markdown]
# Reporting MAPE without saying how many rows you dropped to make it computable is the kind of
# thing that gets caught in an interview.

# %% [markdown]
# ## 2. The three figures
#
# The same model, scored three defensible ways.

# %%
models = report["models"]
xgb_mae = models["xgb"]["mae"]

ablation_path = REPORTS / "forecast_metrics_no_weather.json"
ablation = json.loads(ablation_path.read_text()) if ablation_path.exists() else None

rows = [
    {"comparison": f"vs declared baseline ({naive.HEADLINE_BASELINE})",
     "baseline_mae": models[naive.HEADLINE_BASELINE]["mae"],
     "model_mae": xgb_mae,
     "improvement": models["xgb"]["mae_improvement_vs_naive"]},
]

strongest = min(
    (n for n in models if n != "xgb"), key=lambda n: models[n]["mae"]
)
rows.append({
    "comparison": f"vs STRONGEST baseline ({strongest})",
    "baseline_mae": models[strongest]["mae"],
    "model_mae": xgb_mae,
    "improvement": 1 - xgb_mae / models[strongest]["mae"],
})

if ablation:
    rows.append({
        "comparison": "vs declared baseline, NO weather features",
        "baseline_mae": models[naive.HEADLINE_BASELINE]["mae"],
        "model_mae": ablation["models"]["xgb"]["mae"],
        "improvement": ablation["mae_improvement_vs_naive"],
    })

choices = pd.DataFrame(rows)
choices.round(3)

# %% [markdown]
# ### Which one goes on the resume?
#
# The temptation is the largest. The argument against it has two parts.
#
# **Part 1 — the declared baseline is not the hardest one.** `HEADLINE_BASELINE` was fixed in
# code before any model was fitted, which is the right discipline: it stops you shopping for a
# flattering comparison after the fact. But on this data another variant scores better, and
# quoting an improvement over a beatable opponent overstates what the model does.
#
# **Part 2 — the weather features are optimistic.** They come from the archive, not a real
# day-ahead forecast (Lesson 02, section 5).
#
# Two independent stress tests. And they land on the same number.

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))
colors = ["#999", "#4C72B0", "#4C72B0"]
bars = ax.barh(choices["comparison"], choices["improvement"] * 100, color=colors)
ax.axvline(18, color="crimson", ls="--", lw=2, label="18% (the original resume claim)")
for bar, value in zip(bars, choices["improvement"], strict=True):
    ax.text(value * 100 + 0.4, bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}", va="center")
ax.set(xlabel="MAE improvement (%)", title="Three honest ways to score the same model")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Quote the lower figure.** It survives both stress tests, the two tests agree with each
# other, and it still exceeds the original claim.
#
# The convergence is what makes it trustworthy. One conservative number could be a coincidence;
# two arrived at by unrelated routes is a signal.

# %% [markdown]
# ## 3. Is the difference significant?
#
# The Diebold-Mariano test compares forecast accuracy. Its statistic is the mean loss
# differential over its standard error — and that standard error has to account for the serial
# correlation in half-hourly forecast errors. A plain i.i.d. variance would treat 77,000
# correlated errors as 77,000 independent observations and hugely overstate significance.
#
# `eval/metrics.py` uses a Newey-West variance with a rule-of-thumb bandwidth.

# %%
dm = report["diebold_mariano_xgb_vs_naive"]
print(f"statistic : {dm['statistic']:.2f}   (negative favours XGBoost)")
print(f"p-value   : {dm['p_value']:.2e}")
print(f"n         : {dm['n']:,}")
print()

naive_iid = (
    np.mean(np.abs(truth - xgb) - np.abs(truth - predictions[f'pred_{naive.HEADLINE_BASELINE}']))
    / (np.std(np.abs(truth - xgb) - np.abs(truth - predictions[f'pred_{naive.HEADLINE_BASELINE}']), ddof=1)
       / np.sqrt(len(truth)))
)
print(f"naive i.i.d. statistic would be: {naive_iid:.2f}")
print(f"Newey-West correction shrinks it by a factor of {abs(naive_iid / dm['statistic']):.1f}")

# %% [markdown]
# The correction matters — it shrinks the statistic substantially. It is still overwhelming,
# but the point is that we *applied* the correction rather than quoting the inflated number.

# %% [markdown]
# ## 4. Confidence intervals on MAE
#
# Blocks of one day, not i.i.d. resampling, for the same autocorrelation reason.

# %%
interval_rows = []
for name, block in models.items():
    interval_rows.append({
        "model": name, "mae": block["mae"],
        "ci_low": block["mae_ci95"][0], "ci_high": block["mae_ci95"][1],
        "width": block["mae_ci95"][1] - block["mae_ci95"][0],
    })
intervals = pd.DataFrame(interval_rows).sort_values("mae")

fig, ax = plt.subplots(figsize=(9, 4))
ax.errorbar(intervals["mae"], intervals["model"],
            xerr=[intervals["mae"] - intervals["ci_low"],
                  intervals["ci_high"] - intervals["mae"]],
            fmt="o", capsize=5, lw=2)
ax.set(xlabel="MAE (GBP/MWh)", title="Block-bootstrap 95% intervals")
plt.tight_layout()
plt.show()

intervals.round(3)

# %% [markdown]
# The intervals do not overlap between XGBoost and any baseline. Unlike the credit project —
# where 250 held-out rows gave an AUC interval you could drive a truck through — 77,000
# out-of-sample observations make this a genuinely precise measurement.

# %% [markdown]
# ## 5. Where the improvement comes from

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

for column, label in [("pred_xgb", "XGBoost"),
                      (f"pred_{naive.HEADLINE_BASELINE}", "seasonal naive")]:
    per_period = metrics.by_period_of_day(predictions, "price", column)
    axes[0].plot(per_period["settlement_period"], per_period["mae"], lw=2, label=label)
axes[0].set(xlabel="settlement period", ylabel="MAE", title="By time of day")
axes[0].legend()

labelled = predictions.copy()
labelled["regime"] = risk.label_regime(labelled["start_time"])
regime_rows = []
for regime, group in labelled.groupby("regime"):
    m = metrics.mae(group["price"], group["pred_xgb"])
    b = metrics.mae(group["price"], group[f"pred_{naive.HEADLINE_BASELINE}"])
    regime_rows.append({"regime": regime, "xgb": m, "naive": b, "improvement": 1 - m / b})
regime_table = pd.DataFrame(regime_rows)

x = np.arange(len(regime_table))
axes[1].bar(x - 0.2, regime_table["naive"], 0.4, label="naive")
axes[1].bar(x + 0.2, regime_table["xgb"], 0.4, label="XGBoost")
axes[1].set_xticks(x, [r.replace("_", "\n") for r in regime_table["regime"]])
axes[1].set(ylabel="MAE", title="By regime")
axes[1].legend()
plt.tight_layout()
plt.show()

regime_table.round(3)

# %% [markdown]
# The improvement holds in every regime. **Absolute** MAE is much larger during the crisis — of
# course, prices were several times higher — which is exactly why the *ratio* is the honest
# thing to quote and a raw MAE is not comparable across periods.

# %% [markdown]
# ## 6. Bias
#
# One last check that costs nothing: is the model systematically high or low?

# %%
for name, block in models.items():
    print(f"{name:28} bias {block['bias']:+8.3f} GBP/MWh")

# %% [markdown]
# A large positive or negative bias would mean the model has learned a level offset — often a
# sign that the training window's price level differs from the test window's, which on this
# data is a real risk given the regime shifts. Small bias here means the expanding window is
# tracking the level adequately.

# %% [markdown]
# ## 7. The resume line
#
# > Modeled UK day-ahead electricity spot prices on half-hourly data using lagged demand,
# > weather and seasonality features, cutting forecast error by **22%** against a seasonal
# > naive baseline under walk-forward validation.
#
# And the follow-up you should be ready for — *"why 22 and not 30?"* — has a good answer, which
# is the point of this whole lesson.
#
# ## Exercises
#
# 1. Compute the improvement per settlement period. Where is it largest, and does that match
#    where a battery makes its money?
# 2. Re-run the DM test with `power=2` (squared loss). Does the conclusion change? Should it?
# 3. Split the out-of-sample period in half by date and compute the improvement in each. Is it
#    stable, or is it decaying as the model's training data ages?
#
# ## Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 10: honest scoring, DM test, choosing the number to quote"
# ```
#
# ---
#
# **Next:** `11-strategy.ipynb` — turning the forecast into money.
