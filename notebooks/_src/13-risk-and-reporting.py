# %% [markdown]
# # Lesson 13 — Risk statistics, and the ones that are easy to get wrong
#
# **You will end with:** the final report, and a clear view of the three places where a
# perfectly good strategy gets misreported.

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
from ppa.strategy import risk

daily = pd.read_parquet(DATA_PROCESSED / "backtest.parquet")
report = json.loads((REPORTS / "strategy_metrics.json").read_text())
print(f"{len(daily):,} trading days")

# %% [markdown]
# ## 1. Annualisation — the mistake nobody checks
#
# Sharpe is a *rate*, so it needs a period. Our P&L series is **daily**, and power markets trade
# every day of the year — no weekends off, unlike equities.

# %%
options = pd.DataFrame([
    {"convention": "equity trading days", "periods_per_year": 252,
     "factor": np.sqrt(252), "sharpe": daily["pnl"].mean() / daily["pnl"].std(ddof=1) * np.sqrt(252)},
    {"convention": "calendar days (correct here)", "periods_per_year": 365.25,
     "factor": np.sqrt(365.25), "sharpe": risk.sharpe(daily["pnl"])},
    {"convention": "half-hourly periods (wrong)", "periods_per_year": 48 * 365.25,
     "factor": np.sqrt(48 * 365.25),
     "sharpe": daily["pnl"].mean() / daily["pnl"].std(ddof=1) * np.sqrt(48 * 365.25)},
])
options.round(2)

# %% [markdown]
# Three conventions, three answers, a factor of seven between the extremes. **Nobody reading a
# Sharpe ratio asks which one you used.**
#
# The third row is what the first version of this project did — treating the 48 half-hours of a
# day as independent bets, when the schedule for all of them came from one forecast at one gate
# closure. That is what produced a Sharpe of 50.
#
# The rule: **the annualisation period must match the unit of decision.** Here that is a day.

# %% [markdown]
# ## 2. Drawdown, and the notional that does not exist
#
# Maximum drawdown is usually quoted as a percentage. A percentage of *what*?
#
# This strategy has no capital base — it is quoted per MW of battery. Choosing a notional in
# order to express drawdown as a percentage would be inventing the denominator, and you could
# pick any number you liked.

# %%
cumulative = daily["pnl"].cumsum()
drawdown = cumulative - cumulative.cummax()
times = pd.to_datetime(daily["start_time"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(times, cumulative, lw=1.6)
ax1.plot(times, cumulative.cummax(), lw=1, ls="--", color="grey", label="running peak")
ax1.axvspan(pd.Timestamp(STUDY.volatile_start, tz="UTC"),
            pd.Timestamp(STUDY.volatile_end, tz="UTC"),
            alpha=0.12, color="crimson", label="gas crisis")
ax1.set(ylabel="cumulative P&L (GBP)", title="1 MW / 2 MWh battery")
ax1.legend()

ax2.fill_between(times, drawdown, 0, color="crimson", alpha=0.55)
ax2.set(ylabel="drawdown (GBP)")
plt.tight_layout()
plt.show()

print(f"max drawdown : {risk.max_drawdown(daily['pnl']):,.0f} GBP")
print(f"as a share of total P&L: {abs(risk.max_drawdown(daily['pnl'])) / daily['pnl'].sum():.2%}")

# %% [markdown]
# The drawdown is tiny relative to cumulative profit, which is the signature of a physical
# arbitrage rather than a directional bet. The battery is not taking a view on where prices are
# going; it is harvesting a spread that is almost always there.

# %% [markdown]
# ## 3. Hit rate — count the right denominator
#
# "Percentage of profitable periods" is ambiguous when the strategy sometimes sits flat. Include
# the flat days and the number drifts towards whatever share of the time you traded; exclude
# them and it measures decision quality.

# %%
traded = daily[daily["pnl"] != 0]
print(f"days in sample        : {len(daily):,}")
print(f"days actually traded  : {len(traded):,}")
print(f"hit rate (traded only): {risk.hit_rate(daily['pnl']):.1%}   <- reported")
print(f"hit rate (all days)   : {(daily['pnl'] > 0).mean():.1%}")

# %% [markdown]
# Here the two are close because the battery trades on almost every day. On a more selective
# strategy the gap would be large, and quoting the wrong one would be flattering.

# %% [markdown]
# ## 4. Confidence intervals on the Sharpe
#
# A point estimate of a Sharpe ratio without an interval is not a result. Block bootstrap with
# one-week blocks, because weather and price regimes persist for days.

# %%
xgb = report["schedules"]["xgb"]
lo, hi = xgb["sharpe_ci95"]
print(f"Sharpe {xgb['sharpe']:.2f}   95% CI [{lo:.2f}, {hi:.2f}]")

widths = []
for block in (1, 7, 30):
    a, b = risk.bootstrap_sharpe_ci(daily["pnl"], n_boot=400, block=block)
    widths.append({"block_days": block, "ci_low": a, "ci_high": b, "width": b - a})
pd.DataFrame(widths).round(2)

# %% [markdown]
# Longer blocks give wider intervals — the honest direction. An i.i.d. bootstrap (block=1)
# would understate the uncertainty by pretending consecutive days are independent.

# %% [markdown]
# ## 5. Regimes, declared in advance

# %%
by_regime = pd.DataFrame(xgb["by_regime"]).T
by_regime[["n", "total", "mean_per_period", "sharpe", "hit_rate", "max_drawdown"]].round(2)

# %%
by_year = pd.DataFrame(xgb["by_year"]).T
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(by_year.index, by_year["mean_per_period"], color="#4C72B0")
ax.set(ylabel="GBP per day", title="Daily profit by year")
plt.show()
by_year[["n", "total", "mean_per_period", "sharpe", "hit_rate"]].round(2)

# %% [markdown]
# Profitable in every year and every regime. Profit *per day* peaks during the crisis, because
# spreads were widest — but the Sharpe is comparable throughout, which is the right pattern.
# A strategy whose Sharpe collapsed outside the crisis would be a coincidence dressed up.
#
# The regime dates come from `config.py` and were fixed from market history, not chosen after
# looking at where the strategy worked. That distinction is worth stating out loud, because it
# is unfalsifiable from the outside and it is the first thing a sceptical reader wonders.

# %% [markdown]
# ## 6. What the forecast is worth
#
# The number to lead with. Three schedules, identical battery:

# %%
schedules = pd.DataFrame(report["schedules"]).T[
    ["total", "mean_per_period", "sharpe", "hit_rate"]
]
schedules.round(2)

# %%
print(f"uplift over naive : {report['uplift_vs_naive']:,.0f} GBP "
      f"({report['uplift_vs_naive_pct']:+.1%})")
print(f"share of oracle   : {report['share_of_oracle']:.1%}  "
      f"(naive: {report['naive_share_of_oracle']:.1%})")

# %% [markdown]
# **The Sharpe is not the achievement.** The naive schedule scores about 8 on its own, because
# the intraday spread is nearly always positive — that is physics and demand, not prediction.
#
# The forecast's contribution is the uplift, and the share-of-oracle figure says how much of the
# theoretically available money it captures. Those two numbers are what the model earned.

# %% [markdown]
# ## 7. Reproducibility
#
# One thing worth checking on a project like this: does it give the same answer twice? XGBoost
# with a fixed seed is deterministic in principle, but histogram binning and thread scheduling
# introduce small variation.
#
# Re-running the full pipeline produced 29.9% against a previously recorded 29.8%, and a Sharpe
# of 10.04 against 10.01. That is the level of run-to-run variation to expect, and it is small
# relative to the confidence intervals — which is the relevant comparison.
#
# It is also the reason `RESUME_CLAIMS.md` quotes 22%: a figure that moves in the second decimal
# place between runs should not be quoted to three significant figures.

# %% [markdown]
# ## 8. The report
#
# `make report` writes `reports/metrics.json` (machine-readable, cited by `RESUME_CLAIMS.md`,
# published by CI) and `reports/report.html` (self-contained, charts embedded as data URIs,
# the thing to open in an interview).

# %%
metrics_path = REPORTS / "metrics.json"
full = json.loads(metrics_path.read_text())
print("top-level keys:", list(full))
print()
print(json.dumps({k: v for k, v in full["study"].items()}, indent=2))

# %% [markdown]
# ## Exercises
#
# 1. Compute a rolling 90-day Sharpe and plot it. Is the edge stable, or decaying?
# 2. The report shows a frictionless reference run. What share of gross P&L do frictions consume,
#    and at what friction level would the strategy break even?
# 3. Add a Sortino ratio (downside deviation only). Does it change the picture, and would you
#    expect it to for a strategy with an 87% hit rate?
#
# ## Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 13: risk statistics, annualisation, regimes, the report"
# ```
#
# ---
#
# **Next:** `14-forecast-uncertainty.ipynb` — putting an honest interval around the point
# forecast, which is what a desk asks immediately after "how accurate is it?".
#
# After that `course/15-reproducibility.md` covers `make all` and CI, and `RESUME_CLAIMS.md`
# is the document to reread before an interview.
