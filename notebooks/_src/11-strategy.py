# %% [markdown]
# # Lesson 11 — From forecast to money, and a design that was thrown away
#
# **You will end with:** a backtested strategy, and a working instinct for when a backtest is
# telling you something and when it is telling you nothing.
#
# Beating a baseline on MAE is a forecasting result. It is not money. Getting from one to the
# other requires saying precisely **what you would trade**, and that turns out to be the hard
# part.

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
from ppa.strategy import backtest, risk, signals
from ppa.strategy.signals import PERIOD_HOURS, BatteryConfig

predictions = pd.read_parquet(DATA_PROCESSED / "walkforward_predictions.parquet")
print(f"{len(predictions):,} out-of-sample half-hours")

# %% [markdown]
# ## 1. The first design — and why it was wrong
#
# The obvious move: we forecast the price better than the naive baseline, so trade the
# difference. Take a position on `forecast − naive` and settle against `realised − naive`.
#
# Let us build it and see what happens.

# %%
reference = predictions["pred_same_period_last_week"]
signal = predictions["pred_xgb"] - reference
spread = predictions["price"] - reference

position = np.clip(signal / 20.0, -1, 1)
pnl = position * spread

daily = pd.Series(pnl.to_numpy()).groupby(
    pd.to_datetime(predictions["start_time"]).dt.date.to_numpy()
).sum()

# Annualised as if the half-hourly series were the unit of decision.
naive_sharpe = pnl.mean() / pnl.std() * np.sqrt(48 * 365.25)
print(f"total P&L          {pnl.sum():,.0f}")
print(f"'Sharpe'           {naive_sharpe:.1f}")
print(f"share of periods profitable  {(pnl > 0).mean():.1%}")

# %% [markdown]
# **A Sharpe of ~50.** Stop and think about that number rather than celebrating it.
#
# Renaissance Technologies' Medallion fund is reputed to run somewhere around 2. A Sharpe of
# 50 does not mean you have found something extraordinary; it means you have made a mistake.
# There are two here, and they are worth separating.
#
# ### Mistake 1: the instrument does not exist
#
# The payoff being backtested is `position × (price − price_one_week_ago)`. **There is no
# contract with that payoff.** You cannot go to an exchange and buy "this period's price minus
# the same period last week". The reference is a number we computed, not a price anyone will
# transact at.
#
# A synthetic spread you cannot trade will happily pay you a synthetic return. And notice the
# circularity: the signal is `forecast − reference` and the payoff is `realised − reference`.
# The same term appears in both, so a good forecast is *guaranteed* to look profitable. We
# have measured our forecast accuracy a second time and relabelled it "P&L".
#
# ### Mistake 2: the annualisation
#
# Even taking the P&L seriously, `sqrt(48 × 365.25)` treats every half-hour as an independent
# bet. But the position is driven by a single daily forecast — the 48 periods of a day are one
# decision, not 48. That inflates the Sharpe by roughly seven times on its own.

# %%
print(f"annualisation used above : {np.sqrt(48 * 365.25):.1f}   (half-hourly)")
print(f"correct for daily decisions: {np.sqrt(365.25):.1f}")
print(f"ratio                     : {np.sqrt(48):.1f}x overstatement")

# %% [markdown]
# **The lesson to carry out of this project:** when a backtest produces an implausible number,
# the first hypothesis is always that the backtest is broken — not that you are brilliant.
# Work out what physical transaction the P&L corresponds to. If you cannot name it, there
# isn't one.

# %% [markdown]
# ## 2. The replacement: a battery
#
# What *can* you do with a day-ahead price forecast? The canonical commercial answer is
# **storage arbitrage**.
#
# A 1 MW / 2 MWh battery bids into the day-ahead auction. At gate closure on T-1 we have a
# forecast for all 48 periods of T, so we commit to charge during the 4 cheapest forecast
# periods and discharge during the 4 dearest. Then the day happens and we settle at the
# **realised** prices.
#
# Every leg is a real trade at a real price. The P&L is money.
#
# Physical constraints that keep it honest:
#
# - **round-trip efficiency 85%** — you get back less than you put in;
# - **equal charge and discharge periods** — the battery starts and ends each day empty, so no
#   energy is borrowed across days;
# - **one cycle per day** — degradation makes more uneconomic, and it stops the strategy
#   manufacturing trades out of noise.

# %%
config = BatteryConfig()
print(f"{config.power_mw} MW / "
      f"{config.power_mw * config.periods_per_leg * PERIOD_HOURS} MWh")
print(f"round-trip efficiency {config.round_trip_efficiency:.0%}")
print(f"minimum forecast spread to trade a day: {config.min_forecast_spread} GBP/MWh")

# %% [markdown]
# ### One day, end to end

# %%
day = predictions[predictions["settlement_date"] == "2024-01-17"].copy()
scheduled = signals.build(day, config=config)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(scheduled["settlement_period"], scheduled["price"], lw=2, label="realised price")
ax.plot(scheduled["settlement_period"], scheduled["pred_xgb"], lw=1.6, ls="--",
        label="forecast (known at gate closure)")

charge = scheduled[scheduled["position"] < 0]
discharge = scheduled[scheduled["position"] > 0]
ax.scatter(charge["settlement_period"], charge["price"], s=130, marker="v",
           color="green", zorder=5, label="charge (buy)")
ax.scatter(discharge["settlement_period"], discharge["price"], s=130, marker="^",
           color="crimson", zorder=5, label="discharge (sell)")
ax.set(xlabel="settlement period", ylabel="GBP/MWh",
       title="2024-01-17: schedule chosen from the forecast, settled at realised prices")
ax.legend()
plt.show()

# %%
result = backtest.run_schedule(day, "pred_xgb", config=config)
print(f"bought  {-result.loc[result['position'] < 0, 'cashflow'].sum():.2f} GBP")
print(f"sold    {result.loc[result['position'] > 0, 'cashflow'].sum():.2f} GBP")
print(f"profit  {result['cashflow'].sum():.2f} GBP for the day")

# %% [markdown]
# ## 3. The full backtest
#
# Three schedules are always run side by side, and this is the part that makes the result
# interpretable:
#
# - **xgb** — our forecast;
# - **naive** — the same battery on the seasonal-naive forecast;
# - **oracle** — perfect foresight, the ceiling on what *any* forecast could earn.
#
# Reporting only the first would tell you nothing about whether the forecast mattered.

# %%
report = json.loads((REPORTS / "strategy_metrics.json").read_text())
table = pd.DataFrame(report["schedules"]).T[
    ["total", "mean_per_period", "sharpe", "hit_rate", "max_drawdown"]
]
table.columns = ["total P&L", "GBP/day", "Sharpe", "hit rate", "max drawdown"]
table.round(2)

# %%
print(f"uplift over naive schedule : {report['uplift_vs_naive']:,.0f} GBP "
      f"({report['uplift_vs_naive_pct']:+.1%})")
print(f"share of oracle captured   : {report['share_of_oracle']:.1%}")
print(f"  ...naive captures        : {report['naive_share_of_oracle']:.1%}")

# %% [markdown]
# ## 4. Reading the Sharpe correctly
#
# The xgb schedule scores around 10, which is still a large number. Before quoting it, ask the
# same question as in section 1: what is producing it?
#
# **The naive schedule alone scores about 8.** A battery makes money on almost any day,
# because the intraday peak-to-trough spread is almost always positive — evening peak prices
# exceed overnight prices as a matter of physics and demand, not as a matter of prediction.
#
# So this is a **physical arbitrage**, not a market-timing alpha. The high Sharpe reflects a
# structural feature of power markets that anyone with a battery can harvest.
#
# **The forecast's contribution is the uplift over naive, and nothing else.** That is the
# honest number to quote, and it is the one the README leads with.

# %%
schedules = report["schedules"]
fig, ax = plt.subplots(figsize=(8, 5))
names = ["naive", "xgb", "oracle"]
ax.bar(names, [schedules[n]["total"] for n in names],
       color=["#999", "#4C72B0", "#55a868"])
ax.set(ylabel="total P&L (GBP)", title="What the forecast is worth")
for i, n in enumerate(names):
    ax.text(i, schedules[n]["total"], f"{schedules[n]['total']:,.0f}",
            ha="center", va="bottom")
plt.show()

# %% [markdown]
# ## 5. Frictions
#
# Every MWh moved is charged transaction cost plus slippage, on both legs. A backtest that
# reports only the frictionless number is not reporting a strategy.

# %%
rows = []
for friction in [0.0, 0.25, 0.75, 2.0, 5.0]:
    frame = backtest.run_schedule(predictions, "pred_xgb", friction / 2, friction / 2, config)
    daily = backtest.daily_pnl(frame)
    rows.append({"friction_gbp_per_mwh": friction, "total_pnl": daily["pnl"].sum(),
                 "sharpe": risk.sharpe(daily["pnl"])})
sensitivity = pd.DataFrame(rows)
sensitivity.round(2)

# %% [markdown]
# The strategy is robust to frictions well beyond the assumed level — which makes sense, since
# it is harvesting a spread of tens of GBP/MWh with costs under one. That is worth stating,
# because it means the result does not hinge on an optimistic cost assumption.

# %% [markdown]
# ## 6. Regimes
#
# A strategy that only works during the 2021–22 gas crisis is a coincidence. The regime dates
# are fixed in `config.py` from market history, **not** chosen after seeing where the strategy
# worked.

# %%
by_regime = pd.DataFrame(report["schedules"]["xgb"]["by_regime"]).T
by_regime[["sharpe", "hit_rate", "mean_per_period", "n"]].round(2)

# %%
daily = pd.read_parquet(DATA_PROCESSED / "backtest.parquet")
times = pd.to_datetime(daily["start_time"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(times, daily["cum_pnl"], lw=1.6)
ax1.axvspan(pd.Timestamp("2021-09-01", tz="UTC"), pd.Timestamp("2023-06-30", tz="UTC"),
            alpha=0.12, color="crimson", label="gas crisis")
ax1.set(ylabel="cumulative P&L (GBP)", title="1 MW / 2 MWh battery")
ax1.legend()

drawdown = daily["cum_pnl"] - daily["cum_pnl"].cummax()
ax2.fill_between(times, drawdown, 0, color="crimson", alpha=0.5)
ax2.set(ylabel="drawdown (GBP)")
plt.tight_layout()
plt.show()

# %% [markdown]
# The strategy earns money in all three regimes — most per day during the crisis, when spreads
# were widest, but with a comparable Sharpe throughout. That is the pattern you want: the
# *level* of profit tracks volatility, the *reliability* does not.

# %% [markdown]
# ## 7. What this backtest does not model
#
# Say this before anyone asks:
#
# - battery **degradation** and cycle life — real batteries wear out, and that is a genuine
#   cost per cycle we have not charged;
# - **state of charge** beyond one daily cycle, and any multi-day optimisation;
# - **imbalance exposure** if the battery fails to deliver its committed schedule;
# - **market impact** — we assume a 1 MW bid does not move the clearing price, which is
#   reasonable at this size and would not be at 100 MW;
# - **grid connection and capacity charges**, which are material to a real project's economics;
# - the fact that a real auction takes a **bid curve**, not a point forecast.
#
# With those absent, the honest framing is *"a systematic schedule derived from the forecast,
# backtested with trading frictions"* — not *"profit I would have earned"*.

# %% [markdown]
# ## 8. Exercises
#
# 1. Charge a degradation cost per MWh cycled and find the level at which the strategy stops
#    being worthwhile. Compare it to published figures for lithium-ion cycle costs.
# 2. Allow two cycles a day (8 charge and 8 discharge periods). Does total P&L rise? Does
#    Sharpe? What does the difference tell you?
# 3. Build a schedule from the **oracle** but with each price perturbed by Gaussian noise of
#    increasing standard deviation. Plot P&L against noise. Where does our model sit on that
#    curve — i.e. what forecast error does our realised P&L correspond to?
#
# ## 9. Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 11: battery arbitrage, frictions, regimes"
# ```
#
# ---
#
# **Next:** `course/14-reproducibility.md` — `make all`, CI, and publishing the report.
