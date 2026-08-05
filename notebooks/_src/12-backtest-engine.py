# %% [markdown]
# # Lesson 12 — Building a backtest you can trust
#
# **You will end with:** a cashflow engine tested against hand-computed answers, and a working
# suspicion of any backtest you did not write yourself.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from ppa.config import DATA_PROCESSED, STUDY
from ppa.strategy import backtest, risk, signals
from ppa.strategy.signals import PERIOD_HOURS, BatteryConfig

predictions = pd.read_parquet(DATA_PROCESSED / "walkforward_predictions.parquet")

# %% [markdown]
# ## 1. Test against arithmetic, not against itself
#
# The most common backtest bug is one where the engine is self-consistent and wrong. Comparing
# it against another implementation just means you have two implementations of the same
# misunderstanding.
#
# So every mechanic here is checked against a case where the answer can be computed by hand.

# %%
def one_day(prices, forecast=None, date="2024-01-01"):
    n = len(prices)
    return pd.DataFrame({
        "start_time": pd.date_range(f"{date} 00:00", periods=n, freq="30min", tz="UTC"),
        "settlement_date": date,
        "settlement_period": range(1, n + 1),
        "price": prices,
        "pred_xgb": forecast if forecast is not None else prices,
        "pred_same_period_last_week": prices,
    })


SMALL = BatteryConfig(periods_per_leg=2, round_trip_efficiency=1.0, min_forecast_spread=0.0)

# Buy 2 x 0.5 MWh at 10 and 20; sell 2 x 0.5 MWh at 90 and 80.
prices = [10.0, 90.0, 20.0, 80.0]
frame = backtest.run_schedule(one_day(prices), "pred_xgb", 0.0, 0.0, SMALL)

expected = (90 + 80) * PERIOD_HOURS - (10 + 20) * PERIOD_HOURS
print(f"hand-computed : {expected:.2f} GBP")
print(f"engine        : {frame['cashflow'].sum():.2f} GBP")
frame[["settlement_period", "price", "position", "energy_mwh", "cashflow"]]

# %% [markdown]
# ## 2. Round-trip efficiency belongs on the way out
#
# A battery loses energy. The convention is to apply the loss to the energy you sell, not to
# what you pay: you are billed for every MWh drawn from the grid, and you only get paid for
# what comes back out.

# %%
lossy = BatteryConfig(periods_per_leg=2, round_trip_efficiency=0.5, min_forecast_spread=0.0)
frame = backtest.run_schedule(one_day(prices), "pred_xgb", 0.0, 0.0, lossy)

charge_cost = -frame.loc[frame["position"] < 0, "cashflow"].sum()
revenue = frame.loc[frame["position"] > 0, "cashflow"].sum()

print(f"paid to charge  : {charge_cost:.2f}   (unaffected by efficiency)")
print(f"received to sell: {revenue:.2f}   (= {(90 + 80) * PERIOD_HOURS:.1f} x 0.5)")

# %% [markdown]
# Getting this backwards — applying efficiency to the purchase — would make the strategy look
# better and would be wrong in a way nobody spots by reading the equity curve.

# %% [markdown]
# ## 3. The schedule follows the forecast, not the outturn
#
# The single most important property. Feed the engine a forecast that is exactly *inverted*
# relative to reality and it must lose money — because the decision is made from the forecast
# alone, at gate closure, before the day happens.

# %%
inverted = [90.0, 10.0, 80.0, 20.0]
frame = backtest.run_schedule(one_day(prices, inverted), "pred_xgb", 0.0, 0.0, SMALL)

print(f"charges at periods: {frame.loc[frame['position'] < 0, 'settlement_period'].tolist()} "
      f"(where the FORECAST is low)")
print(f"realised prices there: {frame.loc[frame['position'] < 0, 'price'].tolist()}")
print(f"P&L: {frame['cashflow'].sum():.2f} GBP  <- loses money, correctly")

# %% [markdown]
# A backtest that made money here would be peeking. This test is in
# `tests/test_strategy.py::test_a_bad_forecast_loses_money`, and it is the cheapest possible
# guard against the whole class of look-ahead bugs.

# %% [markdown]
# ## 4. Frictions on both legs

# %%
rows = []
for friction in [0.0, 0.5, 1.0, 2.0]:
    f = backtest.run_schedule(one_day(prices), "pred_xgb", friction, 0.0, SMALL)
    rows.append({
        "friction_per_mwh": friction,
        "mwh_moved": f["energy_mwh"].sum(),
        "friction_paid": f["friction_cost"].sum(),
        "net_pnl": f["cashflow"].sum(),
    })
pd.DataFrame(rows)

# %% [markdown]
# Four periods × 0.5 MWh = 2 MWh moved, so a 1.00 GBP/MWh friction costs 2.00 GBP. The engine
# charges on **energy moved**, on both the buy and the sell — a round trip pays twice, which is
# what actually happens.

# %% [markdown]
# ## 5. Daily aggregation, and why the unit of decision matters
#
# The battery completes one cycle per day from one forecast. The four charge periods are **one
# decision**, not four bets.
#
# Treating them as four independent observations is what produced a Sharpe of 50 in the first
# version of this project (Lesson 11). Aggregating to daily P&L before computing any risk
# statistic makes the unit of analysis match the unit of decision.

# %%
days = pd.concat([one_day(prices, date=d) for d in ("2024-01-01", "2024-01-02", "2024-01-03")],
                 ignore_index=True)
frame = backtest.run_schedule(days, "pred_xgb", 0.0, 0.0, SMALL)
daily = backtest.daily_pnl(frame)

print(f"period-level rows: {len(frame)}")
print(f"daily rows       : {len(daily)}")
print(f"totals agree     : {np.isclose(daily['pnl'].sum(), frame['cashflow'].sum())}")
daily

# %% [markdown]
# ## 6. No compounding
#
# `cum_pnl` is a cumulative **sum**, not a product.
#
# Compounding requires a capital base to compound against, and this strategy does not define
# one — it is quoted per MW of battery capacity. Inventing a notional in order to draw an
# exponential equity curve would be a fabrication, and exponential curves are exactly what make
# mediocre backtests look impressive.

# %%
real = pd.read_parquet(DATA_PROCESSED / "backtest.parquet")
times = pd.to_datetime(real["start_time"])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(times, real["cum_pnl"], lw=1.6, label="additive (what we report)")
compounded = (1 + real["pnl"] / 10000).cumprod() * 10000 - 10000
ax.plot(times, compounded, lw=1.4, ls="--", color="crimson",
        label="compounded on an invented 10k base (do not do this)")
ax.set(ylabel="cumulative P&L (GBP)", title="Additive vs compounded")
ax.legend()
plt.show()

# %% [markdown]
# The compounded curve bends upward and tells a more exciting story about the same trades. The
# only difference is an assumption nobody stated.

# %% [markdown]
# ## 7. No position carried across the walk-forward boundary
#
# Each fold's predictions are independent. The battery starts and ends every day empty, so
# there is no state to carry — which also means there is no way for information to leak across
# a fold boundary through an inherited position.
#
# That is a design property, not a coincidence, and it is checked:

# %%
by_day = real.copy()
schedule = signals.build(predictions, config=BatteryConfig())
balance = schedule.groupby("settlement_date")["position"].sum().abs().max()
print(f"maximum net position over any day: {balance:.6f} MW  (must be 0)")

# %% [markdown]
# ## 8. Sensitivity: does the result depend on my assumptions?
#
# A backtest that only works at one parameter setting is a curve fit.

# %%
grid = []
for efficiency in [0.75, 0.85, 0.95]:
    for friction in [0.25, 0.75, 2.0]:
        config = BatteryConfig(round_trip_efficiency=efficiency)
        frame = backtest.run_schedule(predictions, "pred_xgb", friction / 2, friction / 2, config)
        daily = backtest.daily_pnl(frame)
        grid.append({"efficiency": efficiency, "friction": friction,
                     "total_pnl": daily["pnl"].sum(), "sharpe": risk.sharpe(daily["pnl"])})
sensitivity = pd.DataFrame(grid)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
pivot_pnl = sensitivity.pivot(index="efficiency", columns="friction", values="total_pnl")
sns.heatmap(pivot_pnl, annot=True, fmt=",.0f", cmap="RdYlGn", ax=ax1)
ax1.set(title="Total P&L (GBP)")

pivot_sharpe = sensitivity.pivot(index="efficiency", columns="friction", values="sharpe")
sns.heatmap(pivot_sharpe, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax2)
ax2.set(title="Sharpe")
plt.tight_layout()
plt.show()

# %% [markdown]
# Profitable across the whole grid. That is reassuring — the result is driven by a structural
# feature of the market (a large daily spread) rather than by a fortunate choice of assumption.
#
# It also tells you where the strategy would break: efficiency matters far more than trading
# frictions, which is why battery chemistry and degradation are the real commercial risk, not
# the spread you pay a broker.

# %% [markdown]
# ## Exercises
#
# 1. Add a per-MWh degradation charge and redo the heatmap. At what level does the strategy stop
#    being worthwhile?
# 2. Allow the battery to carry charge overnight (relax the daily energy balance). Does P&L
#    improve enough to justify the extra state, and does it open a leakage surface?
# 3. Implement a bid-curve version: instead of committing to four periods, submit a price-quantity
#    curve and fill only where the clearing price crosses it. What extra data would you need?
#
# ## Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 12: backtest engine, hand-checked mechanics, sensitivity"
# ```
#
# ---
#
# **Next:** `13-risk-and-reporting.ipynb` — the statistics, and the ones that are easy to get wrong.
