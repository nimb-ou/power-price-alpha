# %% [markdown]
# # Lesson 02 — Demand, embedded renewables, and residual load
#
# **You will end with:** the feature that matters more than any price lag, and an understanding
# of why "demand" in GB is not what it sounds like.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from ppa.config import WEATHER_SITES
from ppa.ingest import elexon, neso, openmeteo

demand = neso.load()
print(f"{len(demand):,} half-hourly demand records")
print(f"columns: {list(demand.columns)}")

# %% [markdown]
# ## 1. "National demand" is not total demand
#
# NESO's `ND` measures demand **on the transmission system**. Generation connected to the
# *distribution* network — rooftop solar, small wind farms, the ~30 GW of "embedded"
# generation in GB — never appears as generation at all. It appears as demand that failed to
# materialise.
#
# So on a sunny afternoon, measured demand falls. Not because anyone used less electricity, but
# because a few gigawatts of it came from roofs that the transmission operator cannot see.
#
# **Residual load** is what is left for the wholesale market to actually price:
#
# ```
# residual_load = national_demand − embedded_wind − embedded_solar
# ```
#
# That quantity determines which generator is marginal, and the marginal generator sets the
# price. It is the single most useful non-price feature in this project.

# %%
demand["residual_load"] = (
    demand["national_demand"] - demand["embedded_wind"] - demand["embedded_solar"]
)

sample = demand[demand["settlement_date"].between("2024-06-10", "2024-06-16")].copy()
sample["t"] = range(len(sample))

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(sample["t"], sample["national_demand"], lw=1.6, label="national demand (ND)")
ax.plot(sample["t"], sample["residual_load"], lw=1.6, label="residual load")
ax.fill_between(sample["t"], sample["residual_load"], sample["national_demand"],
                alpha=0.25, color="orange", label="embedded wind + solar")
ax.set(xlabel="half-hours (one week in June 2024)", ylabel="MW",
       title="Embedded generation is invisible as generation")
ax.legend()
plt.show()

# %% [markdown]
# The orange band is largest in the middle of each day — that is solar. Notice that residual
# load has a *deeper midday dip* than measured demand, which is exactly the shape that drives
# negative midday prices in spring.

# %%
demand["period"] = demand["settlement_period"]
profile = demand.groupby("period")[["national_demand", "residual_load", "embedded_solar",
                                    "embedded_wind"]].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
ax1.plot(profile.index, profile["national_demand"], lw=2, label="national demand")
ax1.plot(profile.index, profile["residual_load"], lw=2, label="residual load")
ax1.set(xlabel="settlement period", ylabel="MW", title="Average daily shape")
ax1.legend()

ax2.plot(profile.index, profile["embedded_solar"], lw=2, label="embedded solar")
ax2.plot(profile.index, profile["embedded_wind"], lw=2, label="embedded wind")
ax2.set(xlabel="settlement period", ylabel="MW", title="Embedded generation")
ax2.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Does residual load actually predict price?

# %%
prices = elexon.load()
merged = prices.merge(demand, on=["settlement_date", "settlement_period"], how="inner")

correlations = merged[
    ["price", "national_demand", "residual_load", "embedded_wind", "embedded_solar"]
].corr()["price"].drop("price")
correlations.sort_values(ascending=False).to_frame("correlation with price")

# %% [markdown]
# Residual load correlates more strongly with price than raw demand does — which is the
# economic story working: the market prices what is left after free renewables, not total
# consumption.

# %%
recent = merged[merged["settlement_date"] >= "2023-01-01"]
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(recent["residual_load"], recent["price"], s=3, alpha=0.15,
                c=recent["settlement_period"], cmap="twilight")
ax.axhline(0, color="black", lw=0.8)
ax.set(xlabel="residual load (MW)", ylabel="price (GBP/MWh)",
       title="The merit-order curve, visible in the data")
plt.colorbar(sc, ax=ax, label="settlement period")
plt.show()

# %% [markdown]
# That upward-sloping cloud is the **merit order** — the stack of generators ordered by
# marginal cost. Low residual load means cheap plant sets the price; high residual load pulls
# in expensive peakers. The scatter is wide because fuel prices moved by an order of magnitude
# over the window, which shifts the whole curve up and down.
#
# Note the negative prices at the far left: very low residual load, and the system is paying
# to absorb surplus.

# %% [markdown]
# ## 3. Interconnectors
#
# GB imports and exports through nine subsea links. Imports displace domestic generation, which
# changes which plant is marginal, which changes the price. `neso.py` keeps all the flow
# columns — noting that Viking and Greenlink only exist from 2023/24, so the ingest selects
# columns that are present rather than assuming a fixed schema.

# %%
flows = [c for c in demand.columns if c.endswith("_flow")]
availability = demand.groupby(demand["settlement_date"].str[:4])[flows].apply(
    lambda g: g.notna().mean()
)
availability.round(2)

# %% [markdown]
# ## 4. Weather
#
# Six sites: five population centres plus an offshore point near the main wind farms. One
# location will not do — a windy North Sea with a calm south-east is a completely different
# market state from the reverse, and a single-point average hides that.

# %%
for site, (lat, lon) in WEATHER_SITES.items():
    print(f"{site:14} {lat:6.2f}, {lon:6.2f}")

weather = openmeteo.load()
national = openmeteo.to_national(weather)
print(f"\n{len(weather):,} site-hours -> {len(national):,} national hours")
national.head(3)

# %% [markdown]
# ### Wind speed at 100 m, not 10 m
#
# The default weather variable is `wind_speed_10m`. Turbine hubs sit at 80–120 m, and wind
# speed increases with height. More importantly, turbine power output goes as the **cube** of
# wind speed, so a modest error in speed is a large error in generation.

# %%
merged_w = demand.merge(
    national.assign(hour=national["time"]),
    left_on=None, right_on=None, how="left", left_index=True, right_index=True
).head(0)  # placeholder; the real join happens in assemble.py

sample_hours = national.set_index("time").loc["2024-01-01":"2024-12-31"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
ax1.hist(sample_hours["wind_speed_mean"], bins=50, color="#4C72B0")
ax1.set(xlabel="mean wind speed @100m (km/h)", ylabel="hours", title="Wind speed, 2024")
ax2.scatter(sample_hours["temperature_mean"], sample_hours["heating_degrees"], s=4, alpha=0.3)
ax2.set(xlabel="mean temperature (C)", ylabel="heating degrees",
        title="Degree-hours linearise a V-shaped relationship")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Degree-hours
#
# The relationship between temperature and demand is **V-shaped**: demand rises when it is cold
# (heating) *and* when it is hot (cooling). A single linear temperature coefficient cannot
# express that — it would have to be positive and negative at once.
#
# Splitting into `heating_degrees = max(15.5 − T, 0)` and `cooling_degrees = max(T − 15.5, 0)`
# makes each side linear again. 15.5 °C is the UK convention.
#
# Trees could in principle learn the V themselves, but giving them the right basis costs
# nothing and makes the OLS benchmark in Lesson 07 competitive rather than a straw man.

# %% [markdown]
# ## 5. The honest caveat, stated at the source
#
# This is the **archive** API. It returns the weather that actually happened.
#
# A genuine day-ahead forecast would use the weather *forecast* available at gate closure,
# which is less accurate — particularly for wind. So using outturn weather makes the model
# look better than a real deployment would be.
#
# The project handles this three ways rather than hoping nobody notices:
#
# 1. `features/build.py` degrades weather to **daily aggregates** rather than half-hourly
#    outturn, which is much closer to the information content of a real forecast;
# 2. the limitation is stated in the module docstring, the README and `RESUME_CLAIMS.md`;
# 3. `make forecast-ablation` re-runs the entire walk-forward with **every weather feature
#    removed**, so the size of the advantage is measured rather than argued about.
#
# Lesson 10 reports both numbers.

# %% [markdown]
# ## 6. Exercises
#
# 1. Compute the correlation between `wind_speed_mean` and `embedded_wind`. Is it linear?
#    Try `wind_speed_mean ** 3` and explain the improvement.
# 2. Find the periods with the lowest residual load in the window. What were prices doing?
# 3. Drop the offshore site from `WEATHER_SITES`, rebuild, and re-run Lesson 09. How much of
#    the model's skill came from that one location?
#
# ## 7. Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 02: demand, embedded generation, residual load, weather"
# ```
#
# ---
#
# **Next:** `03-settlement-calendar.ipynb` — the arithmetic that has to be right before any of
# this can be joined.
