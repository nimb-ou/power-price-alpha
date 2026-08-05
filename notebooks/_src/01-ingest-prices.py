# %% [markdown]
# # Lesson 01 — Ingesting GB prices: caching, limits, and a trap
#
# **You will end with:** six and a half years of half-hourly GB prices cached locally, and two
# lessons about public APIs that you only learn by using them.

# %%
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from ppa.config import ELEXON_BASE, ELEXON_DATA_PROVIDER, STUDY
from ppa.ingest import elexon
from ppa.ingest.cache import fixed_chunks

print(f"endpoint : {ELEXON_BASE}/balancing/pricing/market-index")
print(f"provider : {ELEXON_DATA_PROVIDER}")
print(f"window   : {STUDY.start} .. {STUDY.end}")

# %% [markdown]
# ## 1. Which price series?
#
# GB has several electricity prices and they are not interchangeable (see
# `course/00-gb-power-market.md`). We use Elexon's **Market Index Data** — a half-hourly,
# volume-weighted reference price computed from actual short-term trades.
#
# It has two `dataProvider` values: `APXMIDP` and `N2EXMIDP`. They look equivalent in the
# documentation. They are not.

# %%
def probe(provider: str, date: str) -> pd.DataFrame:
    response = requests.get(
        f"{ELEXON_BASE}/balancing/pricing/market-index",
        params={
            "from": f"{date}T00:00Z",
            "to": f"{date}T06:00Z",
            "dataProviders": provider,
            "format": "json",
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    return pd.DataFrame(data)[["settlementPeriod", "price", "volume"]] if data else pd.DataFrame()


for provider in ("APXMIDP", "N2EXMIDP"):
    for date in ("2019-06-15", "2024-06-15"):
        frame = probe(provider, date)
        summary = (
            f"n={len(frame):2d}  mean price {frame['price'].mean():7.2f}  "
            f"mean volume {frame['volume'].mean():8.1f}"
            if not frame.empty
            else "no data"
        )
        print(f"{provider:10} {date}  {summary}")

# %% [markdown]
# **`N2EXMIDP` returns zero prices in 2019.** Not missing — *zero*. Every field is populated,
# the response is a perfectly valid 200, and the numbers are wrong.
#
# Train on that and you teach the model that electricity was free for a year, and that the
# free period ended abruptly in 2020. The model will faithfully learn a structural break that
# never happened.
#
# Nothing in the API documentation warns you. You find it by plotting the data before
# modelling it — which is the actual lesson, and it is why `config.py` pins `APXMIDP` with a
# comment explaining why rather than leaving it as an unexplained default.

# %% [markdown]
# ## 2. The undocumented range limit

# %%
for days in (7, 8, 31):
    end = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    response = requests.get(
        f"{ELEXON_BASE}/balancing/pricing/market-index",
        params={"from": "2024-01-01T00:00Z", "to": f"{end}T00:00Z",
                "dataProviders": "APXMIDP", "format": "json"},
        timeout=60,
    )
    detail = ""
    if response.status_code != 200:
        errors = response.json().get("errors", {})
        detail = "  -> " + str(list(errors.values())[0][0] if errors else "")
    print(f"{days:2d}-day range: HTTP {response.status_code}{detail}")

# %% [markdown]
# Seven days maximum, stated **only in the 400 response body**. So 6.5 years is ~340 requests,
# and `fixed_chunks` splits the range accordingly.
#
# The windows are anchored at `start` rather than at calendar boundaries, deliberately: cache
# keys then stay stable when you re-run with a later end date, instead of re-partitioning the
# whole history and refetching everything.

# %%
chunks = fixed_chunks(STUDY.start, STUDY.end, days=7)
print(f"{len(chunks)} requests to cover {STUDY.start} .. {STUDY.end}")
print(f"first: {chunks[0][0].date()} .. {chunks[0][1].date()}")
print(f"last : {chunks[-1][0].date()} .. {chunks[-1][1].date()}")

short = fixed_chunks(STUDY.start, "2020-01-01", days=7)
print(f"\nre-running with an earlier end reuses the same keys: "
      f"{[c[0] for c in short] == [c[0] for c in chunks[:len(short)]]}")

# %% [markdown]
# ## 3. Caching, and why it is not just about speed
#
# `cache.cached()` writes each chunk to parquet and only touches the network when the file is
# missing. Three reasons, and only the first is obvious:
#
# 1. **Speed.** After the first run, `make ingest` is offline and instant.
# 2. **Courtesy.** These are free public services. Refetching 340 windows on every experiment
#    is rude.
# 3. **Reproducibility.** Elexon restates history occasionally. A cached extract means today's
#    result and next month's are computed on the same data — and if you *want* a refresh,
#    `--force` makes that an explicit decision rather than an accident.
#
# Note that empty results are cached too. A window with genuinely no data would otherwise be
# refetched forever.

# %%
prices = elexon.load()
print(f"{len(prices):,} settlement periods cached")
print(f"{prices['start_time'].min()} .. {prices['start_time'].max()}")
prices.head()

# %% [markdown]
# ## 4. What the prices actually look like

# %%
prices["date"] = pd.to_datetime(prices["settlement_date"])
daily = prices.groupby("date")["price"].agg(["mean", "min", "max"])

fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(daily.index, daily["min"], daily["max"], alpha=0.25, label="daily range")
ax.plot(daily.index, daily["mean"], lw=1, label="daily mean")
ax.axvspan(pd.Timestamp("2021-09-01"), pd.Timestamp("2023-06-30"),
           alpha=0.1, color="crimson", label="gas crisis")
ax.axhline(0, color="black", lw=0.8)
ax.set(ylabel="GBP/MWh", title="GB half-hourly market index price")
ax.legend()
plt.show()

# %%
print(f"mean     {prices['price'].mean():8.2f} GBP/MWh")
print(f"median   {prices['price'].median():8.2f}")
print(f"min      {prices['price'].min():8.2f}")
print(f"max      {prices['price'].max():8.2f}")
print(f"negative {(prices['price'] < 0).sum():,} periods ({(prices['price'] < 0).mean():.2%})")
print(f"zero     {(prices['price'] == 0).sum():,} periods")

# %% [markdown]
# ## 5. Two properties that break standard modelling choices
#
# **Prices go negative.** Over 1,700 periods in our window. When must-run wind and nuclear
# exceed demand, generators pay to keep producing rather than shut down and restart.
#
# That kills three habits at once:
#
# - `log(price)` — undefined;
# - MAPE — divides by a number that touches zero;
# - any distributional assumption that requires positive support.
#
# **The scale changes by an order of magnitude.** A model trained only on 2019 has never seen
# a price above about £100; in 2022 the mean was several times that. This is why the study
# window deliberately spans calm, crisis and normalisation, and why the walk-forward uses an
# expanding rather than rolling window.

# %%
yearly = prices.groupby(prices["date"].dt.year)["price"].agg(
    ["mean", "std", "min", "max", lambda s: (s < 0).sum()]
)
yearly.columns = ["mean", "std", "min", "max", "negative_periods"]
yearly.round(1)

# %% [markdown]
# ## 6. Exercises
#
# 1. Plot the 2019 `N2EXMIDP` series against `APXMIDP` for the same month. How long does the
#   zero period last, and when exactly does the series become usable?
# 2. Find the single highest-priced settlement period in the window and look up what happened
#    that day in GB. (Hint: check the wind output and the temperature.)
# 3. Compute the share of negative-price periods by year. What is driving the trend, and what
#    does it imply for a battery's economics?
#
# ## 7. Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 01: Elexon ingest, caching, the N2EXMIDP trap"
# ```
#
# ---
#
# **Next:** `02-ingest-demand-and-weather.ipynb` — the feature that matters more than any price
# lag.
