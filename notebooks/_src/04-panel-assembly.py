# %% [markdown]
# # Lesson 04 — Assembling the panel against an expected grid
#
# **You will end with:** a 113,902-row half-hourly panel, and a quality report that surfaces
# every missing period rather than quietly filling it.

# %%
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from ppa.config import STUDY
from ppa.data import calendar as cal
from ppa.data.assemble import build, load, report_quality

panel = load()
print(f"{len(panel):,} rows x {len(panel.columns)} columns")

# %% [markdown]
# ## 1. Build against a grid, not by joining
#
# The tempting approach is to merge prices onto demand and see what comes out. The problem:
# if a row is missing from *both* sources you never learn it existed.
#
# So the panel starts from `calendar.expected_periods()` — the complete set of (date, period)
# pairs the range *should* contain, honouring 46/48/50 — and everything else is joined onto
# that. A missing period then shows up as a NaN in a row that exists, instead of as a row that
# silently is not there.

# %%
grid = cal.expected_periods(STUDY.start, STUDY.end)
print(f"expected periods : {len(grid):,}")
print(f"panel rows       : {len(panel):,}")
print(f"match            : {len(grid) == len(panel)}")

# %% [markdown]
# ## 2. The cheapest possible validation
#
# Elexon gives us *both* a UTC timestamp and a (date, period) pair. We key on the pair and
# **recompute** the timestamp from our own calendar — then check the two agree.
#
# That single check validates our calendar against the market operator's, for free, on every
# row.

# %%
from ppa.ingest import elexon

prices = elexon.load()
check = prices.copy()
check["our_utc"] = [
    cal.to_utc(d, p)
    for d, p in zip(check["settlement_date"], check["settlement_period"], strict=True)
]
agreement = (check["our_utc"] == check["start_time"]).mean()
print(f"our calendar agrees with Elexon's own labelling on {agreement:.4%} of rows")

# %% [markdown]
# ## 3. The quality report

# %%
report = report_quality(panel)
for key, value in report.items():
    if key != "price_gap_days":
        print(f"{key:32} {value}")

# %% [markdown]
# Zero missing demand, zero missing weather, zero duplicate keys, timestamps strictly
# increasing. The only gaps are in price, at 1.8%.

# %% [markdown]
# ## 4. Characterising the gaps — and rejecting a hypothesis
#
# 1.8% missing is small, but *where* it is matters. Scattered evenly would suggest a join bug;
# concentrated in blocks suggests real API outages.

# %%
gaps = panel[panel["price"].isna()]
per_day = gaps.groupby("settlement_date").size()

print(f"{len(gaps):,} missing periods across {len(per_day)} days")
print("\ndistribution of gaps per affected day:")
print(per_day.value_counts().sort_index().to_string())

whole_day = per_day[per_day >= 46]
print(f"\nwhole-day outages: {len(whole_day)} days = {whole_day.sum():,} rows "
      f"({whole_day.sum() / len(gaps):.1%} of all missing)")

# %% [markdown]
# Bimodal: 36 days lost entirely, plus a scatter of days losing one to three periods. The
# whole-day outages account for 84% of the missing data — those are real Elexon outages (a run
# in January 2019, several in November 2020).
#
# ### A hypothesis, tested and rejected
#
# The obvious explanation for the isolated gaps: MID is a volume-weighted average of actual
# trades, so a period with no trades has no index price — and the thinnest trading is
# overnight. Let us check.

# %%
isolated = gaps[gaps["settlement_date"].isin(per_day[per_day <= 3].index)]
overnight_share = (isolated["settlement_period"] <= 6).mean()

print(f"isolated gaps: {len(isolated)} rows")
print(f"share falling in SP1-6 (overnight): {overnight_share:.1%}")
print(f"share you would get by chance     : {6 / 48:.1%}")

median_volume_all = panel.groupby("settlement_period")["volume"].median().median()
affected = panel.groupby("settlement_period")["volume"].median().loc[
    isolated["settlement_period"].unique()
].median()
print(f"\nmedian volume in affected periods : {affected:,.0f}")
print(f"median volume across all periods  : {median_volume_all:,.0f}")

# %% [markdown]
# **The hypothesis is wrong.** 29% versus 12.5% by chance is a mild tilt, not a pattern, and
# volume in the affected periods is no lower than elsewhere.
#
# So `tests/test_data.py` asserts what is robust — whole-day outages dominate, the residual is
# negligible — and deliberately asserts **nothing** about the shape of those 55 rows. 0.05% of
# the panel is too little to characterise, and inventing a story about it would be exactly the
# small-sample overreach this project criticises elsewhere.
#
# Recording a rejected hypothesis in a docstring is worth doing. It stops the next person
# (including you, in six months) from re-deriving it.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
gap_dates = pd.to_datetime(gaps["settlement_date"])
ax1.hist(gap_dates, bins=80, color="crimson")
ax1.set(title="When the gaps are", ylabel="missing periods")
ax1.tick_params(axis="x", rotation=45)

ax2.hist(isolated["settlement_period"], bins=range(1, 50, 2), color="#4C72B0")
ax2.set(xlabel="settlement period", title="Isolated gaps are spread across the day")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. What we do *not* do
#
# We do not interpolate the missing prices.
#
# It would be easy — a linear fill across a one-period gap is defensible, and a whole-day fill
# from the previous week is arguable. But those rows would then be **targets the model is
# scored on**, and it would be scored on our interpolation rather than on the market.
#
# Instead `features/build.py` drops rows with no target, and reports how many:
#
# ```
# INFO dropped 2060 rows with no target price
# ```
#
# The features for surviving rows may still reference a missing lag; XGBoost handles NaN
# natively by learning a default split direction, so those rows are kept rather than cascading
# the loss.

# %% [markdown]
# ## 6. The joins, and where each could go wrong

# %%
pd.DataFrame([
    {"source": "Elexon prices", "key": "(settlement_date, settlement_period)",
     "risk": "none — same key space as the grid"},
    {"source": "NESO demand", "key": "(settlement_date, settlement_period)",
     "risk": "none — same key space"},
    {"source": "Open-Meteo weather", "key": "UTC hour (floor of start_time)",
     "risk": "start_time must be correct, which is why the calendar comes first"},
])

# %% [markdown]
# The weather join is the one that depends on the calendar being right: it floors each
# settlement period to its UTC hour. Get the timestamp wrong and every period gets the wrong
# hour's weather — a one-hour shift for half the year, silently.
#
# A cheap end-to-end check on that: solar generation must be zero at night.

# %%
local_hour = panel["start_time"].dt.tz_convert(cal.TZ).dt.hour
night = panel[(local_hour >= 23) | (local_hour <= 2)]
print(f"max embedded solar between 23:00 and 02:00 local: "
      f"{night['embedded_solar'].abs().max():.1f} MW")
print(f"max embedded solar at 12:00-13:00 local        : "
      f"{panel[local_hour == 12]['embedded_solar'].max():,.0f} MW")

# %% [markdown]
# Zero at night, several gigawatts at midday. If the timestamps were shifted by an hour this
# would still roughly hold; if they were shifted by twelve it would not. It is a weak check
# that costs nothing, and it is in the test suite.

# %% [markdown]
# ## 7. Exercises
#
# 1. Rebuild the panel with a deliberately broken calendar (add one hour to every timestamp)
#    and re-run the solar-at-night check. How large a shift does it detect?
# 2. Fill the missing prices by linear interpolation, re-run Lesson 09, and compare MAE. Is
#    the improvement real, or are you scoring the model on your own fill?
# 3. `net_imports` sums the interconnector flows with `min_count=1`. What would `min_count=0`
#    do on a 2019 row, where Viking and Greenlink did not exist?
#
# ## 8. Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 04: panel assembly, expected grid, quality report"
# ```
#
# ---
#
# **Next:** `05-eda.ipynb` — what this market actually looks like.
