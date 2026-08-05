# %% [markdown]
# # Lesson 03 — The settlement-period calendar
#
# **You will end with:** a correct (date, period) ↔ UTC conversion, and an understanding of
# the single most common way GB power datasets are silently corrupted.
#
# This is the signature lesson of the project. If an interviewer asks you one detailed
# question about this repo, there is a good chance it is this one — because most candidates
# have never thought about it.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from ppa.config import PERIODS_PER_NORMAL_DAY
from ppa.data import calendar as cal
from ppa.ingest import elexon, neso

print(f"market timezone: {cal.TZ}")
print(f"periods in a normal day: {PERIODS_PER_NORMAL_DAY}")

# %% [markdown]
# ## 1. The problem, stated precisely
#
# GB settles electricity in half-hour blocks called **settlement periods**, numbered from 1,
# where period 1 begins at 00:00 **local time**.
#
# Local. Not UTC. And the UK observes British Summer Time, so:
#
# | Day | Local hours | Periods |
# |---|---|---|
# | Normal | 24 | 48 |
# | Last Sunday in March | 23 (01:00 → 02:00) | **46** |
# | Last Sunday in October | 25 (01:00–02:00 twice) | **50** |
#
# `periods_in_day` derives this from the actual UTC offset at consecutive midnights, rather
# than from a hardcoded table of transition dates — so it stays correct if the UK ever changes
# its DST rules (which Parliament has debated more than once).

# %%
transitions = cal.transition_days("2019-01-01", "2025-06-30")
transitions

# %%
for date in ["2024-01-15", "2024-03-31", "2024-06-15", "2024-10-27"]:
    print(f"{date}  ->  {cal.periods_in_day(date)} periods"
          f"{'   <-- clock change' if cal.is_transition_day(date) else ''}")

# %% [markdown]
# ## 2. Why this corrupts data rather than crashing it
#
# Our two sources express time differently:
#
# - **Elexon** returns a UTC `startTime` per record.
# - **NESO** returns `SETTLEMENT_DATE` (a local date) plus `SETTLEMENT_PERIOD` (1–50).
#
# The obvious join is to build a timestamp from the NESO pair and match it to Elexon's. And
# the obvious way to build that timestamp is:
#
# ```python
# timestamp = pd.Timestamp(date) + pd.Timedelta(minutes=30 * (period - 1))   # WRONG
# ```
#
# That treats the local date as though it were UTC. Let us measure exactly how wrong it is.

# %%
def naive_timestamp(date: str, period: int) -> pd.Timestamp:
    """The tempting, wrong conversion."""
    return pd.Timestamp(date, tz="UTC") + pd.Timedelta(minutes=30 * (period - 1))


rows = []
for date in ["2024-01-15", "2024-06-15", "2024-03-31", "2024-10-27"]:
    for period in [1, 20, 40]:
        if period > cal.periods_in_day(date):
            continue
        correct = cal.to_utc(date, period)
        naive = naive_timestamp(date, period)
        rows.append({
            "date": date,
            "period": period,
            "correct_utc": correct,
            "naive_utc": naive,
            "error_hours": (naive - correct).total_seconds() / 3600,
        })
pd.DataFrame(rows)

# %% [markdown]
# **One full hour of error for six months of every year**, and it is not a constant offset —
# it appears and disappears at the transitions. So a lag feature computed as "48 periods ago"
# is misaligned by an hour for half the year and by a *different* amount around each clock
# change.
#
# Nothing raises. Nothing looks odd. You get a perfectly ordinary DataFrame that is wrong.

# %% [markdown]
# ## 3. The spring day: 46 periods
#
# At 01:00 BST begins, clocks jump to 02:00, and that hour never exists. Periods 1 and 2
# cover 00:00–01:00, and period **3** is 02:00 local — which in absolute time is exactly 30
# minutes after period 2 ended.

# %%
spring = "2024-03-31"
frame = pd.DataFrame({
    "period": range(1, cal.periods_in_day(spring) + 1),
    "utc": [cal.to_utc(spring, p) for p in range(1, cal.periods_in_day(spring) + 1)],
})
frame["local"] = frame["utc"].dt.tz_convert(cal.TZ)
frame["gap_from_previous"] = frame["utc"].diff()
frame.head(6)

# %% [markdown]
# Read the `local` column: 00:00, 00:30, **02:00**, 02:30. The wall clock jumps. Read
# `gap_from_previous`: every step is 30 minutes. Absolute time is continuous; only the label
# jumps.
#
# That distinction is the whole trick, and it is why `to_utc` works by adding a timedelta to a
# tz-aware instant rather than constructing a local wall-clock time and localising it. The
# latter hits a *nonexistent* time in spring and an *ambiguous* one in autumn.

# %% [markdown]
# ## 4. The autumn day: 50 periods
#
# The harder one. 01:00–02:00 happens twice, so a naive implementation produces two periods
# with the same local label and pandas cannot tell them apart.

# %%
autumn = "2024-10-27"
frame = pd.DataFrame({
    "period": range(1, cal.periods_in_day(autumn) + 1),
    "utc": [cal.to_utc(autumn, p) for p in range(1, cal.periods_in_day(autumn) + 1)],
})
frame["local"] = frame["utc"].dt.tz_convert(cal.TZ)
frame["utc_offset"] = frame["local"].apply(lambda t: t.utcoffset().total_seconds() / 3600)
frame.iloc[1:8]

# %% [markdown]
# Periods 3–4 and 5–6 carry the same local wall-clock times, distinguished only by the UTC
# offset flipping from +1 to 0. Our conversion numbers them 1..50 in absolute-time order and
# never repeats.
#
# Notice also that **period 1 of 27 October starts at 23:00 UTC on 26 October** — the
# settlement day begins on the previous calendar day in UTC. Any code that assumes "the
# settlement date equals the UTC date" is broken for six months a year.

# %%
print(f"SP1  on {autumn}: {cal.to_utc(autumn, 1)}  (UTC)")
print(f"SP50 on {autumn}: {cal.to_utc(autumn, 50)} (UTC)")
print(f"SP1  on 2024-10-28: {cal.to_utc('2024-10-28', 1)} (UTC)")

# %% [markdown]
# ## 5. The property that must hold
#
# Whatever the implementation, these must be true:
#
# 1. Consecutive periods are exactly 30 minutes apart in **absolute** time.
# 2. The last period of a day and the first of the next are 30 minutes apart.
# 3. `from_utc(to_utc(d, p)) == (d, p)` for every valid pair.
#
# Property 2 is where an off-by-one hour shows up, so check it at the transitions.

# %%
for date, following in [("2024-03-31", "2024-04-01"), ("2024-10-27", "2024-10-28")]:
    stamps = [cal.to_utc(date, p) for p in range(1, cal.periods_in_day(date) + 1)]
    gaps = {(b - a) for a, b in zip(stamps, stamps[1:], strict=False)}
    boundary = cal.to_utc(following, 1) - stamps[-1]
    roundtrip = all(cal.from_utc(s) == (date, i + 1) for i, s in enumerate(stamps))
    print(f"{date}: {len(stamps)} periods | internal gaps {gaps} | "
          f"boundary {boundary} | round-trip {roundtrip}")

# %% [markdown]
# All three hold. These are exactly the assertions in `tests/test_calendar.py`, which is where
# they belong — a notebook that checks something once is a demonstration, a test that checks
# it on every commit is an invariant.

# %% [markdown]
# ## 6. The trap that hides the bug
#
# Here is why this survives casual inspection: **46 + 50 = 96 = 2 × 48.**
#
# The two transitions cancel. A yearly row count is exactly `365 × 48`, so every sanity check
# based on totals passes while the individual days are wrong.

# %%
grid = cal.expected_periods("2023-01-01", "2023-12-31")
counts = grid.groupby("settlement_date").size()

print(f"total periods in 2023: {len(grid):,}  ({365 * 48:,} if every day had 48)")
print(f"days with 46: {(counts == 46).sum()}   with 48: {(counts == 48).sum()}   "
      f"with 50: {(counts == 50).sum()}")

# %% [markdown]
# ## 7. Does the real data agree?
#
# The strongest possible check: does Elexon's own (date, period) labelling match our
# conversion of its UTC timestamps?

# %%
prices = elexon.load()
sample = prices[prices["settlement_date"].isin(["2024-03-31", "2024-10-27", "2024-06-15"])]

check = sample.copy()
check["our_utc"] = [
    cal.to_utc(d, p) for d, p in zip(check["settlement_date"], check["settlement_period"], strict=True)
]
check["matches"] = check["our_utc"] == check["start_time"]

print(check.groupby("settlement_date")["matches"].agg(["sum", "size"]))
print(f"\noverall agreement: {check['matches'].mean():.1%}")

# %% [markdown]
# Our calendar reproduces the market operator's labelling exactly, including on both clock
# changes. That is the validation that matters — not that the code is self-consistent, but
# that it agrees with the institution that defines the convention.

# %%
demand = neso.load()
counts = demand.groupby("settlement_date").size()
odd = counts[counts != 48]
print(f"NESO days not equal to 48 periods: {len(odd)}")
print(odd.head(10).to_string())

# %% [markdown]
# NESO's own file shows the same 46/50 structure. Both sources agree with each other and with
# `calendar.py`.

# %% [markdown]
# ## 8. Exercises
#
# 1. Implement the naive conversion, build a full panel with it, and compute
#    `price_lag_48` (positional shift) versus our merge-based `price_lag_2d`. How many rows
#    differ, and where are they concentrated?
# 2. What happens to a 7-day rolling mean computed by positional shift across the October
#    transition? Quantify the error in GBP/MWh.
# 3. The EU has periodically debated abolishing DST. If the UK dropped it in 2027, which
#    functions in `calendar.py` would need changing? (Answer: none — and explain why.)
#
# ## 9. Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 03: settlement-period calendar, 46/48/50, DST handling"
# ```
#
# ---
#
# **Next:** `04-panel-assembly.ipynb` — joining three sources against an expected grid so that
# a missing period is reported rather than silently interpolated.
