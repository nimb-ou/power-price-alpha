# Lesson 00 — How the GB power market actually works

**Time:** ~45 minutes. **You will end with:** a working environment, and enough market
structure to know what you're forecasting and why the obvious approach is wrong.

Skipping this lesson is how people produce electricity price models that are subtly
nonsensical. The domain facts below each map to a specific line of code later in the course.

---

## 1. Electricity is not a normal commodity

You cannot store it at scale, and supply must equal demand continuously — not on average,
*continuously*. Two consequences drive everything else:

- **Prices are extremely spiky.** When the last available generator is an expensive
  gas peaker, price jumps by an order of magnitude within a single half hour. Prices also go
  **negative** when there is more must-run wind and nuclear than demand: generators pay to
  keep producing rather than shut down. Any model that assumes log-normal prices, or that
  applies `log(price)`, breaks the moment it meets a negative print.
- **Demand and weather are the state variables.** Not "market sentiment". Cold and dark means
  high demand; windy means abundant cheap supply. The single most predictive feature you will
  build is not a price lag — it is **residual load**: demand minus embedded wind and solar.

## 2. Settlement periods — the 46/48/50 problem

GB settles in **half-hour blocks called settlement periods**, numbered from 1. Period 1 runs
00:00–00:30 **local time**.

A normal day has 48 periods. But GB observes British Summer Time:

| Day | Periods | Why |
|---|---|---|
| Normal day | 48 | 24 h × 2 |
| Last Sunday in March (clocks forward) | **46** | 01:00 jumps to 02:00; one hour never happens |
| Last Sunday in October (clocks back) | **50** | 01:00–02:00 happens twice |

This matters more than it sounds. Our two main sources disagree on how they express time:

- **Elexon** returns a UTC `startTime` per record.
- **NESO** returns `SETTLEMENT_DATE` (a local calendar date) plus `SETTLEMENT_PERIOD` (1–50).

If you join those by naively computing `timestamp = date + (period-1) × 30min`, you produce
wrong timestamps on two days a year, and — much worse — every lag feature computed as "48
periods ago" is silently misaligned across those boundaries. Since we build features on
lags of 48, 96 and 336 periods, that error propagates for a week either side.

`src/ppa/data/calendar.py` exists solely to get this right, and `tests/test_calendar.py`
pins the behaviour on the four known transition days in our window. Lesson 03 builds it.

> **Interview note.** If someone asks you a single question about this project, there is a
> good chance it is "how did you handle the clock change?" Most candidates have not thought
> about it.

## 3. Which price are we forecasting?

There are several GB electricity prices and they are not interchangeable:

| Price | What it is | Set when |
|---|---|---|
| **Day-ahead auction** (N2EX / EPEX) | Clears the bulk of wholesale volume for each period of tomorrow | ~11:00 local on day D for delivery D+1 |
| **Intraday / within-day** | Continuous trading to adjust positions closer to delivery | D to real time |
| **Market Index Price (MID)** | Volume-weighted average of short-term trades in a period, reported by Elexon | After the period |
| **Imbalance price (system price)** | What you pay if your physical position is out | After settlement |

We use Elexon's **MID with `dataProvider = APXMIDP`**: a half-hourly, volume-weighted
reference price for GB short-term power. It is the cleanest keyless public series covering
our whole window at half-hourly granularity.

Be precise about this when you describe the project. The honest sentence is: *"half-hourly GB
market index prices, forecast one day ahead using only information available at day-ahead
gate closure."* That is what the code does. Claiming to forecast the N2EX auction clearing
price specifically would be a slightly different thing, and `course/README` and
`RESUME_CLAIMS.md` both say so.

We also checked `N2EXMIDP`: it returns zero prices before roughly 2020, so using it would
poison the early training window with fake zeros. That is recorded in `STATE.md` under
decisions — the kind of thing you discover by looking at the data rather than the docs.

## 4. The information set, and why it's the whole game

"Day ahead" is a claim about **what you knew when**.

At gate closure on day D (~11:00 local), you are forecasting all 48 periods of day D+1. At
that instant you have:

- Prices up to some cutoff on day D — **not** later on D, and certainly nothing from D+1.
- Demand outturn up to that same cutoff.
- Weather **forecasts** for D+1 — not the outturn we can download today.
- The calendar: it is knowable forever in advance that D+1 is a Tuesday, or a bank holiday.

The seductive bug is using *outturn* weather for D+1, because that is what
`archive-api.open-meteo.com` hands you and it is far more accurate than a real forecast
would have been. Do that and your MAE improvement looks wonderful and means nothing.

We handle this explicitly (Lesson 08), we state the residual assumption honestly in the
README, and `tests/test_no_leakage.py` asserts that no feature carries a source timestamp
later than its decision time.

## 5. The forecast is not the strategy

Beating a naive baseline on MAE is a forecasting result. It is not money. To get from one to
the other you need to say what you would actually *trade*, and at what cost.

Lessons 11–13 are careful about this and equally careful about what the backtest does **not**
model: physical delivery, credit and collateral, market impact, and imbalance exposure. A
backtest that ignores those and reports a Sharpe of 4 is a red flag to anybody who has traded.
The goal here is a modest, defensible number with its confidence interval attached.

## 6. Set up

```bash
cd ~/Desktop/First_project/power-price-alpha
make venv
```

Verify:

```bash
.venv/bin/python -c "from ppa.config import STUDY, PERIODS_PER_NORMAL_DAY; print(STUDY.start, STUDY.end, PERIODS_PER_NORMAL_DAY)"
```

Expected:

```
2019-01-01 2025-06-30 48
```

Now confirm the three data sources are alive from your machine, before writing any client
code against them:

```bash
curl -s "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index?from=2024-01-15T00:00Z&to=2024-01-15T02:00Z&dataProviders=APXMIDP&format=json" | head -c 300
echo
curl -s "https://api.neso.energy/api/3/action/package_show?id=historic-demand-data" | head -c 200
echo
curl -s "https://archive-api.open-meteo.com/v1/archive?latitude=51.51&longitude=-0.13&start_date=2024-01-15&end_date=2024-01-15&hourly=temperature_2m&timezone=UTC" | head -c 200
```

Each should return JSON. None needs a key.

## 7. Exercises before you move on

1. In the Elexon response above, work out which settlement period `2024-01-15T00:30:00Z`
   corresponds to. Then work out what it would be on 2024-10-27 (clock-change day).
2. Look up when GB day-ahead prices last went negative and for how long. What would
   `log(price)` have done?
3. Write down, in one sentence, the exact information set available at gate closure. You will
   check your feature code against that sentence repeatedly.

## 8. Commit checkpoint

```bash
git add -A
git commit -m "Session 0: scaffold repo, market config, Makefile, CI skeleton, lesson 00"
```

---

**Next:** `notebooks/01-ingest-elexon.ipynb` — build a cached, retrying client for the Elexon
MID endpoint and pull six years of half-hourly prices.
