# Worked solutions

Answers to the exercises at the end of each lesson. Where a number is given, it came from
running the code against the cached data — if you re-run and get something different, trust
your run.

Several solutions end in "measure it before deciding". That is not evasion. Most of what
separates a backtest from a story is having written down, in advance, what result would make
you throw the idea away.

---

## Lesson 00 — How the GB power market works

**1. Settlement period for `2024-01-15T00:30:00Z`.** In January the UK is on GMT, so UTC and
local time coincide. Period 1 covers 00:00–00:30 local, so 00:30 is the start of **period 2**.

On 2024-10-27, the clock goes back at 02:00 BST. That day has **50 periods**: the hour
01:00–02:00 local occurs twice, so periods 3, 4, 5 and 6 all map to local times that repeat.
This is exactly why `calendar.py` derives the period count from actual UTC offsets rather than
assuming 48 — and why a naive `hour * 2 + 1` conversion silently corrupts that day.

**2. Negative prices and `log(price)`.** GB day-ahead prices go negative regularly now,
typically on windy low-demand nights and increasingly on sunny spring weekends. `log(price)`
would be undefined — `NaN` for every negative period, silently dropping exactly the periods a
battery most wants to know about, since a negative price is when charging *pays you*. This is
the single strongest argument against the reflexive log-transform of a price series, and it is
why nothing in this repo takes a log.

**3. The information set at gate closure.** Something like:

> At 11:00 on day D, I know: every settled price up to and including the most recent published
> period on D, demand and generation outturn up to the same point, the weather *forecast* for
> D+1, and the calendar.

I do **not** know: any price on D+1, any outturn weather on D+1, or the later periods of D
itself. Write this on paper. Lesson 08 checks every feature against it and lesson 09 turns it
into a test.

---

## Lesson 01 — Ingesting prices

**1. `N2EXMIDP` zeros.** The N2EX series returns zeros for the whole of 2019 and becomes usable
in early 2020. This is why the repo uses `APXMIDP` throughout. The instructive part is *how the
failure presents*: not an error, not a gap, but a plausible-looking series of zeros that a
`.dropna()` will not catch and a mean will happily average in. Always plot a new series before
modelling it.

**2. The highest-priced period.** It lands in the 2021-22 gas crisis, on a still, cold winter
evening — low wind output, high demand, and the marginal plant a gas peaker with a fuel cost
that had itself gone up several hundred percent. The point of looking it up is that the extreme
is physically explicable rather than a data error, which is what earns it the right to stay in
the sample.

**3. Negative-price share by year.** It rises steadily, driven by growing wind and solar
capacity that bids at or below zero (renewables with support contracts will pay to generate).
For a battery this is unambiguously good: revenue is the *spread*, and a negative floor widens
the spread without needing the peak to rise at all.

---

## Lesson 02 — Demand and weather

**1. Wind speed against embedded wind.** The raw correlation is moderate and clearly non-linear
on a scatter plot. `wind_speed ** 3` improves it markedly, because the power in a moving fluid
goes as the cube of velocity — this is physics, not curve-fitting. The relationship flattens at
the top from rated-power limiting, and drops to zero above cut-out speed, so even the cube is
an approximation of a sigmoid-with-a-cliff.

**2. Lowest residual load.** Windy, mild, low-demand periods — typically overnight on a spring
weekend. Prices in those periods are at or below zero. Residual load is the single most
economically meaningful engineered feature in the repo precisely because it puts demand and
weather on the same axis: what the dispatchable fleet actually has to serve.

**3. Dropping the offshore site.** Skill falls, but by less than you would guess, because the
sites are strongly correlated with each other — a windy day in the North Sea is usually a windy
day in the Irish Sea. The gradient-boosted model routes around a missing correlate easily. The
lesson generalises: feature ablation on correlated inputs measures *marginal* value, not total
value, and reporting it as "this feature was worth X" overstates the case for whichever
correlate you happened to remove.

---

## Lesson 03 — The settlement calendar

**1. Positional shift versus merge-based lag.** The two agree on every ordinary day and diverge
on the two clock-change days a year — and, critically, on every row whose lag window *crosses*
one. A 48-period positional shift is a 48-period shift; on a 50-period day it points at the
wrong clock time, and on a 46-period day it overshoots. The differing rows cluster tightly
around late March and late October.

The number of affected rows is small. That is what makes the bug dangerous: it is far too small
to show up in an aggregate MAE, and perfectly capable of producing a wrong answer on the day
somebody actually asks about.

**2. A 7-day rolling mean across the October transition.** The error is a few GBP/MWh on
affected rows — small in absolute terms, and systematically wrong rather than noisy, which is
worse. A systematic error does not average out across a backtest; it biases every fold that
contains one of those days.

**3. If the UK abolished DST in 2027.** **Nothing in `calendar.py` would need to change**, and
that is the whole design argument. The module never hard-codes 48, never hard-codes when the
clocks change, and never assumes a fixed offset. It derives the period count from the actual
UTC offsets of the day in question, via the tz database. Abolishing DST simply means every day
returns 48, and the code that computes that is the same code. Correctness under a rule change
you did not anticipate is the strongest evidence that you modelled the domain rather than the
symptom.

---

## Lesson 04 — Assembling the panel

**1. A deliberately broken calendar.** Shift every timestamp by an hour and the solar-at-night
check fires immediately — you see non-zero embedded solar in periods that are firmly dark. It
detects the full one-hour shift, which is the point: the check exists because a silent join
error looks like nothing at all, and this gives it something loud to trip on. Quality checks
should be written against a corruption you can inject.

**2. Interpolating missing prices.** MAE improves, and the improvement is **not real**. You are
scoring the model partly on values you invented, and linear interpolation invents values that
are unusually easy to predict because they lie on a straight line between two known points.
The correct treatment is what the repo does: leave them missing, let XGBoost learn a default
direction, and drop rows with no *target*.

**3. `min_count=0` on a 2019 row.** It would return **0.0** for interconnectors that did not
exist yet, rather than `NaN`. That is a factual claim that Viking and Greenlink were flowing
zero megawatts in 2019, when the truth is they were not built. `min_count=1` keeps it `NaN`,
which the model handles natively and which does not assert something false. "Absent" and "zero"
are different, and conflating them is one of the most common quiet errors in panel assembly.

---

## Lesson 05 — Exploratory analysis

**1. Intraday profile by regime.** The evening peak stays in roughly the same place — it is
driven by human behaviour, which did not change — but grows dramatically in *amplitude* during
the crisis, and the overnight trough deepens as renewables grow. The battery's opportunity is
the amplitude, so the shape being stable while the scale moves is exactly the regime story.

**2. Extreme periods above the 99.9th percentile.** Heavily concentrated in the evening peak
and in winter, and heavily concentrated in the crisis window. This is why RMSE and MAE tell
different stories here, and why the model optimises absolute error: a handful of scarcity hours
dominate squared loss and would drag the model towards being better at four hours a year and
worse the rest of the time.

**3. Average daily peak-to-trough spread by year.** It grows. That is the battery's revenue
opportunity, and it is growing for a structural reason — more intermittent generation means
more hours where the marginal plant is very cheap and more where it is very expensive.
Extrapolating it is a different and much harder question.

---

## Lesson 06 — Baselines

**1. Mean of T−7 and T−14.** It beats both individually, which surprises people and should not:
averaging two roughly unbiased noisy estimates cuts variance without adding bias. It is worth
running because it demonstrates that a "naive" baseline can be strengthened cheaply, and a
model that only beats the *weakest* naive has not shown much.

**2. Weekday/weekend-aware variant.** It helps, because the failure mode of the plain 7-day lag
is precisely the weekday/weekend boundary. Whether it beats the two-day lag depends on the
regime — which is exercise 3's point.

**3. Baseline MAE per regime.** **The ranking does change.** In calm periods the 7-day lag is
competitive; in the crisis, when prices move fast, the shorter two-day lag wins clearly because
recency dominates seasonality when the level is moving. This is the entire argument against
quoting a single improvement number against a single baseline — and it is why this repo quotes
its headline against the *strongest* baseline over the full period, and then also reports the
regime split.

---

## Lesson 07 — Statsmodels

**1. More Fourier harmonics.** The 48-period ripple in the residual ACF shrinks but does not
vanish. Fourier terms model a *smooth* daily cycle; the real profile has kinks — a sharp
evening ramp — which a low-order trigonometric basis cannot reproduce without many harmonics.
At that point you have spent your parameter budget approximating a shape a tree would learn in
two splits.

**2. Forty-eight per-period regressions.** Total MAE improves slightly. You have gained the
ability to let each period have its own coefficients — the evening peak genuinely does respond
differently to wind than 04:00 does. You have lost roughly 48× the data per fit and any ability
to pool strength across neighbouring periods, so the coefficients get noisy at the edges. The
honest summary: it is a variance-for-bias trade that gradient boosting makes better
automatically, by splitting on the period only where splitting helps.

**3. Robust regression (`sm.RLM`).** The coefficients move noticeably, and mostly become *more*
economically sensible — the wind coefficient in particular. OLS is being dragged by scarcity
spikes, which are exactly the observations Huber loss down-weights. If your economic sanity
checks only pass under a robust loss, that is telling you the tail is driving the fit.

---

## Lesson 08 — Features

**1. Removing residual-load features.** Skill drops materially, but far less than the total
improvement — most of the model's edge comes from price history, with residual load adding a
meaningful increment. The lesson to take is about *attribution*: the improvement is not
separable into clean per-feature contributions when features are correlated, and any sentence
of the form "demand information was worth N%" is an overstatement of a marginal effect.

**2. Adding `price_lag_1d`.** MAE improves substantially, and **that number is the size of the
lie you would be telling**. Yesterday's price at the same period is not knowable for all of
tomorrow at gate closure this morning — for the later periods of D+1 you would be using
information from after the decision. It is excluded by design and `PERIOD_LAGS_DAYS` starts at
2 for exactly this reason. Running the exercise is worthwhile precisely because the improvement
is tempting.

**3. Half-hourly outturn weather.** This is the optimistic extreme, and it improves things
further still. The honest number sits between the two bounds the repo already reports: 22.1%
with weather removed entirely, and 29.9% with daily archive aggregates. Closer to the upper end
for temperature, because day-ahead temperature forecasts are genuinely good; closer to the
lower end for wind, because they are not.

---

## Lesson 09 — Walk-forward and leakage

**1. `price_lag_1d_BANNED`.** See lesson 08 solution 2. Measure it, write the number down, and
then remember that any published forecasting result that does not tell you its information set
may have that number baked in.

**2. Rolling window instead of expanding.** Performance during the crisis improves — a rolling
window discards pre-crisis data that no longer describes the market — and degrades afterwards,
when the extra history would have helped. That is exactly what the expanding-window argument
predicts, and it means the choice is a bet on how stationary you think the future is. The repo
uses expanding because it is the more conservative default and because switching to rolling
*after* seeing that it helped in one regime would be selection on the test set.

**3. Refitting every 7 days.** MAE improves slightly and wall-clock roughly quadruples. Whether
to ship it depends on whether the improvement clears the block-bootstrap interval — and it does
not clearly do so. Paying 4× the compute for a change inside your own error bars is a bad
trade, and being able to say that with a number is the point of the exercise.

---

## Lesson 10 — Honest scoring

**1. Improvement per settlement period.** Largest in the evening peak and the morning ramp —
the periods where the naive baseline's assumption that "last week looked like this" breaks
down, because those are the periods most sensitive to weather and demand. **Yes, it matches
where a battery makes its money**, which is a genuinely favourable result and worth stating:
the model is most useful exactly where the value is.

**2. The DM test with squared loss.** The conclusion does not change — the model still wins,
decisively. Should it change? No, and the more interesting question is why you would run it. If
the two losses disagreed, it would mean the model wins on typical periods and loses on extremes
(or the reverse), which is a real and reportable finding. Running both is cheap insurance
against a headline that only holds under one loss.

**3. Splitting the out-of-sample period in half.** The improvement is *not* constant — it is
larger in the volatile first half. That is not decay from ageing training data; it is that
there was more to be gained when prices moved more. Distinguishing "the model is degrading"
from "the opportunity shrank" requires comparing against the baseline in each half, which is
what the regime table does.

---

## Lesson 11 — The strategy

**1. Degradation cost per MWh.** Published lithium-ion cycle costs land in the range of a few
GBP per MWh throughput. Charging that, the strategy remains worthwhile across most of the
period — but the marginal days stop being worth trading, and total P&L falls by more than the
naive calculation suggests, because the strategy trades on many low-spread days that were only
just profitable. The right response is to raise `min_forecast_spread` until the marginal day
clears its own degradation cost.

**2. Two cycles a day.** Total P&L rises and Sharpe *falls*. The second cycle captures a
narrower spread — by construction, since the widest spread was taken first — so you are adding
revenue at a worse risk-adjusted rate. The difference between those two answers is the whole
reason to look at both: "more money" and "better strategy" are not the same claim.

**3. The noisy-oracle curve.** This is the most useful exercise in the lesson. Perturb the
oracle's prices with Gaussian noise of increasing σ, schedule from the perturbed prices, and
plot realised P&L against σ. It gives you a P&L-to-forecast-error transfer function, and you
can then read off where the actual model sits — around 60% of oracle. That converts "my MAE is
27" into "my forecast is worth about as much as a perfect forecast blurred by σ ≈ X", which is
the version a trading desk can act on.

---

## Lesson 12 — The backtest engine

**1. Degradation charge and the heatmap.** See lesson 11 solution 1. The heatmap makes the
sensitivity visible: the profitable region shrinks from the low-spread corner inward.

**2. Carrying charge overnight.** P&L improves — some of the best spreads straddle midnight,
particularly overnight troughs before a morning ramp. Whether it justifies the extra state is
arguable; whether it opens a leakage surface is not. It does. Once the battery's state at the
start of day D+1 depends on decisions made on D, you must be scrupulous that those decisions
used only D-1 information, and it becomes possible to write a scheduler that quietly optimises
across the boundary using prices it should not have seen. The daily energy balance is a
constraint that makes an entire class of bug impossible, which is worth real P&L.

**3. A bid-curve version.** You would need the day-ahead auction's clearing mechanism and the
full order book, not the market index price the repo uses. The index is a volume-weighted
average of trades; a bid curve fills against a clearing price. Modelling fills against an
average price would systematically overstate execution, because you would fill at the average
on days when you would actually have been outbid.

---

## Lesson 13 — Risk and reporting

**1. Rolling 90-day Sharpe.** It is not stable — it peaks through the crisis and falls
afterwards. It is tracking the *opportunity* (spread width), not the model's skill. This is the
single best illustration in the repo of why the headline number is the uplift over the naive
schedule rather than the Sharpe: uplift divides out the opportunity, Sharpe does not.

**2. Frictions as a share of gross P&L.** Frictions consume a meaningful minority of gross P&L,
and the break-even friction level is several times the assumed 0.75 GBP/MWh. That margin is
comfortable, which is worth stating explicitly — a strategy whose edge dies at 1.2× assumed
costs is not a strategy, it is a fee estimate.

**3. Sortino.** It is *higher* than Sharpe, and yes, you should expect that with an 87% hit
rate: most periods are positive, so downside deviation is computed from a small number of
observations and comes out below the full standard deviation. The honest reading is that
Sortino flatters a high-hit-rate strategy almost by construction, and quoting it instead of
Sharpe would be choosing the metric after seeing the result.

---

## Lesson 14 — Forecast uncertainty

**1. Sweeping `CALIBRATION_DAYS`.** Short windows (15 days) give a noisy offset that
over-reacts to a recent volatile stretch; long windows (240 days) average across regimes and
under-cover during transitions. There is no value that dominates, and the best value **does**
depend on the regime — shorter is better when volatility is changing fast, longer when it is
stable. 60 days is a compromise chosen before looking at the sweep, which is the only defensible
way to choose it.

**2. Coverage against staleness.** Coverage degrades measurably as the test block ages away
from its calibration window. Within a 30-day refit cycle the degradation is modest; extending
the refit interval to 90 days makes it severe. This is the quantitative argument for the refit
cadence, and it is a better argument than the one in lesson 09 exercise 3, because it is about
the interval rather than the point forecast.

**3. Adaptive conformal inference.** It does better in the gas-crisis regime, which is exactly
where a fixed offset struggles — ACI updates α online from recent coverage, so it widens during
a volatility spike instead of waiting for the next refit. The cost is that it no longer has the
clean finite-sample guarantee; it has a long-run coverage guarantee instead. That is a good
trade here and it should be stated rather than glossed over.

**4. Asymmetric offsets.** Breach rates are **not** symmetric — the upper bound is breached far
more often than the lower, because prices spike up and floor out near zero. Fitting separate
offsets improves Winkler, and it also makes the interval look more like the actual conditional
distribution, which is skewed. This is a real improvement and the reason it is an exercise
rather than shipped code is that it should be designed in rather than bolted on.

**5. The interval-aware trading rule.** The discipline is the exercise. Write down first: "I
keep it if uplift over the current rule exceeds X% *and* the days it declines to trade have
below-average realised spread." Then run it. If you decide what counts as success after seeing
the P&L, you have not tested anything — which is precisely how the Sharpe-of-50 strategy in
lesson 11 survived as long as it did.

---

## Lesson 15 — Reproducibility

**1. A CI gate on `mae_improvement_vs_naive`.** Write it, and do not make it blocking. Run-to-
run variation from XGBoost's non-determinism is a fraction of a percentage point, so a 3-point
gate would not fire spuriously — but a legitimate change to the feature set or the study window
would trip it, and the fix would be to edit the recorded baseline, which is a ritual rather than
a control. Make it a PR annotation. Keep the *blocking* checks on things with no sampling error
in them: the leakage tests and the calendar tests.

**2. Caching pip and the fixture panel.** The pip cache saves the bulk of CI wall-clock and is
low-risk, keyed on a hash of `requirements.txt`. Caching the fixture panel is where the danger
is: if the key misses an input — `calendar.py`, say, or `assemble.py` — CI will validate a stale
panel and report success, which defeats the entire purpose of running the pipeline in CI. Cache
the dependencies; recompute the data.
