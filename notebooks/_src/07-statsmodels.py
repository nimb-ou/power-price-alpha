# %% [markdown]
# # Lesson 07 — A linear benchmark you can read
#
# **You will end with:** an OLS model whose coefficients you can check against economics, and
# residual diagnostics that tell you what the model has *not* captured.
#
# The point of this lesson is not to beat XGBoost. It is that a tree gives you a number and a
# linear model gives you an argument.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from ppa.config import DATA_PROCESSED
from ppa.data.assemble import load as load_panel
from ppa.eval import metrics
from ppa.features.build import feature_columns
from ppa.models import naive, sarimax

features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
columns = feature_columns(features)
print(f"{len(features):,} rows, {len(columns)} features")

# %% [markdown]
# ## 1. Why bother with a linear model
#
# Three reasons, and only the first is about accuracy:
#
# 1. **It is the literature's benchmark.** Electricity price forecasting has used regression
#    with calendar terms and exogenous load for decades. A gradient-boosting result that has
#    never been compared to one is not comparable to anything published.
# 2. **The coefficients are checkable.** Residual load *must* have a positive coefficient —
#    more demand net of renewables means a more expensive marginal generator. If it comes out
#    negative, something upstream is broken, and no tree importance score tells you that as
#    directly.
# 3. **Residual diagnostics are a debugging tool.** Autocorrelation in the residuals tells you
#    what structure remains. A single MAE does not.

# %% [markdown]
# ## 2. Fit it
#
# Same features the tree sees, so the comparison is like for like. OLS has no native NaN
# handling, so rows with any missing feature are dropped rather than imputed — imputing would
# make it a different model than the one being compared.

# %%
train = features[features["settlement_date"] < "2024-01-01"]
test = features[features["settlement_date"] >= "2024-01-01"]

model, used = sarimax.fit_ols(train, columns)
print(f"trained on {int(model.nobs):,} rows (of {len(train):,} — the rest had missing lags)")
print(f"R-squared: {model.rsquared:.4f}")

# %% [markdown]
# ## 3. Do the coefficients make economic sense?

# %%
coefficients = sarimax.coefficient_table(model)
coefficients.head(15).round(4)

# %%
economic_checks = [
    ("residual_am_mean", "positive", "more residual load -> more expensive marginal plant"),
    ("price_lag_2d", "positive", "prices are persistent"),
    ("price_lag_7d", "positive", "prices are persistent"),
    ("heating_degrees_fcst", "positive", "cold -> heating demand -> higher price"),
    ("wind_speed_mean_fcst", "negative", "windy -> free supply -> lower price"),
]

rows = []
for name, expected, why in economic_checks:
    if name not in model.params.index:
        continue
    value = model.params[name]
    actual = "positive" if value > 0 else "negative"
    rows.append({"feature": name, "coefficient": round(value, 4),
                 "expected": expected, "actual": actual,
                 "agrees": actual == expected, "reasoning": why})
pd.DataFrame(rows)

# %% [markdown]
# Check each row. Where the sign disagrees with the economics, you have found something worth
# understanding — usually **multicollinearity**: with `price_lag_2d`, `price_lag_3d`,
# `price_lag_7d` and `price_lag_14d` all in the model, the individual coefficients get
# unstable even though their sum is stable.
#
# That is a real limitation of reading OLS coefficients on correlated features, and it is worth
# saying rather than glossing over. It is also the reason regularised regression exists.

# %%
lagged = [c for c in used if c.startswith("price_lag")]
print("correlation among the price lags:")
features[lagged].corr().round(3)

# %% [markdown]
# ## 4. Residual diagnostics
#
# The diagnostics say what the model missed.

# %%
diagnostics = sarimax.residual_diagnostics(model)
pd.Series(diagnostics).to_frame("value")

# %% [markdown]
# **Ljung-Box** tests whether the residuals are autocorrelated. On half-hourly power data the
# answer is essentially always yes, and the p-value will be indistinguishable from zero.
#
# That is not a failure to hide — it is a statement that a linear model on these features
# cannot capture everything, which is exactly the gap gradient boosting fills. Reporting it is
# more honest than not testing.
#
# **Durbin-Watson** near 2 would mean no first-order autocorrelation; well below 2 means
# positive autocorrelation, i.e. the model systematically under- and over-shoots in runs.

# %%
residuals = pd.Series(model.resid)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
axes[0].hist(residuals, bins=120, color="#4C72B0")
axes[0].set(title="Residual distribution", xlabel="GBP/MWh")

sm.qqplot(residuals.sample(5000, random_state=0), line="s", ax=axes[1])
axes[1].set(title="Q-Q plot — the tails are the story")

lags = range(1, 97)
acf = [residuals.autocorr(lag) for lag in lags]
axes[2].bar(list(lags), acf, color="#4C72B0")
axes[2].axhline(0, color="black", lw=0.8)
axes[2].set(xlabel="lag (periods)", title="Residual autocorrelation")
plt.tight_layout()
plt.show()

# %% [markdown]
# The Q-Q plot shows heavy tails at both ends — the model is badly wrong during price spikes,
# which is where the money is. The residual ACF shows a clear 48-period ripple: **the daily
# cycle is not fully captured** even with three harmonics of Fourier terms.
#
# Both are actionable. The second in particular suggests more harmonics or period-specific
# effects, which is precisely the kind of interaction a tree finds without being told.

# %% [markdown]
# ## 5. Scoring it against the baselines

# %%
predictions = sarimax.predict_ols(model, test, used)
baseline = naive.predict(test, naive.HEADLINE_BASELINE)

rows = [
    {"model": "OLS + Fourier", **metrics.summary(test["price"], predictions)},
    {"model": f"naive ({naive.HEADLINE_BASELINE})", **metrics.summary(test["price"], baseline)},
]
comparison = pd.DataFrame(rows).set_index("model")
comparison["mae_improvement"] = 1 - comparison["mae"] / comparison.loc[
    f"naive ({naive.HEADLINE_BASELINE})", "mae"
]
comparison.round(3)

# %% [markdown]
# A single train/test split, not walk-forward, so this is indicative rather than the headline —
# Lesson 09 does it properly. But it establishes that a linear model on these features already
# beats the baseline comfortably. Most of the value is in the **features**, not the algorithm.
#
# That is a useful thing to know before spending a day tuning XGBoost.

# %% [markdown]
# ## 6. Why not SARIMAX at half-hourly frequency?
#
# The textbook answer for a seasonal series is SARIMA with a seasonal period of 48. There are
# two reasons this project does not do that, and stating them is better than silently omitting
# the harder model.
#
# **Computational.** A seasonal state-space model with period 48 on 100k observations does not
# finish in usable time — the state vector alone is enormous.
#
# **Conceptual, and more important.** SARIMA's seasonal component models *stochastic*
# seasonality — a cycle whose shape drifts randomly. GB's intraday cycle is not stochastic. It
# is driven by the clock and by human routine, and deterministic Fourier terms represent that
# better *and* cheaper.
#
# So SARIMAX is fitted where it genuinely earns its place: the **daily mean** series, where the
# weekly cycle is a real candidate for stochastic seasonality.

# %%
panel = load_panel()
sarimax_model = sarimax.fit_sarimax_daily(panel)
print(sarimax_model.summary().tables[1])

# %% [markdown]
# ## Exercises
#
# 1. Add three more Fourier harmonics for the daily cycle and refit. Does the 48-period ripple
#    in the residual ACF shrink?
# 2. Fit the OLS separately for each settlement period (48 tiny regressions). Compare total MAE
#    against the pooled model. What have you gained, and what have you lost?
# 3. Replace OLS with `sm.RLM` (robust regression, Huber loss). How do the coefficients move,
#    and which economic checks in section 3 now agree?
#
# ## Commit checkpoint
#
# ```bash
# git add -A && git commit -m "Lesson 07: OLS benchmark, economic sign checks, residual diagnostics"
# ```
#
# ---
#
# **Next:** `08-features.ipynb` — the information set, made concrete.
