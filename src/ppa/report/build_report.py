"""Assemble reports/metrics.json and reports/report.html.

`metrics.json` is the machine-readable record that `RESUME_CLAIMS.md` cites and
CI publishes. `report.html` is the human-readable version with charts, and it is
the thing to open in an interview.

    python -m ppa.report.build_report
"""

from __future__ import annotations

import json
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import base64  # noqa: E402
import io  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ppa.config import DATA_PROCESSED, REPORTS, STUDY  # noqa: E402
from ppa.eval import metrics  # noqa: E402
from ppa.models import naive  # noqa: E402

log = logging.getLogger(__name__)


def _embed(fig: plt.Figure) -> str:
    """PNG as a data URI, so report.html is a single self-contained file."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _chart_equity(backtest: pd.DataFrame) -> str:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    times = pd.to_datetime(backtest["start_time"])

    ax1.plot(times, backtest["cum_pnl"], lw=1.4, label="cumulative P&L (net of frictions)")
    ax1.axvspan(
        pd.Timestamp(STUDY.volatile_start, tz="UTC"),
        pd.Timestamp(STUDY.volatile_end, tz="UTC"),
        alpha=0.12,
        color="crimson",
        label="gas crisis",
    )
    ax1.set(ylabel="cumulative P&L (GBP)", title="1 MW / 2 MWh battery — cumulative profit")
    ax1.legend()

    cumulative = backtest["cum_pnl"]
    drawdown = cumulative - cumulative.cummax()
    ax2.fill_between(times, drawdown, 0, color="crimson", alpha=0.5)
    ax2.set(ylabel="drawdown", xlabel="")
    return _embed(fig)


def _chart_error_by_period(predictions: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for column, label in [
        ("pred_xgb", "XGBoost"),
        (f"pred_{naive.HEADLINE_BASELINE}", "seasonal naive"),
    ]:
        table = metrics.by_period_of_day(predictions, "price", column)
        ax.plot(table["settlement_period"], table["mae"], lw=1.8, label=label)
    ax.set(
        xlabel="settlement period (1 = 00:00 local)",
        ylabel="MAE (GBP/MWh)",
        title="Where in the day the model helps",
    )
    ax.legend()
    return _embed(fig)


def _chart_monthly_mae(predictions: pd.DataFrame) -> str:
    frame = predictions.copy()
    frame["month"] = pd.to_datetime(frame["start_time"]).dt.to_period("M").astype(str)

    rows = []
    for month, group in frame.groupby("month"):
        rows.append(
            {
                "month": month,
                "xgb": metrics.mae(group["price"], group["pred_xgb"]),
                "naive": metrics.mae(
                    group["price"], group[f"pred_{naive.HEADLINE_BASELINE}"]
                ),
            }
        )
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(table["month"], table["naive"], lw=1.5, label="seasonal naive")
    ax.plot(table["month"], table["xgb"], lw=1.8, label="XGBoost")
    ax.set(ylabel="MAE (GBP/MWh)", title="Monthly out-of-sample MAE")
    ax.set_xticks(table["month"][::6])
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    return _embed(fig)


def build() -> dict[str, Any]:
    forecast = json.loads((REPORTS / "forecast_metrics.json").read_text())
    strategy = json.loads((REPORTS / "strategy_metrics.json").read_text())

    predictions = pd.read_parquet(DATA_PROCESSED / "walkforward_predictions.parquet")
    backtest = pd.read_parquet(DATA_PROCESSED / "backtest.parquet")

    ablation_path = REPORTS / "forecast_metrics_no_weather.json"
    ablation = None
    if ablation_path.exists():
        block = json.loads(ablation_path.read_text())
        ablation = block["models"]["xgb"]
        ablation["mae_improvement_vs_naive"] = block["mae_improvement_vs_naive"]

    report: dict[str, Any] = {
        "study": {
            "start": STUDY.start,
            "end": STUDY.end,
            "initial_train_days": STUDY.initial_train_days,
            "refit_days": STUDY.refit_days,
            "regime_split": [STUDY.volatile_start, STUDY.volatile_end],
        },
        "forecast": forecast,
        "strategy": strategy,
        "weather_ablation": ablation,
    }

    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2, default=str))

    charts = {
        "equity": _chart_equity(backtest),
        "error_by_period": _chart_error_by_period(predictions),
        "monthly_mae": _chart_monthly_mae(predictions),
    }
    (REPORTS / "report.html").write_text(_render(report, charts))
    return report


def _render(report: dict[str, Any], charts: dict[str, str]) -> str:
    forecast = report["forecast"]
    strategy = report["strategy"]
    xgb = strategy["schedules"]["xgb"]

    model_rows = "".join(
        f"<tr><td>{name}</td><td>{block['mae']:.3f}</td><td>{block['rmse']:.3f}</td>"
        f"<td>{block['mae_improvement_vs_naive']:+.1%}</td>"
        f"<td>[{block['mae_ci95'][0]:.3f}, {block['mae_ci95'][1]:.3f}]</td></tr>"
        for name, block in forecast["models"].items()
    )
    schedule_rows = "".join(
        f"<tr><td>{name}</td><td>{block['total']:,.0f}</td><td>{block['mean_per_period']:.1f}</td>"
        f"<td>{block['sharpe']:.2f}</td><td>{block['hit_rate']:.1%}</td>"
        f"<td>{block['max_drawdown']:,.0f}</td></tr>"
        for name, block in strategy["schedules"].items()
    )
    regime_rows = "".join(
        f"<tr><td>{name}</td><td>{block['sharpe']:.2f}</td><td>{block['hit_rate']:.1%}</td>"
        f"<td>{block['mean_per_period']:.1f}</td><td>{block['n']:,}</td></tr>"
        for name, block in xgb["by_regime"].items()
    )

    ablation = report.get("weather_ablation")
    ablation_html = ""
    if ablation:
        ablation_html = f"""
<h3>Weather ablation</h3>
<p>Weather features are built from the Open-Meteo <i>archive</i> (outturn), not the forecast
that would have been available at gate closure, so the headline number is optimistic by an
unknown amount. Re-running with every weather feature removed brackets the gap:</p>
<table><tr><th>run</th><th>MAE</th><th>vs naive</th></tr>
<tr><td>with archive weather (headline)</td><td>{forecast['models']['xgb']['mae']:.3f}</td>
    <td>{forecast['mae_improvement_vs_naive']:+.1%}</td></tr>
<tr><td>no weather features</td><td>{ablation['mae']:.3f}</td>
    <td>{ablation['mae_improvement_vs_naive']:+.1%}</td></tr>
</table>
<p>A real deployment sits between the two — closer to the top row for temperature
(day-ahead temperature forecasts are accurate) and closer to the bottom for wind.</p>"""

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Power Price Alpha — results</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1000px;
        margin: 2rem auto; padding: 0 1rem; line-height: 1.55; color: #1a1a1a; }}
 h1 {{ border-bottom: 2px solid #333; padding-bottom: .4rem; }}
 h2 {{ margin-top: 2.2rem; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .93rem; }}
 th, td {{ border: 1px solid #ddd; padding: .45rem .6rem; text-align: right; }}
 th {{ background: #f4f4f4; text-align: left; }}
 td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; margin: 1rem 0; }}
 .note {{ background: #fff8e1; border-left: 4px solid #f0ad4e; padding: .8rem 1rem;
          margin: 1rem 0; font-size: .93rem; }}
 .headline {{ font-size: 1.4rem; font-weight: 600; color: #0a5; }}
</style></head><body>

<h1>Power Price Alpha</h1>
<p>GB half-hourly day-ahead price forecasting, and a battery-arbitrage backtest with
frictions. Study window {report['study']['start']} to {report['study']['end']};
expanding-window walk-forward, {report['study']['initial_train_days']}-day initial training,
refit every {report['study']['refit_days']} days.</p>

<h2>Forecast accuracy</h2>
<p>Out-of-sample: <b>{forecast['n_predictions']:,}</b> half-hours
({forecast['period']['start'][:10]} to {forecast['period']['end'][:10]}).
Headline baseline <code>{forecast['headline_baseline']}</code>, declared in
<code>models/naive.py</code> before any model was fitted.</p>

<p class="headline">MAE improvement vs seasonal naive:
{forecast['mae_improvement_vs_naive']:+.1%}</p>

<table><tr><th>model</th><th>MAE</th><th>RMSE</th><th>vs naive</th><th>MAE 95% CI</th></tr>
{model_rows}</table>

<p>Diebold-Mariano (XGBoost vs naive): statistic
{forecast['diebold_mariano_xgb_vs_naive']['statistic']:.2f},
p = {forecast['diebold_mariano_xgb_vs_naive']['p_value']:.2e}. Newey-West variance, so the
serial correlation in half-hourly forecast errors is accounted for.</p>

<img src="data:image/png;base64,{charts['monthly_mae']}" alt="monthly MAE">
<img src="data:image/png;base64,{charts['error_by_period']}" alt="MAE by settlement period">
{ablation_html}

<h2>Strategy: {strategy['battery']['power_mw']:.0f} MW / {strategy['battery']['energy_mwh']:.0f} MWh battery</h2>
<p>The schedule is fixed at day-ahead gate closure from the forecast alone: charge in the
cheapest forecast periods, discharge in the dearest, one cycle per day, settling against
realised prices. Round-trip efficiency {strategy['battery']['round_trip_efficiency']:.0%};
frictions {strategy['frictions']['total_per_mwh']:.2f} GBP/MWh on both legs.</p>

<table>
<tr><th>schedule</th><th>total P&amp;L (GBP)</th><th>GBP/day</th><th>Sharpe</th><th>hit rate</th><th>max DD</th></tr>
{schedule_rows}</table>

<p><b>The forecast's contribution is the gap between <code>xgb</code> and
<code>naive</code>: {strategy['uplift_vs_naive']:,.0f} GBP
({strategy['uplift_vs_naive_pct']:+.1%}).</b> The model captures
{strategy['share_of_oracle']:.1%} of what perfect foresight would earn; the naive forecast
captures {strategy['naive_share_of_oracle']:.1%}.</p>

<p>Sharpe 95% CI (block bootstrap, one-week blocks):
[{xgb['sharpe_ci95'][0]:.2f}, {xgb['sharpe_ci95'][1]:.2f}].
Traded {xgb['signal']['days_traded']:,} of {xgb['signal']['total_days']:,} days.</p>

<img src="data:image/png;base64,{charts['equity']}" alt="cumulative P&L">

<h3>By regime</h3>
<table><tr><th>regime</th><th>Sharpe</th><th>hit rate</th><th>GBP/day</th><th>days</th></tr>
{regime_rows}</table>

<div class="note">
<b>Why the Sharpe is high, and what it is not.</b> This is a <i>physical arbitrage</i>, not a
market-timing alpha. The intraday peak-to-trough price spread is almost always positive, so a
battery makes money on most days regardless of forecast quality — the naive schedule earns a
Sharpe of {strategy['schedules']['naive']['sharpe']:.1f} on its own. The forecast's real
contribution is the uplift over that baseline, not the headline Sharpe.
<br><br>
<b>Not modelled:</b> battery degradation and cycle life; state-of-charge constraints beyond
one cycle a day; balancing and imbalance exposure; market impact (we assume our bid does not
move the clearing price); grid connection and capacity charges; and the fact that a real
auction takes a bid curve rather than a point forecast.
<br><br>
<b>Weather:</b> features come from the Open-Meteo archive (outturn) rather than a genuine
day-ahead forecast — see the ablation above for the size of that advantage.
</div>

</body></html>"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = build()

    forecast = report["forecast"]
    strategy = report["strategy"]
    xgb = strategy["schedules"]["xgb"]

    print(f"wrote {REPORTS / 'metrics.json'}")
    print(f"wrote {REPORTS / 'report.html'}")
    print()
    print(f"  MAE improvement vs naive : {forecast['mae_improvement_vs_naive']:+.1%}")
    if ablation := report.get("weather_ablation"):
        print(f"    without weather        : {ablation['mae_improvement_vs_naive']:+.1%}")
    print(f"  battery P&L              : {xgb['total']:,.0f} GBP "
          f"({xgb['mean_per_period']:.1f}/day)")
    print(f"  uplift vs naive schedule : {strategy['uplift_vs_naive']:,.0f} GBP "
          f"({strategy['uplift_vs_naive_pct']:+.1%})")
    print(f"  share of oracle          : {strategy['share_of_oracle']:.1%}")
    print(f"  Sharpe (daily P&L)       : {xgb['sharpe']:.2f}  "
          f"CI {[round(v, 2) for v in xgb['sharpe_ci95']]}")
    print(f"  max drawdown             : {xgb['max_drawdown']:,.0f} GBP")


if __name__ == "__main__":
    main()
