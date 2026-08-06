"""Walk-forward evaluation.

The only defensible way to evaluate a forecaster on time series. A random
train/test split would let the model learn from July to predict June, which on a
market with strong regime persistence produces a spectacular and meaningless
score.

The scheme:

    |<---- initial train (2y) ---->|  predict 30d  |
    |<---- initial train + 30d ------->|  predict 30d  |
    |<---- initial train + 60d ----------->|  predict 30d  |
                                                          ...

An **expanding** window, refitted every 30 days, always predicting forward. Two
choices worth defending:

- *Expanding, not rolling.* GB power has structural breaks (the 2021-22 gas
  crisis) but also stable seasonal structure. Discarding 2019 loses the only
  examples of a calm market the model will ever see — which is exactly what it
  needs when the market normalises again in 2024.
- *Refit every 30 days, not every day.* Daily refitting is ~2,400 fits for a
  difference well inside the noise. Thirty days is the honest compromise and it
  is stated rather than tuned.

Every fold's predictions are out-of-sample by construction. Combined with the
information-set discipline in `features/build.py`, that is what makes the
headline number mean something.

    python -m ppa.models.walkforward
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ppa.config import DATA_PROCESSED, REPORTS, STUDY
from ppa.eval import metrics
from ppa.eval.regimes import label_regime
from ppa.features.build import build as build_features
from ppa.features.build import feature_columns
from ppa.models import conformal, naive, xgb

log = logging.getLogger(__name__)

PREDICTIONS_NAME = "walkforward_predictions.parquet"


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int


def make_folds(
    dates: pd.Series,
    initial_train_days: int = STUDY.initial_train_days,
    refit_days: int = STUDY.refit_days,
) -> list[Fold]:
    """Expanding-window folds over the sorted unique dates."""
    unique = pd.to_datetime(pd.Series(dates.unique())).sort_values().reset_index(drop=True)

    folds: list[Fold] = []
    cursor = initial_train_days
    index = 0
    while cursor < len(unique):
        test_end_position = min(cursor + refit_days, len(unique)) - 1
        folds.append(
            Fold(
                index=index,
                train_end=unique.iloc[cursor - 1],
                test_start=unique.iloc[cursor],
                test_end=unique.iloc[test_end_position],
                n_train=cursor,
                n_test=test_end_position - cursor + 1,
            )
        )
        cursor += refit_days
        index += 1
    return folds


def run(
    features: pd.DataFrame | None = None,
    initial_train_days: int = STUDY.initial_train_days,
    refit_days: int = STUDY.refit_days,
    drop_weather: bool = False,
    with_quantiles: bool = True,
) -> pd.DataFrame:
    """Produce out-of-sample predictions for every model over every fold.

    `drop_weather` removes every `*_fcst` column. Weather features are built
    from the Open-Meteo *archive* (outturn), not the forecast that would have
    been available at gate closure, so the main run is optimistic by an unknown
    amount. Running with and without weather brackets that gap: the truth for a
    real deployment lies between the two, closer to the with-weather end for
    temperature (day-ahead temperature forecasts are good) and closer to the
    without-weather end for wind.

    `with_quantiles` fits a second, multi-quantile model per fold for the P10 /
    P50 / P90 interval. It roughly doubles the run, which is why the weather
    ablation turns it off: the ablation exists to bracket the point forecast,
    and the interval would be bracketed with it for no extra information.
    """
    features = features if features is not None else build_features()
    features = features.sort_values("start_time").reset_index(drop=True)
    features["date"] = pd.to_datetime(features["settlement_date"])

    columns = feature_columns(features)
    if drop_weather:
        columns = [c for c in columns if not c.endswith("_fcst")]
        log.info("weather ablation: dropped %d forecast columns",
                 len(feature_columns(features)) - len(columns))
    folds = make_folds(features["date"], initial_train_days, refit_days)
    log.info("%d folds, %d features", len(folds), len(columns))

    outputs: list[pd.DataFrame] = []
    for fold in folds:
        train = features[features["date"] <= fold.train_end]
        test = features[
            (features["date"] >= fold.test_start) & (features["date"] <= fold.test_end)
        ]
        if test.empty:
            continue

        predictions, model = xgb.fit_predict(train, test, columns)

        block = test[
            ["settlement_date", "settlement_period", "start_time", "price"]
        ].copy()
        block["fold"] = fold.index
        block["pred_xgb"] = predictions
        for name in naive.BASELINES:
            block[f"pred_{name}"] = naive.predict(test, name)

        if with_quantiles:
            # One quantile fit per fold, two intervals out of it. The model is
            # fitted on the window minus a trailing calibration slice, so the
            # raw and conformalised bounds come from the *same* estimator and
            # the comparison isolates the conformal step rather than confounding
            # it with a difference in training data.
            fit_part, calibration = conformal.split_calibration(train)
            drawn = xgb.fit_predict_quantiles(fit_part, test, columns)
            for position, level in enumerate(xgb.QUANTILES):
                block[f"pred_xgb_p{int(level * 100)}"] = drawn[:, position]

            if len(calibration):
                calibrated = xgb.fit_predict_quantiles(fit_part, calibration, columns)
                adjuster = conformal.fit(
                    calibration["price"].to_numpy(),
                    calibrated[:, 0],
                    calibrated[:, -1],
                    alpha=1 - (xgb.QUANTILES[-1] - xgb.QUANTILES[0]),
                    periods=calibration["settlement_period"].to_numpy(),
                )
                low, high = adjuster.apply(
                    drawn[:, 0], drawn[:, -1], test["settlement_period"].to_numpy()
                )
                block["pred_xgb_p10_conformal"] = low
                block["pred_xgb_p90_conformal"] = high
                block["conformal_offset"] = adjuster.global_offset

        outputs.append(block)
        if fold.index % 10 == 0:
            log.info(
                "fold %d/%d  train<=%s  test %s..%s  n_train=%d",
                fold.index,
                len(folds),
                fold.train_end.date(),
                fold.test_start.date(),
                fold.test_end.date(),
                len(train),
            )

    combined = pd.concat(outputs, ignore_index=True)
    log.info("%d out-of-sample predictions over %d folds", len(combined), len(outputs))
    return combined


def _quantile_columns(predictions: pd.DataFrame) -> dict[float, str]:
    """Map quantile level -> column name, for whichever levels are present."""
    return {
        level: f"pred_xgb_p{int(level * 100)}"
        for level in xgb.QUANTILES
        if f"pred_xgb_p{int(level * 100)}" in predictions.columns
    }


def _non_point_columns(predictions: pd.DataFrame) -> set[str]:
    """Every `pred_*` column that is not a point forecast.

    Matched by shape rather than by an explicit list. The first version listed
    the three raw quantile columns, which meant the conformalised bounds — added
    later, and also named `pred_*` — sailed into the baseline comparison table
    and were scored with MAE. A P90 evaluated as a point forecast looks like a
    terrible model (-35% against the naive) rather than a correctly fitted upper
    bound, which is exactly the confusion the exclusion exists to prevent.
    """
    return {
        column
        for column in predictions.columns
        if re.match(r"^pred_xgb_p\d+(_conformal)?$", column)
    }


def score_intervals(predictions: pd.DataFrame) -> dict[str, Any] | None:
    """Pinball loss per quantile, plus coverage and width for the P10-P90 band.

    Returns None when the run was made without quantiles, rather than a block of
    NaNs — a missing measurement and a measurement that came out empty are
    different things and should not serialise identically.
    """
    columns = _quantile_columns(predictions)
    if len(columns) < 2:
        return None

    truth = predictions["price"].to_numpy()
    levels = sorted(columns)
    low, high = levels[0], levels[-1]

    # The conformalised band, when the run produced one. Scored with the same
    # function as the raw band so the two rows are directly comparable — a
    # separate code path here is how a "before and after" ends up measuring two
    # different things.
    conformal_block: dict[str, Any] | None = None
    if {"pred_xgb_p10_conformal", "pred_xgb_p90_conformal"} <= set(predictions.columns):
        usable = predictions.dropna(subset=["pred_xgb_p10_conformal"])
        conformal_block = {
            **metrics.interval_score(
                usable["price"].to_numpy(),
                usable["pred_xgb_p10_conformal"],
                usable["pred_xgb_p90_conformal"],
                alpha=1 - (high - low),
            ),
            "mean_offset": float(usable["conformal_offset"].mean()),
            "calibration_days": conformal.CALIBRATION_DAYS,
        }

    return {
        "quantiles": levels,
        "pinball": {
            f"p{int(level * 100)}": metrics.pinball(truth, predictions[column], level)
            for level, column in sorted(columns.items())
        },
        # Mean pinball across the fitted quantiles: a coarse CRPS, and the one
        # number that summarises the whole predictive distribution.
        "mean_pinball": float(
            np.mean([
                metrics.pinball(truth, predictions[column], level)
                for level, column in columns.items()
            ])
        ),
        "interval": metrics.interval_score(
            truth, predictions[columns[low]], predictions[columns[high]], alpha=1 - (high - low)
        ),
        "interval_conformal": conformal_block,
        # The P50 is fitted under pinball loss and the headline forecast under
        # absolute error. They are different estimators of a similar quantity;
        # reporting both stops either being quietly swapped for the other.
        "median_mae": metrics.mae(truth, predictions[columns[0.5]])
        if 0.5 in columns
        else None,
    }


def score_by_regime(predictions: pd.DataFrame) -> dict[str, Any]:
    """MAE and improvement per market regime.

    The strategy layer has reported by regime since it was written; the forecast
    layer did not, which left the obvious question — is the improvement just the
    gas crisis? — answerable only by rerunning things by hand.
    """
    labelled = predictions.copy()
    labelled["regime"] = label_regime(labelled["start_time"])

    out: dict[str, Any] = {}
    for name, group in labelled.groupby("regime", sort=False):
        truth = group["price"].to_numpy()
        baseline_mae = metrics.mae(truth, group[f"pred_{naive.HEADLINE_BASELINE}"])
        model_mae = metrics.mae(truth, group["pred_xgb"])
        out[str(name)] = {
            "n": int(len(group)),
            "mean_price": float(np.mean(truth)),
            "price_volatility": float(np.std(truth)),
            "mae_xgb": model_mae,
            "mae_naive": baseline_mae,
            "mae_improvement_vs_naive": metrics.improvement(baseline_mae, model_mae),
        }
    return out


def score(predictions: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers, including the improvement the resume claims."""
    truth = predictions["price"].to_numpy()

    excluded = _non_point_columns(predictions)
    models = {
        column.removeprefix("pred_"): predictions[column].to_numpy()
        for column in predictions.columns
        if column.startswith("pred_") and column not in excluded
    }

    baseline_name = naive.HEADLINE_BASELINE
    baseline = models[baseline_name]
    baseline_mae = metrics.mae(truth, baseline)

    per_model: dict[str, Any] = {}
    for name, prediction in models.items():
        block: dict[str, Any] = dict(metrics.summary(truth, prediction))
        block["mae_improvement_vs_naive"] = metrics.improvement(baseline_mae, block["mae"])
        lo, hi = metrics.bootstrap_mae_ci(truth, prediction)
        block["mae_ci95"] = [lo, hi]
        per_model[name] = block

    test = metrics.diebold_mariano(truth, models["xgb"], baseline)

    return {
        "headline_baseline": baseline_name,
        "n_predictions": int(len(predictions)),
        "period": {
            "start": str(predictions["start_time"].min()),
            "end": str(predictions["start_time"].max()),
        },
        "models": per_model,
        "mae_improvement_vs_naive": per_model["xgb"]["mae_improvement_vs_naive"],
        "diebold_mariano_xgb_vs_naive": test,
        "by_regime": score_by_regime(predictions),
        "intervals": score_intervals(predictions),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-train-days", type=int, default=STUDY.initial_train_days)
    parser.add_argument("--refit-days", type=int, default=STUDY.refit_days)
    parser.add_argument("--no-weather", action="store_true",
                        help="ablation: drop archive-weather features")
    parser.add_argument("--no-quantiles", action="store_true",
                        help="skip the P10/P50/P90 model (roughly halves the run)")
    args = parser.parse_args()

    predictions = run(
        initial_train_days=args.initial_train_days,
        refit_days=args.refit_days,
        drop_weather=args.no_weather,
        with_quantiles=not args.no_quantiles,
    )
    suffix = "_no_weather" if args.no_weather else ""
    predictions.to_parquet(DATA_PROCESSED / f"walkforward_predictions{suffix}.parquet", index=False)

    report = score(predictions)
    report["weather_features_used"] = not args.no_weather
    (REPORTS / f"forecast_metrics{suffix}.json").write_text(json.dumps(report, indent=2))

    print()
    print("=" * 74)
    print(f"  out-of-sample: {report['n_predictions']:,} half-hours "
          f"({report['period']['start'][:10]} .. {report['period']['end'][:10]})")
    print(f"  headline baseline: {report['headline_baseline']}")
    print()
    print(f"  {'model':<28}{'MAE':>9}{'RMSE':>9}{'vs naive':>11}")
    for name, block in report["models"].items():
        improvement = block["mae_improvement_vs_naive"]
        marker = "  <-- baseline" if name == report["headline_baseline"] else ""
        print(f"  {name:<28}{block['mae']:>9.3f}{block['rmse']:>9.3f}{improvement:>10.1%}{marker}")
    print()
    dm = report["diebold_mariano_xgb_vs_naive"]
    print(f"  Diebold-Mariano (xgb vs naive): stat {dm['statistic']:.2f}, p = {dm['p_value']:.2e}")

    print()
    print(f"  {'regime':<24}{'n':>8}{'mean px':>10}{'MAE xgb':>10}{'MAE naive':>11}{'vs naive':>10}")
    for name, block in report["by_regime"].items():
        print(f"  {name:<24}{block['n']:>8,}{block['mean_price']:>10.1f}"
              f"{block['mae_xgb']:>10.2f}{block['mae_naive']:>11.2f}"
              f"{block['mae_improvement_vs_naive']:>10.1%}")

    if intervals := report.get("intervals"):
        band = intervals["interval"]
        print()
        print(f"  {'P10-P90 interval':<22}{'coverage':>10}{'width':>10}{'Winkler':>11}")
        print(f"  {'raw quantile fit':<22}{band['coverage']:>10.1%}"
              f"{band['mean_width']:>10.1f}{band['winkler']:>11.1f}")
        if conf := intervals.get("interval_conformal"):
            print(f"  {'conformalised':<22}{conf['coverage']:>10.1%}"
                  f"{conf['mean_width']:>10.1f}{conf['winkler']:>11.1f}")
        print(f"  nominal {band['nominal_coverage']:.0%};  mean pinball "
              f"{intervals['mean_pinball']:.3f};  {band['crossings']} quantile crossings")
    print("=" * 74)


if __name__ == "__main__":
    main()
