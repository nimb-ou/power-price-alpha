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
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ppa.config import DATA_PROCESSED, REPORTS, STUDY
from ppa.eval import metrics
from ppa.features.build import build as build_features
from ppa.features.build import feature_columns
from ppa.models import naive, xgb

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
) -> pd.DataFrame:
    """Produce out-of-sample predictions for every model over every fold.

    `drop_weather` removes every `*_fcst` column. Weather features are built
    from the Open-Meteo *archive* (outturn), not the forecast that would have
    been available at gate closure, so the main run is optimistic by an unknown
    amount. Running with and without weather brackets that gap: the truth for a
    real deployment lies between the two, closer to the with-weather end for
    temperature (day-ahead temperature forecasts are good) and closer to the
    without-weather end for wind.
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


def score(predictions: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers, including the improvement the resume claims."""
    truth = predictions["price"].to_numpy()

    models = {
        column.removeprefix("pred_"): predictions[column].to_numpy()
        for column in predictions.columns
        if column.startswith("pred_")
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
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-train-days", type=int, default=STUDY.initial_train_days)
    parser.add_argument("--refit-days", type=int, default=STUDY.refit_days)
    parser.add_argument("--no-weather", action="store_true",
                        help="ablation: drop archive-weather features")
    args = parser.parse_args()

    predictions = run(
        initial_train_days=args.initial_train_days,
        refit_days=args.refit_days,
        drop_weather=args.no_weather,
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
    print("=" * 74)


if __name__ == "__main__":
    main()
