# Overridable so CI can use the interpreter it installed into, rather than a
# virtualenv that only exists on a developer machine.
PY ?= .venv/bin/python
RUFF ?= $(PY) -m ruff
MYPY ?= $(PY) -m mypy
JUPYTER ?= $(PY) -m jupyter

.DEFAULT_GOAL := help

# Default study window. Spans the calm pre-2021 market, the 2021-22 gas crisis,
# and the post-crisis normalisation — which is what makes regime analysis possible.
START ?= 2019-01-01
END   ?= 2025-06-30

# CI overrides these to run a genuine but scaled-down walk-forward on the
# committed fixture window.
TRAIN_DAYS ?= 730
REFIT_DAYS ?= 30

PROC := data/processed
PANEL      := $(PROC)/panel.parquet
FEATURES   := $(PROC)/features.parquet
FORECAST   := $(PROC)/walkforward_predictions.parquet
ABLATION   := $(PROC)/walkforward_predictions_no_weather.parquet
BACKTEST   := $(PROC)/backtest.parquet
REPORT     := reports/metrics.json

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv and install dependencies
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

fixtures: ## Copy the committed test fixture into data/raw (offline; used by CI)
	mkdir -p data/raw
	cp -R tests/fixtures/raw/. data/raw/

ingest: ## Download prices, demand and weather into data/raw (cached; safe to re-run)
	$(PY) -m ppa.ingest.elexon    --start $(START) --end $(END)
	$(PY) -m ppa.ingest.neso      --start $(START) --end $(END)
	$(PY) -m ppa.ingest.openmeteo --start $(START) --end $(END)

# --- the pipeline -----------------------------------------------------------
#
# Each stage is a real file target depending on the previous stage's output, so
# make skips work that is already done. This matters here more than in most
# projects: the walk-forward takes ~10 minutes, and with phony targets a plain
# `make report` re-ran it three times — once for `backtest`, once for
# `forecast-ablation`, and once more for `report` — because a phony prerequisite
# is always considered out of date.
#
# `panel` deliberately does NOT depend on `ingest`. Data reaches data/raw from
# either `make ingest` (network) or `make fixtures` (committed sample), and the
# loaders raise a clear "run make ingest first" if neither has happened.

$(PANEL):
	$(PY) -m ppa.data.assemble --start $(START) --end $(END)

$(FEATURES): $(PANEL)
	$(PY) -m ppa.features.build

$(FORECAST): $(FEATURES)
	$(PY) -m ppa.models.walkforward --initial-train-days $(TRAIN_DAYS) --refit-days $(REFIT_DAYS)

# --no-quantiles: the ablation exists to bracket the *point* forecast against
# the optimism of using outturn weather. Fitting the interval model here too
# would double its runtime to bracket a number nothing reads.
$(ABLATION): $(FEATURES)
	$(PY) -m ppa.models.walkforward --no-weather --no-quantiles --initial-train-days $(TRAIN_DAYS) --refit-days $(REFIT_DAYS)

$(BACKTEST): $(FORECAST)
	$(PY) -m ppa.strategy.backtest

$(REPORT): $(BACKTEST) $(ABLATION)
	$(PY) -m ppa.report.build_report

panel: $(PANEL)                       ## Assemble the half-hourly panel with quality checks
features: $(FEATURES)                 ## Build the day-ahead feature matrix
forecast: $(FORECAST)                 ## Walk-forward backtest: naive, SARIMAX, XGBoost
forecast-ablation: $(ABLATION)        ## Re-run the walk-forward with weather features removed
backtest: $(BACKTEST)                 ## Turn forecasts into a battery schedule and backtest it
report: $(REPORT)                     ## Write reports/metrics.json and reports/report.html

rebuild: clean report                 ## Force the whole pipeline to re-run

# --- development ------------------------------------------------------------

notebooks: ## Regenerate .ipynb lessons from notebooks/_src/*.py (no outputs)
	$(PY) tools/py2nb.py

notebooks-check: notebooks ## Execute every lesson end to end; fails on any cell error
	MPLBACKEND=Agg $(JUPYTER) nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=1800 --output-dir=/tmp/ppa-nb notebooks/*.ipynb

test: ## Run the test suite (network tests excluded)
	$(PY) -m pytest -m "not network"

lint: ## ruff + mypy
	$(RUFF) check src tests tools
	$(MYPY)

all: ingest report test ## Full reproducible pipeline (fetches data)
	@echo "--- reports/metrics.json ---"
	@cat reports/metrics.json

clean: ## Remove derived data and reports (keeps the raw API cache and the venv)
	rm -rf data/processed/* artifacts/* reports/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-cache: ## Also drop the raw API cache — next ingest refetches everything
	rm -rf data/raw/*

.PHONY: help venv fixtures ingest panel features forecast forecast-ablation \
        backtest report rebuild notebooks notebooks-check test lint all clean clean-cache
