PY := .venv/bin/python

.DEFAULT_GOAL := help

# Default study window. Spans the calm pre-2021 market, the 2021-22 gas crisis,
# and the post-crisis normalisation — which is what makes regime analysis possible.
START ?= 2019-01-01
END   ?= 2025-06-30

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv and install dependencies
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

ingest: ## Download prices, demand and weather into data/raw (cached; safe to re-run)
	$(PY) -m ppa.ingest.elexon    --start $(START) --end $(END)
	$(PY) -m ppa.ingest.neso      --start $(START) --end $(END)
	$(PY) -m ppa.ingest.openmeteo --start $(START) --end $(END)

panel: ingest ## Assemble the half-hourly panel with quality checks
	$(PY) -m ppa.data.assemble --start $(START) --end $(END)

features: panel ## Build the day-ahead feature matrix
	$(PY) -m ppa.features.build

forecast: features ## Walk-forward backtest: naive, SARIMAX, XGBoost
	$(PY) -m ppa.models.walkforward

backtest: forecast ## Turn forecasts into positions and run the trading backtest
	$(PY) -m ppa.strategy.backtest

report: backtest ## Write reports/metrics.json and reports/report.html
	$(PY) -m ppa.report.build_report

test: ## Run the test suite (network tests excluded)
	$(PY) -m pytest -m "not network"

lint: ## ruff + mypy
	.venv/bin/ruff check src tests
	.venv/bin/mypy

all: report test ## Full reproducible pipeline
	@echo "--- reports/metrics.json ---"
	@cat reports/metrics.json

clean: ## Remove derived data and reports (keeps the raw API cache and the venv)
	rm -rf data/processed/* artifacts/* reports/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-cache: ## Also drop the raw API cache — next ingest refetches everything
	rm -rf data/raw/*

.PHONY: help venv ingest panel features forecast backtest report test lint all clean clean-cache
