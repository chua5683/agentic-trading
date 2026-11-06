PY := .venv/bin/python
CONFIG ?= configs/default.yaml
SYMBOL ?= BTC-USD
START ?= 2024-01-01
END ?= 2025-11-01

.PHONY: help setup test doctor backtest evaluate analyze offline clean

help:
	@echo "make setup      - create .venv and install dependencies"
	@echo "make test       - run the full test suite"
	@echo "make doctor     - check environment, data vendors, LLM + broker status"
	@echo "make backtest   - backtest $(SYMBOL) $(START)->$(END), write reports/"
	@echo "make evaluate   - multi-regime table vs buy & hold (reports/regimes.md)"
	@echo "make analyze    - one auditable decision for the latest bar"
	@echo "make offline    - offline backtest on synthetic data (no network)"
	@echo "make clean      - remove caches and reports"

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev]"

test:
	$(PY) -m pytest -q

doctor:
	$(PY) -m agentic_trading doctor --config $(CONFIG)

backtest:
	$(PY) -m agentic_trading backtest --config $(CONFIG) --symbol $(SYMBOL) \
		--start $(START) --end $(END) --out reports

evaluate:
	$(PY) -m agentic_trading evaluate --config $(CONFIG) \
		--symbols $(SYMBOL) ETH-USD --out reports/regimes.md

analyze:
	$(PY) -m agentic_trading analyze --config $(CONFIG) --symbol $(SYMBOL) --verbose

offline:
	$(PY) -m agentic_trading backtest --config $(CONFIG) --symbol $(SYMBOL) --offline --out reports

clean:
	rm -rf reports data_cache .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
