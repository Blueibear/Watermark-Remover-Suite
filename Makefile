.PHONY: setup dev install download-models verify-sample bench test test-cov test-cov-html format lint clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .[develop]

install:
	$(PIP) install -e .

format:
	$(VENV)/bin/black .
	$(VENV)/bin/ruff check --fix .

lint:
	$(VENV)/bin/ruff check .

test:
	$(VENV)/bin/pytest -v

test-cov:
	$(VENV)/bin/pytest --cov --cov-report=term-missing --cov-report=xml

test-cov-html:
	$(VENV)/bin/pytest --cov --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

download-models:
	$(PY) -m watermark_remover.models.download_models --all

verify-sample:
	wmr image samples/images/demo.jpg --out out/demo_lama.jpg --method lama --mask auto || true
	wmr video samples/videos/demo.mp4 --out out/demo_lama.mp4 --method lama --window 48 --overlap 12 --temporal-guidance flow,K=8 --seam-blend flow_fade || true

bench:
	wmr bench samples/videos/demo.mp4 --report out/bench.json || true

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache build dist *.egg-info out/** htmlcov .coverage coverage.xml
