.PHONY: help run clean venv install test test-cov lint typecheck check pre-commit

SERVICE_NAME  ?= "timeseries-zarr-py"
VENV_DIR      ?= venv
PYTHON        ?= python3

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make venv       - create virtual environment and install all dependencies"
	@echo "make install    - install dependencies into existing venv"
	@echo "make pre-commit - install pre-commit hooks"
	@echo "make test       - run tests"
	@echo "make test-cov   - run tests with coverage report"
	@echo "make lint       - run linter + formatter with auto-fix"
	@echo "make typecheck  - run mypy --strict"
	@echo "make check      - run lint check, typecheck, and tests"
	@echo "make run        - run the writer locally via docker-compose"
	@echo "make clean      - remove all files from locally mounted input / output directories"

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r timeseries_zarr/requirements.txt
	$(VENV_DIR)/bin/pip install -r requirements-test.txt
	@echo ""
	@echo "Virtual environment created. Activate with:"
	@echo "  source $(VENV_DIR)/bin/activate"

install:
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r timeseries_zarr/requirements.txt
	$(VENV_DIR)/bin/pip install -r requirements-test.txt

test:
	$(VENV_DIR)/bin/python -m pytest tests/ -v

test-cov:
	$(VENV_DIR)/bin/python -m pytest tests/ -v --cov=timeseries_zarr --cov-report=term-missing

lint:
	$(VENV_DIR)/bin/ruff check --fix timeseries_zarr/ tests/
	$(VENV_DIR)/bin/ruff format timeseries_zarr/ tests/

typecheck:
	$(VENV_DIR)/bin/mypy timeseries_zarr/

check:
	$(VENV_DIR)/bin/ruff check timeseries_zarr/ tests/
	$(VENV_DIR)/bin/ruff format --check timeseries_zarr/ tests/
	$(VENV_DIR)/bin/mypy timeseries_zarr/
	$(VENV_DIR)/bin/python -m pytest tests/

pre-commit:
	$(VENV_DIR)/bin/pre-commit install

run:
	docker-compose -f docker-compose.yml down --remove-orphans
	docker-compose -f docker-compose.yml build
	docker-compose -f docker-compose.yml up --exit-code-from writer

clean:
	rm -f data/input/*
	rm -f data/output/*
