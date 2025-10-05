.PHONY: help dev test lint fmt lab clean precommit

help:
	@echo "make dev       - install dev deps and package (editable)"
	@echo "make test      - run unit tests"
	@echo "make lint      - ruff + black (check only)"
	@echo "make fmt       - auto-format (ruff fix + black + isort)"
	@echo "make lab       - launch JupyterLab"
	@echo "make clean     - remove caches and build artifacts"
	@echo "make precommit - install pre-commit hooks and run on all files"

dev:
	python -m pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	black --check .
	isort --check-only .

fmt:
	ruff check . --fix
	black .
	isort .

lab:
	jupyter lab

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info

precommit:
	python -m pip install --upgrade pre-commit
	pre-commit install
	pre-commit run --all-files
