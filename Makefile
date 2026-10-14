.PHONY: install test lint format

install:
	python -m pip install -U pip
	python -m pip install -e .

test:
	pytest -q

lint:
	ruff check src tests

format:
	black src tests
