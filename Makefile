.PHONY: install test lint serve clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check yara tests

serve:
	uvicorn yara.api:app --reload

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
