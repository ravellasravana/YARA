.PHONY: install test lint clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check yara tests

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
