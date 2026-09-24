.PHONY: install test cover lint ci serve docker-build docker-up docker-ingest docker-down clean

install:
	pip install -e ".[dev]"

test:
	pytest

# Same invocation the CI test job runs, so a red pipeline is reproducible
# locally without reading the workflow file.
cover:
	pytest --cov=yara --cov-report=term-missing --cov-report=xml

lint:
	ruff check yara tests

ci: lint cover

serve:
	uvicorn yara.api:app --reload

docker-build:
	docker build -t yara:local .

docker-up:
	docker compose up --build -d

# One-shot CLI container writing into the shared volume. The API caches the
# index at startup, so restart it to pick up what this just ingested.
docker-ingest:
	docker compose run --rm ingest
	docker compose up -d --force-recreate api

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
