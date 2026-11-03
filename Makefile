.PHONY: check lint test test-fast

check: lint test-fast

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

test-fast:
	uv run pytest -q -m "not slow"

test:
	uv run pytest -q
