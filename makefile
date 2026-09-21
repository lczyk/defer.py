.SUFFIXES:

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} /^[a-zA-Z_.\/-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: lint
lint:  ## Lint and format-check with ruff
	uvx ruff check .
	uvx ruff format --check .

.PHONY: test
test:  ## Run the tests
	uvx pytest -q

.PHONY: verify
verify: lint test  ## Run all checks

.PHONY: example
example:  ## Run the examples
	uv run --no-project python example.py

.PHONY: bench
bench:  ## Run the benchmarks
	uv run --no-project python bench_defer.py
