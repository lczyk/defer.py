.SUFFIXES:

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} /^[a-zA-Z_.\/-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: lint
lint:  ## Lint and format-check with ruff
	uvx ruff check defer.py test_defer.py
	uvx ruff format --check defer.py test_defer.py

.PHONY: test
test:  ## Run the tests
	uvx pytest -q

.PHONY: verify
verify: lint test  ## Run all checks

.PHONY: example
example:  ## Run the examples
	uv run --no-project python example.py
