# Developer entry points. `make help` lists them.
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PY_PATHS := src tests scripts .claude

.DEFAULT_GOAL := help
.PHONY: help install install-all check lint type test fmt doctor clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv, install the package with dev tools, enable hooks
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev]"
	@if git rev-parse --git-dir >/dev/null 2>&1; then \
		git config core.hooksPath scripts/hooks; \
		echo "Git hooks enabled from scripts/hooks"; \
	fi

install-all: install ## Also install the Claude, MCP and API extras
	$(BIN)/pip install -e ".[dev,claude,mcp,api]"

check: lint type test ## Everything CI runs

lint: ## Ruff lint and format check
	$(BIN)/ruff check $(PY_PATHS)
	$(BIN)/ruff format --check $(PY_PATHS)

type: ## Strict mypy over the package
	$(BIN)/mypy

test: ## Unit tests (offline, deterministic)
	$(BIN)/pytest -q

fmt: ## Format and auto-fix
	$(BIN)/ruff format $(PY_PATHS)
	$(BIN)/ruff check --fix $(PY_PATHS)

doctor: ## Check the local toolchain
	$(BIN)/aie doctor

clean: ## Remove caches and build outputs
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis build *.egg-info

# The book's sources are not part of the public repository. When they are
# present, they add their own targets and lint paths.
-include book/book.mk
