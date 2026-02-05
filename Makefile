SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

VENV_DIR ?= .venv
VENV_BIN := $(VENV_DIR)/bin
PYTHON_BIN ?= python3
DIST_DIR ?= dist
DIST_VENV_DIR ?= .venv-dist
DIST_VENV_BIN := $(DIST_VENV_DIR)/bin
TWINE_BOOTSTRAP ?= 1

INSTALL_SAM2 ?= 1
SAM2_REPO_URL ?= https://github.com/facebookresearch/sam2.git
SAM2_ROOT ?= third_party/sam2
SAM2_REPO_REF ?=
SAM2_DOWNLOAD_CHECKPOINTS ?= 1
SAM2_INSTALL_NOTEBOOKS ?= 0

.PHONY: help install install-nosam2 run gen-tasks build test-install check-dist publish publish-test clean

help: ## Show available targets
	@printf "Usage: make <target> [VAR=value]\n\n"
	@printf "Targets:\n"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' Makefile
	@printf "\nVars (defaults):\n"
	@printf "  VENV_DIR=%s\n" "$(VENV_DIR)"
	@printf "  PYTHON_BIN=%s\n" "$(PYTHON_BIN)"
	@printf "  DIST_DIR=%s\n" "$(DIST_DIR)"
	@printf "  DIST_VENV_DIR=%s\n" "$(DIST_VENV_DIR)"
	@printf "  TWINE_BOOTSTRAP=%s\n" "$(TWINE_BOOTSTRAP)"
	@printf "  INSTALL_SAM2=%s\n" "$(INSTALL_SAM2)"
	@printf "  SAM2_REPO_URL=%s\n" "$(SAM2_REPO_URL)"
	@printf "  SAM2_ROOT=%s\n" "$(SAM2_ROOT)"
	@printf "  SAM2_REPO_REF=%s\n" "$(SAM2_REPO_REF)"
	@printf "  SAM2_DOWNLOAD_CHECKPOINTS=%s\n" "$(SAM2_DOWNLOAD_CHECKPOINTS)"
	@printf "  SAM2_INSTALL_NOTEBOOKS=%s\n" "$(SAM2_INSTALL_NOTEBOOKS)"

install: ## Create venv + install deps + SAM2 (default on)
	@PYTHON_BIN="$(PYTHON_BIN)" \
	VENV_DIR="$(VENV_DIR)" \
	INSTALL_SAM2="$(INSTALL_SAM2)" \
	SAM2_REPO_URL="$(SAM2_REPO_URL)" \
	SAM2_ROOT="$(SAM2_ROOT)" \
	SAM2_REPO_REF="$(SAM2_REPO_REF)" \
	SAM2_DOWNLOAD_CHECKPOINTS="$(SAM2_DOWNLOAD_CHECKPOINTS)" \
	SAM2_INSTALL_NOTEBOOKS="$(SAM2_INSTALL_NOTEBOOKS)" \
	bash install.sh

install-nosam2: ## Install deps without SAM2
	@$(MAKE) install INSTALL_SAM2=0

run: ## Run pipeline (ARGS required). Example: make run ARGS="-i frames -o out -p prompts.yaml"
	@if [[ -z "$(ARGS)" ]]; then \
		echo "ERROR: ARGS is required. Example:"; \
		echo "  make run ARGS=\"-i /path/to/frames -o /path/to/output -p /path/to/prompts.yaml\""; \
		exit 1; \
	fi
	@$(VENV_BIN)/tracewave $(ARGS)

gen-tasks: ## Generate tasks (ARGS optional)
	@$(VENV_BIN)/tracewave-gen-tasks $(ARGS)

build: ## Build sdist + wheel into dist/
	@$(PYTHON_BIN) -m build

test-install: build ## Install wheel into a clean venv and verify import/CLI
	@$(PYTHON_BIN) -m venv "$(DIST_VENV_DIR)"
	@$(DIST_VENV_BIN)/pip install --upgrade pip
	@$(DIST_VENV_BIN)/pip install "$(DIST_DIR)"/*.whl
	@$(DIST_VENV_BIN)/python -c "import tracewave; print('tracewave import OK')"
	@$(DIST_VENV_BIN)/tracewave --help >/dev/null

check-dist: ## Verify dist metadata with twine
	@$(PYTHON_BIN) -m venv "$(DIST_VENV_DIR)"
	@if [[ "$(TWINE_BOOTSTRAP)" == "1" ]]; then \
		$(DIST_VENV_BIN)/pip install --upgrade pip twine packaging; \
	fi
	@$(DIST_VENV_BIN)/python -m twine check "$(DIST_DIR)"/*

publish: check-dist ## Upload dist/* to PyPI
	@$(DIST_VENV_BIN)/python -m twine upload "$(DIST_DIR)"/*

publish-test: check-dist ## Upload dist/* to TestPyPI
	@$(DIST_VENV_BIN)/python -m twine upload --repository testpypi "$(DIST_DIR)"/*

clean: ## Remove venv and caches
	@rm -rf "$(VENV_DIR)" "$(DIST_VENV_DIR)" "$(DIST_DIR)" build *.egg-info
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
