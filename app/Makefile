PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

SAM2_URL ?= https://huggingface.co/facebook/sam2-hiera-tiny/resolve/main/sam2_hiera_tiny.pt
SAM2_WEIGHTS ?= src/sam2_configs/sam2_hiera_tiny.pt
SAM2_SHA256 ?=

.PHONY: help venv install install-full sam2-weights run clean

help:
	@echo "Targets:"
	@echo "  make venv          Create a local virtual environment"
	@echo "  make install       Install core dependencies in the venv"
	@echo "  make install-full  Install core + optional extras (sam2, yaml)"
	@echo "  make sam2-weights  Download SAM2 weights to src/sam2_configs/"
	@echo "  make run           Launch TraceWave"
	@echo "  make clean         Remove build and cache artifacts"

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e .

install-full: venv sam2-weights
	$(PIP) install -e ".[sam2,yaml]"

sam2-weights: $(SAM2_WEIGHTS)

$(SAM2_WEIGHTS):
	@mkdir -p $(dir $@)
	@if [ -f "$@" ]; then \
		echo "SAM2 weights already present at $@"; \
	else \
		echo "Downloading SAM2 weights to $@"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -L -o "$@" "$(SAM2_URL)"; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -O "$@" "$(SAM2_URL)"; \
		else \
			echo "Error: curl or wget is required to download weights."; \
			exit 1; \
		fi; \
	fi
	@if [ -n "$(SAM2_SHA256)" ]; then \
		if command -v sha256sum >/dev/null 2>&1; then \
			echo "$(SAM2_SHA256)  $@" | sha256sum -c -; \
		elif command -v shasum >/dev/null 2>&1; then \
			echo "$(SAM2_SHA256)  $@" | shasum -a 256 -c -; \
		else \
			echo "Warning: sha256sum/shasum not found; skipping checksum verification."; \
		fi; \
	fi

run:
	$(PY) -m src.tracewave

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache __pycache__
