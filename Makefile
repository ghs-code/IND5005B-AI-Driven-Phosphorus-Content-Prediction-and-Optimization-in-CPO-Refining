ifneq (,$(wildcard .env))
include .env
export
endif

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: help init-config install ui test verify

help:
	@echo "Available targets:"
	@echo "  make init-config # copy .env.example to .env if missing"
	@echo "  make install     # install project dependencies"
	@echo "  make ui          # launch the Streamlit enterprise workbench"
	@echo "  make test        # run unittest suite"
	@echo "  make verify      # run compile and test checks"

init-config:
	@test -f .env || cp .env.example .env
	@echo "Config file ready: .env"

install:
	$(PIP) install -U pip
	$(PIP) install -e .

ui:
	$(PYTHON) scripts/run_ui.py

test:
	$(PYTHON) -m unittest discover -s tests

verify:
	$(PYTHON) -m compileall -q src scripts tests
	$(PYTHON) -m unittest discover -s tests
	$(PYTHON) -c "import cpo_phosphorus.workflows as w; print('workflow imports ok')"
