.PHONY: all dev capture visualize clean setup

PYTHON = .venv/bin/python

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

capture:
	$(PYTHON) run_capture.py

visualize:
	$(PYTHON) run_visualize.py

all: capture visualize

# Development mode: single short prompt, fewer generation tokens
dev:
	$(PYTHON) run_capture.py --prompt 0 --max-gen-tokens 10
	$(PYTHON) run_visualize.py --prompt 0

clean:
	rm -rf output/
