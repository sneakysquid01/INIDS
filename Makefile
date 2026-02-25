PYTHON ?= python
PIP ?= pip

.PHONY: setup test lint preprocess train train-all demo demo-api drift-report web conflict-check clean

setup:
	$(PIP) install -r requirements.txt

# Run project tests
test:
	$(PYTHON) -m pytest -q

# Basic import and syntax check
lint:
	$(PYTHON) -m compileall src web_app

# Build preprocessing artifacts
preprocess:
	$(PYTHON) src/preprocess_train.py

# Train baseline models
train:
	$(PYTHON) -m src.train_cli --suite baseline

# Train complete model suite
train-all:
	$(PYTHON) -m src.train_cli --suite full --include-multiclass

# Run quick terminal inference demo
demo:
	$(PYTHON) src/run_demo.py

# Run API-based end-to-end demo flow (requires running web app)
demo-api:
	$(PYTHON) src/run_end_to_end_demo.py

# Generate drift report from baseline vs current dataset
drift-report:
	$(PYTHON) src/drift_monitor.py

# Start web application
web:
	$(PYTHON) web_app/app.py

# Detect unresolved Git merge conflict markers in tracked source/docs files
conflict-check:
	rg -n "^(<<<<<<<|=======|>>>>>>>)" -g '!*.map' src tests web_app docs .env.example Makefile pyproject.toml requirements.txt || true

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
