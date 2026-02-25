PYTHON ?= python
PIP ?= pip

<<<<<<< codex/evaluate-repository-quality-hm1ro9
.PHONY: setup test lint preprocess train train-all demo demo-api drift-report web conflict-check clean
=======
.PHONY: setup test lint preprocess train train-all demo web clean
>>>>>>> main

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

<<<<<<< codex/evaluate-repository-quality-hm1ro9
# Run API-based end-to-end demo flow (requires running web app)
demo-api:
	$(PYTHON) src/run_end_to_end_demo.py

# Generate drift report from baseline vs current dataset
drift-report:
	$(PYTHON) src/drift_monitor.py

=======
>>>>>>> main
# Start web application
web:
	$(PYTHON) web_app/app.py

<<<<<<< codex/evaluate-repository-quality-hm1ro9
# Detect unresolved Git merge conflict markers in tracked source/docs files
conflict-check:
	rg -n "^(<<<<<<<|=======|>>>>>>>)" -g '!*.map' src tests web_app docs .env.example Makefile pyproject.toml requirements.txt || true

=======
>>>>>>> main
clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
