PYTHON ?= python
PIP ?= pip

.PHONY: setup test lint preprocess train train-all demo web clean

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

# Start web application
web:
	$(PYTHON) web_app/app.py

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
