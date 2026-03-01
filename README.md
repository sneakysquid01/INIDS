# Intelligent Network Intrusion Detection System (INIDS)

INIDS is a machine-learning based intrusion detection project built around the NSL-KDD dataset. It includes:
- Model training and comparison utilities
- A Flask web app for live and batch prediction
- Operational APIs for alerts, policy, actions, audit, metrics, and ingestion
- Tests for core services and API behavior

## Project layout
- `src/`: training, detection, ingestion, operations, and utility services
- `web_app/`: Flask app, templates, and static assets
- `tests/`: pytest suite
- `data/`: NSL-KDD input files
- `models/`: trained models and preprocessing artifacts
- `results/`: training outputs and registry data

## Requirements
- Python 3.10+
- pip
- Optional: `make` for shortcut commands

## Quick start
1. Create and activate a virtual environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run tests:
```bash
python -m pytest -q
```
4. Start the web app:
```bash
python -m web_app.app
```
5. Open:
- `http://127.0.0.1:5000/`

## Make targets
If `make` is available:

```bash
make setup
make test
make lint
make preprocess
make train
make train-all
make web
make conflict-check
```

## Key API endpoints
- `GET /api/health`
- `POST /api/predict`
- `GET /api/alerts`
- `GET|POST /api/policy`
- `GET /api/actions`
- `POST /api/actions/cleanup`
- `GET /api/audit`
- `GET /api/metrics`
- `POST /api/ingest`
- `POST /api/ingest/log`
- `POST /api/ingest/process`

## Environment configuration
Copy `.env.example` to `.env` and adjust values as needed:
- host/port/debug
- API key roles
- rate limits
- firewall adapter
- operations DB path

## Notes
- Use module mode (`python -m web_app.app`) for reliable startup across platforms.
- Auth is disabled unless API keys are configured in environment variables.
