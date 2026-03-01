# INIDS Operations Runbook

## 1. Start locally
1. Install dependencies: `pip install -r requirements.txt`
2. Copy env template: `cp .env.example .env` (or set env vars in shell)
3. Run app: `python -m web_app.app`

## 2. Health checks
- `GET /api/health`
- `GET /api/metrics` (requires analyst role when auth enabled)

## 3. Detection and ingestion checks
- Predict one record: `POST /api/predict`
- Queue records: `POST /api/ingest`
- Process queue: `POST /api/ingest/process`

## 4. Prevention and policy checks
- Read/update policy: `GET|POST /api/policy`
- List actions: `GET /api/actions`
- Cleanup expired actions: `POST /api/actions/cleanup`

## 5. Drift and demo tooling
- Generate drift report: `make drift-report`
- Run API demo flow: `make demo-api`

## 6. Docker Compose
From repo root:
- `docker compose -f deploy/compose/docker-compose.yml up`

## 7. Incident review checklist
- Check `/api/alerts` for latest critical/high alerts.
- Review `/api/audit` for policy/action changes.
- Confirm prevention mode and `dry_run` policy before enabling execution.
