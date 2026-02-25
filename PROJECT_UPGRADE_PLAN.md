# INIDS → Production-Style IDS/IPS Upgrade Plan

## 1) Target Architecture (Tailored to Current Repo)

```text
                           ┌────────────────────────────┐
                           │       Web UI (Flask)       │
                           │  dashboard, alerts, cases  │
                           └──────────────┬─────────────┘
                                          │ REST/WebSocket
                                 ┌────────▼────────┐
                                 │   API Layer     │
                                 │ auth/rbac/audit │
                                 └───┬──────────┬──┘
                                     │          │
                         ┌───────────▼───┐   ┌──▼──────────────────────┐
                         │ Detection Svc │   │ Prevention Orchestrator │
                         │ model + rules │   │ block/quarantine/ttl    │
                         └──────┬────────┘   └──────┬───────────────────┘
                                │                   │
                    ┌───────────▼───────────┐   ┌──▼──────────────────┐
                    │ Feature Pipeline       │   │ Response Adapters   │
                    │ normalize/validate     │   │ nftables/iptables   │
                    └───────────┬───────────┘   │ cloud-fw/webhook     │
                                │               └───────────────────────┘
      ┌─────────────────────────▼──────────────────────────┐
      │ Ingestion Layer                                     │
      │ scapy capture | zeek/suricata logs | pcap replay    │
      └─────────────────────────┬──────────────────────────┘
                                │
                        ┌───────▼─────────────────────────────────────────┐
                        │ Data + Ops                                      │
                        │ postgres (alerts/incidents/audit), redis queue, │
                        │ object storage (models), prometheus + grafana   │
                        └──────────────────────────────────────────────────┘
```

### Why this design matches your codebase
- Reuses your existing model artifacts, schema alignment, and Flask app as the base UI/API surface.
- Isolates prevention logic from prediction logic so you can run monitor-only mode safely.
- Adds operational components (queue, DB, metrics) needed to claim production readiness.

---

## 2) Proposed Repository Structure

```text
INIDS/
├─ data/
├─ models/
├─ results/
├─ src/
│  ├─ training/
│  │  ├─ preprocess.py
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  └─ registry.py
│  ├─ detection/
│  │  ├─ service.py              # model + rule ensemble
│  │  ├─ threshold_policy.py
│  │  ├─ drift_monitor.py
│  │  └─ explainability.py
│  ├─ prevention/
│  │  ├─ orchestrator.py
│  │  ├─ policy_engine.py
│  │  ├─ adapters/
│  │  │  ├─ nftables.py
│  │  │  ├─ ufw.py
│  │  │  └─ mock_adapter.py
│  │  └─ rollback.py
│  ├─ ingestion/
│  │  ├─ live_capture.py
│  │  ├─ zeek_parser.py
│  │  ├─ suricata_parser.py
│  │  └─ replay.py
│  ├─ pipeline/
│  │  ├─ feature_builder.py
│  │  ├─ validators.py
│  │  └─ schema_contract.py
│  ├─ api/
│  │  ├─ app.py
│  │  ├─ routes/
│  │  │  ├─ alerts.py
│  │  │  ├─ incidents.py
│  │  │  ├─ policies.py
│  │  │  └─ health.py
│  │  └─ deps.py
│  ├─ storage/
│  │  ├─ models_repo.py
│  │  ├─ alerts_repo.py
│  │  └─ audit_repo.py
│  └─ common/
│     ├─ schema.py
│     ├─ label_utils.py
│     ├─ logging.py
│     └─ settings.py
├─ web_app/
│  ├─ templates/
│  └─ static/
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  ├─ e2e/
│  └─ fixtures/
├─ deploy/
│  ├─ docker/
│  ├─ k8s/
│  └─ compose/
├─ docs/
│  ├─ architecture.md
│  ├─ operations_runbook.md
│  ├─ threat_model.md
│  └─ api_contracts.md
├─ Makefile
├─ pyproject.toml
└─ README.md
```

---

## 3) 12-Week Milestone Plan (Final-Year-Project Friendly)

## Phase 0 (Week 1): Foundation Cleanup
**Goal:** make developer workflow reproducible.
- Package setup (`pyproject.toml`) and imports fixed so tests run without manual `PYTHONPATH`.
- Merge training scripts into one CLI (`train`, `evaluate`, `export`).
- Add `.env.example` and settings module.

**Deliverable:** one-command local setup + CI green.

## Phase 1 (Week 2-3): Detection Service v1
**Goal:** stable real-time detection API.
- Build `detection/service.py` with model loading, threshold profiles, and rule fallback.
- Add alert object schema with severity and confidence bands.
- Add endpoints: `/predict`, `/alerts`, `/health`.

**Deliverable:** live alerts generated via API + dashboard.

## Phase 2 (Week 4-5): Ingestion + Feature Pipeline
**Goal:** process real traffic/logs continuously.
- Add ingestion connectors (scapy + zeek/suricata parser).
- Add queue (Redis/Kafka-lite) between ingestion and detection.
- Validate schema contract at pipeline boundary.

**Deliverable:** streaming detection with back-pressure handling.

## Phase 3 (Week 6-7): Prevention Engine v1
**Goal:** transition IDS → IPS.
- Create policy engine: monitor-only, approve-before-block, auto-block.
- Implement firewall adapter (start with mock + ufw/nftables).
- Add TTL block rules + auto-unblock rollback.

**Deliverable:** safe prevention actions with audit trail.

## Phase 4 (Week 8-9): MLOps and Trust
**Goal:** prove robustness and maintainability.
- Add model registry metadata + model version pinning.
- Add drift monitoring and retraining trigger policy.
- Add explainability (top contributing features).

**Deliverable:** model governance dashboard panel.

## Phase 5 (Week 10): Security and Access Control
**Goal:** make system operationally safe.
- Add authentication + RBAC (admin/analyst/viewer).
- Add tamper-evident audit logs for every policy/action change.
- Add rate limiting and API key support for integrations.

**Deliverable:** multi-user operational control plane.

## Phase 6 (Week 11-12): Productionization + Demo Polish
**Goal:** final defense-ready project.
- Docker compose stack (app + db + queue + metrics).
- SLO dashboard (latency, alert volume, action success rate).
- Final report assets: architecture, threat model, demo script, KPI results.

**Deliverable:** end-to-end live demo with measurable KPIs.

---

## 4) KPI Targets (What “works wonderfully” should mean)

Use these in your final report and viva:
- Detection F1 (binary) ≥ 0.98 on held-out set.
- False Positive Rate ≤ 2.0% in replay benchmark.
- End-to-end alert latency p95 ≤ 2 seconds.
- Prevention action success rate ≥ 99% (mock/staging).
- Mean Time To Respond (MTTR) ≤ 10 seconds in auto mode.
- Zero untracked blocking actions (100% audit coverage).

---

## 5) Priority Backlog (First 15 Tasks)

1. Create `pyproject.toml` + make `src` importable.
2. Add `Makefile` commands: `make setup test train run`.
3. Merge `train_model.py` + `train_all_models.py` into one CLI.
4. Define `Alert` and `PolicyDecision` pydantic/dataclass schemas.
5. Build detection service with threshold profiles (`strict`, `balanced`, `lenient`).
6. Add alert persistence (Postgres or SQLite for MVP).
7. Add `/api/alerts` list/filter endpoint.
8. Build prevention orchestrator with `dry_run` toggle.
9. Implement mock firewall adapter and unit tests.
10. Implement UFW/nftables adapter and integration tests.
11. Add scheduler for TTL unblock jobs.
12. Add RBAC + JWT login.
13. Add Prometheus metrics endpoint.
14. Add drift monitor job + daily report.
15. Add end-to-end demo script (capture → detect → prevent → rollback).

---

## 6) Demo Storyline (for Evaluation Day)

1. Start stack with one command (`docker compose up`).
2. Replay mixed normal/attack traffic.
3. Show live dashboard alert feed and confidence.
4. Flip policy from monitor-only to auto-prevent.
5. Trigger attack burst and show automatic temporary block.
6. Show audit log entry + automatic unblock after TTL.
7. Show metrics panel and final KPI table.

---

## 7) Risk Register + Mitigations

- **High false positives** → use confidence calibration + class thresholds + allowlist.
- **Unsafe blocking** → staged rollout, dry-run mode, and max-block caps.
- **Data drift** → periodic drift checks and retraining policy.
- **Pipeline lag** → queue + async workers + p95 latency alarms.
- **Evaluator asks “production where?”** → show RBAC, audit logs, health checks, and SLO dashboards.

---

## 8) Minimum “Production-Style” Definition (for this project)

Call it production-style once all below are true:
- Reproducible setup and CI tests pass.
- Real-time ingestion + detection + alert storage work continuously.
- Prevention is policy-controlled, reversible, and audited.
- Role-based access control is enabled.
- Operational observability exists (health + metrics + logs).

