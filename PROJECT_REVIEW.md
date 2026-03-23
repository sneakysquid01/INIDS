# INIDS Deep Maintenance and Correctness Review

Date: 2026-03-24
Scope: Full repository maintenance and correctness audit focused on runtime safety, dead code, and maintainability without architectural rewrites.

## 1) Architecture Map

### 1.1 Entry points
- HTTP/Web runtime: `web_app/app.py`
- Standalone pipeline worker CLI: `src/pipeline/worker.py`
- Training/preprocessing CLIs:
  - `src/preprocess_train.py`
  - `src/train_cli.py`
  - wrappers: `src/train_model.py`, `src/train_all_models.py`
- Demo/ops scripts:
  - `src/run_end_to_end_demo.py` (duplicate of `tools/run_end_to_end_demo.py`)
  - `src/run_demo.py`, `src/realtime_simulation.py`, `src/capture_live_traffic.py`

### 1.2 Core modules and responsibilities
- Event bus + event contracts: `src/core/event_bus.py`
- Detection inference (single-model path): `src/detection_service.py`
- Multi-engine framework:
  - base/result types: `src/detection/engine_base.py`
  - registry: `src/detection/engine_registry.py`
  - fusion/aggregation: `src/detection/aggregator.py`
  - engines: `src/detection/engines/{ml_engine,signature_engine,anomaly_engine,threshold_engine}.py`
  - threat-intel engine: `src/threat_intel/ti_engine.py`
- Prevention path:
  - risk scoring: `src/ips/risk_engine.py`
  - policy decisions: `src/ips/policy_engine.py`
  - action execution: `src/ips/action_executor.py`
  - cleanup/reconcile scheduler: `src/ips/scheduler.py`
  - service policy config: `src/prevention_service.py`
  - allowlist/escalation/FP management: `src/prevention/*`
- Streaming runtime:
  - stream consumer: `src/pipeline/stream_processor.py`
  - backpressure control: `src/pipeline/backpressure.py`
  - worker orchestration: `src/pipeline/worker.py`
  - ingestion queues: `src/ingestion_service.py`
- Persistence:
  - Ops DB (SQLite/Postgres): `src/ops_store.py`
  - policy versioning: `src/policy/policy_store.py`
  - model registry file: `src/model_registry.py`
- Security and controls:
  - RBAC API-key guard: `src/auth_service.py`
  - request rate limiting: `src/rate_limiter.py`
- Observability + HA:
  - metrics: `src/metrics_service.py`
  - JSON logging: `src/observability/json_logging.py`
  - SIEM buffer/export: `src/observability/siem_exporter.py`
  - leader election: `src/ha/leader_election.py`
  - health checks: `src/ha/health_check.py`

### 1.3 End-to-end runtime flow

```
HTTP /api/predict or /api/ingest/*
    -> normalize / parse / enrich features
    -> detection engines (single-model or multi-engine)
    -> DetectionEvent on EventBus
    -> RiskEngine => RiskScoreEvent
    -> PolicyEngine => PolicyDecisionEvent
    -> ActionExecutor => ActionEvent + firewall adapter calls
    -> OpsStore persistence (alerts/actions/audit)
    -> Metrics + SIEM exporter + websocket realtime emit
```

### 1.4 Streaming ingestion path
- `POST /api/ingest`
  - If pipeline active: writes JSON payloads to Redis stream `SETTINGS.pipeline_stream_key`.
  - If pipeline inactive: queues records in `InMemoryIngestionQueue` or `RedisStreamIngestionQueue`.
- `StreamProcessor.run()`
  - `XREADGROUP` from consumer group
  - decode -> feature enrichment -> `EngineRegistry.evaluate_all()` -> `EngineAggregator.aggregate()`
  - callback publishes `DetectionEvent`
  - `XACK` only after successful processing

### 1.5 Detection -> Risk -> Policy -> Action pipeline
- `web_app/app.py` subscribes:
  - `DetectionEvent` -> `_on_detection_event()` -> `risk_engine.calculate()`
  - `RiskScoreEvent` -> `_on_risk_event()` -> `policy_engine.decide()`
  - `PolicyDecisionEvent` -> `_on_policy_decision_event()` -> `action_executor.execute()`
- Prevention actions persisted in `ops_store.actions`, audited in `ops_store.audits`.

### 1.6 Observability path
- Metrics counters/gauges/histograms via `MetricsService`
- `/api/metrics` Prometheus exposition
- EventBus subscribers mirror detection/risk/policy/action events to `SiemExporter`
- Optional JSON structured logs (`INIDS_JSON_LOGGING`)
- Health probes: `/api/health`, `/api/health/live`, `/api/health/ready`

### 1.7 ML preprocessing / training / inference
- Preprocessing: one-hot categorical + standard scaling (`src/preprocess_train.py`)
- Training: model suite in `src/train_cli.py` (RF/GB/DT/AB/MLP)
- Runtime inference:
  - single-model path via `DetectionService`
  - multi-engine path via `MLEngine` in `EngineRegistry`
- Drift monitor utility: `src/drift_monitor.py` (PSI report)

## 2) Dead Code Discovery (with evidence)

### 2.1 Never-imported modules (static import graph)
Evidence from AST import graph scan:
- `src/analyze_performance.py`
- `src/capture_live_traffic.py`
- `src/generate_confusion_matrix.py`
- `src/imbalance_handler.py`
- `src/integrations/__init__.py`
- `src/realtime_simulation.py`
- `src/run_demo.py`
- `src/run_end_to_end_demo.py`
- `src/train_all_models.py`
- `src/train_model.py`
- `tools/analyze_performance.py`
- `tools/capture_live_traffic.py`
- `tools/generate_confusion_matrix.py`
- `tools/realtime_simulation.py`
- `tools/run_demo.py`
- `tools/run_end_to_end_demo.py`

Interpretation: these are mostly CLI/demo utilities, not runtime-integrated modules.

### 2.2 Duplicate functionality
Evidence (content hash equality):
- `src/capture_live_traffic.py` == `tools/capture_live_traffic.py`
- `src/run_end_to_end_demo.py` == `tools/run_end_to_end_demo.py`

Likely duplicate ownership boundary (`src` vs `tools`) rather than separate behaviors.

### 2.3 Imported but unused symbols (example evidence)
- `src/pipeline/worker.py`: `json` import unused
- `src/observability/siem_exporter.py`: `time` import unused
- `src/feature_engineering.py`: `NUMERIC_FEATURES` import unused
- `src/policy/policy_store.py`: `field` import unused
- `src/preprocess_train.py`: `Pipeline` import unused
- `src/analyze_performance.py` + `tools/analyze_performance.py`: several unused imports

### 2.4 Categorization
- Safe to delete (after deprecation notice):
  - one copy of each exact duplicate script in `src/` or `tools/`
  - unused marker package `src/integrations/__init__.py` if not part of planned public API
- Needs integration:
  - `imbalance_handler.py` (currently detached from training CLI)
  - `capture_live_traffic.py` and `realtime_simulation.py` (demo-only, not ingestion-integrated)
- Needs redesign:
  - module demo APIs that synthesize placeholder values not backed by authoritative runtime state
- Potential future capability:
  - performance analysis / confusion matrix scripts (offline model QA utilities)

## 3) Deep Bug Hunt and Fixes Applied

### 3.1 High-impact correctness fixes
1. SocketIO fallback import bug fixed
- File: `web_app/app.py`
- Issue: fallback `_NoopSocketIO` lacked `.on()` so module import could fail when `flask_socketio` missing.
- Fix: added no-op decorator-compatible `on()`.

2. Runtime-unreachable startup path fixed
- File: `web_app/app.py`
- Issue: two `if __name__ == "__main__"` blocks; earlier one could start server before remaining routes/security bootstrap executed.
- Fix: removed duplicate mid-file main block; consolidated startup into single final main block and moved module broadcaster startup into shared helper.

3. Module APIs returning 500 due wrong engine object assumptions fixed
- File: `web_app/app.py`
- Affected endpoints:
  - `/api/modules/multi-engine`
  - `/api/modules/engine-playground`
- Issue: code used `e.id/e.enabled` but registry returns dict entries.
- Fix: switched to dict-key access (`engine_id`, `enabled`, `ready`, `engine_type`).

4. Dashboard metrics timestamp parsing crash fixed
- File: `web_app/app.py`
- Endpoint: `/api/dashboard/metrics`
- Issue: attempted `float(created_at)` on ISO datetime strings.
- Fix: added `_to_epoch_seconds()` with ISO parsing fallback + safer counting logic.

5. Streaming bypass inconsistency in log ingestion fixed
- File: `web_app/app.py`
- Endpoint: `/api/ingest/log`
- Issue: did not honor active stream pipeline path/backpressure.
- Fix: aligned with `/api/ingest` behavior: stream to Redis when pipeline is active, enforce shedding response.

6. Concurrency hardening
- Files:
  - `src/rate_limiter.py`
  - `src/ingestion_service.py`
- Issue: shared mutable structures were unguarded in threaded runtime.
- Fix: added locks around mutation/read paths.

7. Consumer-group recovery improvement
- File: `src/pipeline/stream_processor.py`
- Issue: stale pending messages from dead consumers were not reclaimed.
- Fix: added best-effort reclaim loop using `XPENDING` + `XCLAIM` (guarded/optional for client compatibility).

### 3.2 Maintainability cleanup (non-functional)
- Removed clearly unused imports in several modules listed in section 2.3.
- Added support for `INIDS_OPS_DB_PATH` env alias in settings loader (`src/settings.py`) while preserving `OPS_DB_PATH`.

## 4) Streamlining and Simplification Performed
- Unified engine-list module responses to registry’s actual dict schema.
- Unified ingestion log path behavior with existing streaming/non-streaming ingestion semantics.
- Reduced duplicate startup complexity by single main-block execution path.
- Removed low-value unused imports to lower cognitive noise.

## 5) Runtime Integrity Verification

### 5.1 Verification executed
- Syntax/bytecode compile:
  - `python -m compileall -q src web_app` -> PASS
- Focused pytest subset (no tmpdir fixture use):
  - `tests/test_rate_limiter.py`
  - `tests/test_ingestion_service.py`
  - `tests/test_stream_consumer_group_recovery.py`
  - `tests/test_feature_engineering_runtime.py`
  - `tests/test_app_import_path.py`
  - Result: 25 passed
- Manual API smoke checks (Flask test client):
  - `/api/modules/multi-engine` -> 200
  - `/api/modules/engine-playground` -> 200
  - `/api/dashboard/metrics` -> 200
  - `/api/modules/alert-lifecycle` -> 200
- Forced import scenario without `flask_socketio`: module load succeeded.

### 5.2 Environment limitations during broader test execution
- Full/expanded pytest runs hit filesystem permission failures in temp/cache directories (`PermissionError` on pytest tmpdir cleanup and `.pytest_cache`).
- This prevented reliable pass/fail determination for tests requiring `tmp_path` fixtures in this environment.

### 5.3 Risky change callouts
- Stale pending reclaim in `StreamProcessor` introduces additional Redis calls (`XPENDING`/`XCLAIM`) during idle periods.
  - Risk: minor operational overhead and possible duplicate processing in misconfigured consumer groups.
  - Mitigation: best-effort guarded implementation; only runs periodically and only when APIs exist.

## 6) Testing Coverage Gap Analysis

### 6.1 Modules with weak or no direct tests
- `src/detection/engines/ml_engine.py`
- `src/detection/engines/threshold_engine.py`
- `src/pipeline/worker.py`
- `src/logging_config.py`
- Offline scripts (`run_demo`, `capture_live_traffic`, `analyze_performance`, etc.)

### 6.2 Gaps by risk area
- Concurrency paths:
  - no dedicated stress tests for `InMemoryRateLimiter` lock correctness
  - limited tests for concurrent ingestion queue producer/consumer pressure
- Redis outage/recovery:
  - no end-to-end test for newly added `XPENDING/XCLAIM` reclaim path
  - no test for Redis reconnect while pipeline is already running
- Duplicate delivery handling:
  - idempotency covered for action executor, but not for stream-processor re-delivery + event bus effects end-to-end
- Escalation downgrade behavior:
  - basic cooldown de-escalation exists, but not long-run downgrade across multiple severity mixes

### 6.3 Recommended new tests
1. `test_stream_processor_reclaims_stale_pending_messages()` with fake redis implementing `xpending_range/xclaim`.
2. `test_api_dashboard_metrics_parses_iso_created_at()` regression guard.
3. `test_import_without_flask_socketio()` explicit guard for fallback decorators.
4. `test_ingest_log_routes_to_stream_when_pipeline_enabled()` parity with `/api/ingest`.
5. `test_rate_limiter_thread_safety_under_parallel_requests()`.
6. `test_engine_playground_and_multi_engine_module_schema_consistency()`.

## 7) Complete Project Capability Report

### 7.1 What the system does end-to-end
INIDS accepts network-feature events (API/log parsers/stream), performs ML + multi-engine detection, computes risk, applies policy decisions, executes prevention actions (mock/UFW/nftables/webhook), persists operational state, and exposes health/metrics/SIEM/web dashboard interfaces.

### 7.2 Supported detection capabilities
- Supervised ML classifier inference (`DetectionService`, `MLEngine`)
- Signature rules (`SignatureEngine`, YAML rules)
- Threshold/rate heuristics (`ThresholdEngine`)
- Unsupervised anomaly detection with buffered auto-fit (`AnomalyEngine`)
- Threat intelligence IOC matching (`TIEngine`)
- Multi-engine vote aggregation strategies (`ANY_TRIGGER`, `MAJORITY`, `UNANIMOUS`, `WEIGHTED`)

### 7.3 Prevention capabilities
- Policy-driven decisions: allow, alert, rate-limit, temp block, block, pending approval
- Firewall adapter abstraction:
  - in-memory mock
  - UFW
  - nftables
  - webhook-based external control
- Approval gate for pending blocks
- Allowlist bypass
- Expired-action cleanup + reconciliation

### 7.4 Streaming architecture
- Redis Streams consumer group model with at-least-once semantics
- In-process pipeline worker + backpressure controller
- Stream callback to EventBus for full risk/policy/action flow
- Best-effort stale pending claim support (new)

### 7.5 ML pipeline capabilities
- NSL-KDD preprocessing pipeline (one-hot + scaling)
- Binary and optional multiclass split generation
- Multi-model training CLI with registry + artifact outputs
- Basic drift report generation (PSI)

### 7.6 Threat intelligence support
- CSV and JSON feed loading
- In-memory indicator cache with TTL purge
- Feed summary and lookup APIs
- Optional scheduled refresh (leader-only)

### 7.7 Observability support
- Prometheus metrics endpoint
- Structured JSON logging option
- SIEM export buffer + flush API + leader auto-flush thread
- Websocket event emissions for realtime UI
- Health/readiness/live endpoints with subsystem probes

### 7.8 HA readiness
- Redis-based leader election with TTL renewal
- Standalone-leader fallback when Redis unavailable
- Leader-gated singleton tasks (cleanup/reconcile/TI refresh/SIEM flush)

### 7.9 Known limitations
- Demo module APIs still contain synthetic placeholders and inconsistent domain fields (`classification`, demo-only status enums).
- Partial test-environment fragility around temporary directory permissions prevented full automated regression execution here.
- Several scripts are duplicated across `src/` and `tools/` with unclear ownership boundaries.

### 7.10 Technical debt areas
- Very large `web_app/app.py` mixes transport, orchestration, and demo UI data assembly.
- Mixed casing/enum conventions in status/action fields across modules.
- Broad exception handling in endpoint paths may hide precise failure causes.
- Limited explicit integration tests for thread/race behaviors under load.

### 7.11 Current maturity stage
Stage: Late prototype / early production hardening.
- Strengths: event-driven core, policy/action pipeline, HA/observability scaffolding, broad test suite footprint.
- Needed next: module decomposition, stronger contract tests, cleanup of duplicate utilities, and consistency enforcement for statuses/schemas.

## 8) Change Log (This Audit)

Files edited in this pass:
- `web_app/app.py`
- `src/pipeline/stream_processor.py`
- `src/rate_limiter.py`
- `src/ingestion_service.py`
- `src/settings.py`
- `src/pipeline/worker.py`
- `src/observability/siem_exporter.py`
- `src/feature_engineering.py`
- `src/policy/policy_store.py`
- `src/preprocess_train.py`
- `src/analyze_performance.py`
- `tools/analyze_performance.py`

No architecture rewrite was performed; changes were surgical and backward-compatible where possible.
