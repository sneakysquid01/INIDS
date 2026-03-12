# PROJECT_REVIEW

## Scope and Method
This review was performed as a maintenance/correctness audit of the activated IDS/IPS runtime.

Review approach:
- Read core runtime, support modules, and tests across `src/`, `web_app/`, and `tests/`.
- Verified implementation evidence (not design assumptions).
- Applied only surgical edits for correctness and operational safety.
- Re-ran full regression suite after fixes.

Final validation:
- `pytest tests/ -q` equivalent run passed.
- Current result: **143 passing tests**.

---

## 1) End-to-End Architecture Map

### 1.1 Entry Points
- HTTP entrypoint: `web_app/app.py`
  - App startup, route registration, EventBus subscriptions, pipeline/scheduler lifecycle.
- Streaming worker path (in-process): `src/pipeline/stream_processor.py` + `src/pipeline/worker.py`
  - Bootstrapped from `web_app/app.py` when pipeline is enabled and Redis is reachable.
- Optional standalone worker CLI: `python -m src.pipeline.worker`
  - Not used by default runtime, but available for process-separated operation.

### 1.2 Core Event Types
Defined in `src/core/event_bus.py`:
- `DetectionEvent`
- `RiskScoreEvent`
- `PolicyDecisionEvent`
- `ActionEvent`
- `AuditEvent`

Dispatcher:
- `EventBus.publish()` executes handlers synchronously in calling thread, with per-handler exception isolation.

### 1.3 Detection -> Risk -> Policy -> Action Pipeline
Path in `web_app/app.py`:
1. Detection produced (HTTP prediction path or streaming callback) -> `DetectionEvent`
2. `_on_detection_event`
   - Allowlist bypass check
   - `risk_engine.calculate(...)` -> `RiskScoreEvent`
3. `_on_risk_event`
   - `policy_engine.decide(...)` -> `PolicyDecisionEvent`
4. `_on_policy_decision_event`
   - Executes action for `BLOCK/TEMP_BLOCK/RATE_LIMIT` via `ActionExecutor`
   - Publishes `ActionEvent` if execution returns action
   - Records escalation hit

Related modules:
- `src/ips/risk_engine.py`
- `src/ips/policy_engine.py`
- `src/ips/action_executor.py`
- `src/prevention/*`

### 1.4 Streaming Ingestion Path
Path:
- `POST /api/ingest` in `web_app/app.py`
  - If pipeline running: `XADD` into Redis stream
  - Else: enqueue to in-memory ingestion queue
- `StreamProcessor.run()` in `src/pipeline/stream_processor.py`
  - `XREADGROUP` from stream
  - decode payload
  - feature enrichment (`enrich_single_row`)
  - evaluate all enabled+ready engines
  - aggregate result
  - callback to app (`_stream_result_callback`) -> `DetectionEvent`
  - `XACK` after successful processing

Backpressure path:
- `PipelineWorker` lag monitor updates `BackpressureController`
- `/api/ingest` returns 503 on SHEDDING mode

### 1.5 Prevention Execution Path
- Policy decisions drive `ActionExecutor.execute(...)`.
- Adapter abstraction in `src/firewall_adapters.py`:
  - Mock/UFW/nftables/webhook adapters.
- Action idempotency guard:
  - `OpsStore.has_active_block(target)` check prevents duplicate enforcement.
- Scheduled maintenance:
  - `PreventionScheduler` cleanup/reconcile loop (leader-gated in HA mode).

### 1.6 Observability Path
- Structured logging:
  - base formatter in `src/logging_config.py`
  - optional JSON formatter in `src/observability/json_logging.py`
- SIEM buffering/export:
  - `src/observability/siem_exporter.py`
  - EventBus subscriptions for detection/risk/policy/action
  - periodic auto-flush thread + `/api/siem/flush`
- Metrics:
  - `src/metrics_service.py`
  - Prometheus text from `/api/metrics`
  - includes dynamic per-engine counters

### 1.7 ML Preprocessing / Inference Path
- Synchronous prediction path:
  - `/api/predict` -> `DetectionService.predict_from_features(...)`
- Multi-engine detect path:
  - `/api/detect` -> normalize numeric -> `enrich_single_row` (best effort) -> engine registry + aggregator
- ML engine wrapper:
  - `src/detection/engines/ml_engine.py`
  - restricts model input to `FEATURE_COLUMNS` using `DEFAULT_FEATURE_ROW`

### 1.8 Threat Intelligence Path
- TI cache/manager: `src/threat_intel/feed_manager.py`
- TI engine: `src/threat_intel/ti_engine.py`
- Startup feed load from local dir (CSV/JSON) and periodic refresh thread.
- API:
  - `/api/threat-intel/stats`
  - `/api/threat-intel/lookup`

### 1.9 HA Readiness Path
- Leader election: `src/ha/leader_election.py`
- Health aggregation: `src/ha/health_check.py`
- Endpoints:
  - `/api/health`
  - `/api/health/live`
  - `/api/health/ready`
- Singleton tasks gated by leadership:
  - scheduler cleanup/reconcile
  - TI refresh
  - SIEM periodic flush

---

## 2) Dead Code and Integration Audit

### 2.1 Safe to Delete (or move to docs/examples)
These are not part of live runtime path and do not appear in app wiring:
- `src/analyze_performance.py`
- `src/capture_live_traffic.py`
- `src/generate_confusion_matrix.py`
- `src/realtime_simulation.py`
- `src/run_demo.py`
- `src/run_end_to_end_demo.py`

Rationale:
- Utility/demo scripts. Keep if used operationally; otherwise archive under `tools/` or `examples/`.

### 2.2 Imported but Unused / Redundant
- `EscalationLevel` import in `web_app/app.py` was unused.
  - Action taken: removed import.

### 2.3 Defined but Not Instantiated in Main Runtime
- `WebhookFirewallAdapter` in `src/firewall_adapters.py`
  - Capability exists but no active selection in current adapter factory.
  - Category: **Potential future capability**.
- `RedisStreamIngestionQueue` in `src/ingestion_service.py`
  - Framework exists, but app currently uses direct stream path for pipeline and in-memory queue fallback.
  - Category: **Needs integration or redesign** (overlaps with current `/api/ingest` + stream processor design).

### 2.4 Runtime-Reachable but Effectively Optional by Flags
- Pipeline runtime (`INIDS_PIPELINE_ENABLED` + Redis URL)
- JSON logging (`INIDS_JSON_LOGGING`)
- TI feed loading (`INIDS_TI_FEED_DIR`)

These are expected optional capabilities, not dead code.

### 2.5 Duplicate/Overlapping Functionality
- Ingestion has two models:
  - in-memory queue processing (`/api/ingest` + `/api/ingest/process`)
  - Redis stream pipeline processing
- This is intentional transitional architecture but creates overlap.
- Category: **Needs redesign** for long-term simplification.

---

## 3) Deep Bug Hunt Findings and Fixes

### Fixed in this maintenance pass

1. Policy history ordering bug
- File: `src/policy/policy_store.py`
- Issue:
  - `history(limit)` returned reversed earliest `limit` items instead of latest versions.
- Fix:
  - Use tail slice (`self._versions[-effective:]`) and reverse for newest-first ordering.

2. Threat intel JSON feed robustness
- File: `src/threat_intel/feed_manager.py`
- Issue:
  - Non-dict items in JSON feed array could raise errors via `.get(...)`.
- Fix:
  - Skip non-dict items safely.

3. Threat intel feed metadata growth
- File: `src/threat_intel/feed_manager.py`
- Issue:
  - `_feed_metadata` grew unbounded during periodic refresh.
- Fix:
  - Bound metadata list length (trim policy applied).

4. Rate limiter cardinality growth
- File: `src/rate_limiter.py`
- Issue:
  - `_events` dictionary could grow unbounded with many unique keys.
- Fix:
  - Add stale-key pruning when cardinality exceeds threshold.

5. Direct-script startup import-path failure
- File: `web_app/app.py`
- Issue:
  - `python web_app/app.py` could fail before `src` path was inserted.
- Fix:
  - Ensure workspace root is inserted into `sys.path` before `src.*` imports.

6. Startup/shutdown hardening
- File: `web_app/app.py`
- Improvements:
  - runtime config logging
  - guarded `__main__` startup block
  - graceful keyboard interrupt shutdown path

### High-risk areas reviewed (no breaking edits applied)
- Redis consumer-group handling and ACK timing: acceptable (ACK after successful processing).
- Scheduler/threads use daemon patterns and stop hooks: acceptable for current architecture.
- EventBus handler exception isolation: acceptable, though synchronous dispatch means handler latency propagates.

---

## 4) Streamlining and Simplification Notes

Changes applied without behavior redesign:
- Removed duplicate/unused import (`EscalationLevel`) in app wiring.
- Removed duplicate `BASE_DIR` assignment in `web_app/app.py`.
- Kept architecture stable; no control-flow redesign.

Future simplification opportunities:
- Converge ingestion models (in-memory process endpoint vs direct stream path).
- Standardize background thread lifecycle under a single runtime manager object.

---

## 5) Runtime Integrity Verification

Verified after changes:
- Streaming pipeline remains functional (code path preserved and tested).
- Detection engines still register and run.
- Escalation semantics unchanged (state machine logic untouched).
- HTTP endpoints remain backward-compatible; new endpoints additive.
- Full test suite passes.

Current verification result:
- **143 tests passed**.

Risky changes called out:
- Rate limiter stale-key pruning introduces bounded-cardinality behavior under extreme key churn.
  - Intended to prevent memory growth.
  - Does not change normal path for typical key populations.

---

## 6) Testing Coverage Gap Analysis

### Areas now covered reasonably well
- Pipeline callback/startup/backpressure
- Prevention runtime (allowlist, escalation, idempotency)
- TI manager/engine/apis
- Observability and policy runtime APIs
- HA endpoints and shutdown paths

### Remaining gaps
1. Redis outage and reconnect behavior under live worker loops
- Missing: long-running integration that simulates Redis flap during active stream consumption.

2. Consumer group pending/reclaim edge cases
- Missing: tests for stuck pending messages, crashed consumer ownership, and replay strategies.

3. Concurrency stress
- Missing: race/stress tests for EventBus fanout under high publish volume and mixed handler latency.

4. Scheduler leadership transitions
- Missing: transition tests (leader->follower->leader) with live scheduler/reconcile behavior.

5. SIEM periodic flush durability semantics
- Missing: explicit guarantees for dropped/retained events across flush failures.

6. Security/auth edge cases
- Missing: comprehensive RBAC matrix tests for every new endpoint and mixed auth modes.

### Suggested next tests
- `tests/test_stream_consumer_group_recovery.py`
- `tests/test_redis_outage_recovery.py`
- `tests/test_scheduler_leader_transition.py`
- `tests/test_event_bus_stress.py`
- `tests/test_endpoint_auth_matrix.py`

---

## 7) Capability Report for New Engineers

### What the system does end-to-end
INIDS ingests network-flow-like records, performs multi-engine detection, computes risk, applies policy decisions, and optionally executes prevention actions, while exporting metrics and SIEM-friendly events.

### Supported detection capabilities
- Supervised ML detection (`MLEngine`)
- Signature rule detection (YAML-driven)
- Threshold/rate anomaly detection
- Optional unsupervised anomaly engine (registerable)
- Threat intelligence match engine (feed-driven)

### Prevention capabilities
- Policy decisions: ALLOW/ALERT/RATE_LIMIT/TEMP_BLOCK/BLOCK
- Action execution adapters: mock, UFW, nftables (webhook adapter available)
- Allowlist bypass support (IP/CIDR)
- Escalation tracking by source IP
- Duplicate enforcement guard via active-action dedupe

### Streaming architecture
- Redis Streams consumer group processing
- At-least-once consumption model
- Backpressure controller (normal/sampling/shedding)
- HTTP ingest can route records to stream when pipeline is enabled

### ML pipeline capabilities
- Default synchronous model inference endpoint (`/api/predict`)
- Multi-engine aggregation endpoint (`/api/detect`)
- Feature enrichment for engine evaluation with safe fallback

### Threat intelligence support
- Local CSV/JSON feed loading
- In-memory IoC cache with expiration handling
- TI lookup and stats API endpoints
- Periodic refresh thread support

### Observability support
- Structured text logging and optional JSON logging
- SIEM exporter with flush API and periodic draining
- Prometheus-compatible metrics, including dynamic per-engine counters

### HA readiness
- Redis-backed leader election (standalone fallback when Redis absent)
- Leader-gated singleton tasks
- Liveness/readiness endpoints and aggregated subsystem probes

### Known limitations
- EventBus is synchronous; slow handlers can increase request/processing latency.
- Consumer-group recovery behavior for pending messages is basic.
- Ingestion architecture still has overlapping in-memory and stream models.
- Some adapters/capabilities exist but are not exposed through runtime config paths.

### Technical debt areas
- Consolidate runtime lifecycle/thread management into unified supervisor.
- Clarify ownership between detection_service path and multi-engine path.
- Reduce overlap between demo/utility scripts and production runtime code.

### Current maturity stage
- **Late prototype / early beta**
  - Strongly improved runtime completeness and test coverage.
  - Correctness baseline is now significantly stronger.
  - Still requires resilience hardening for production-grade distributed operation.

---

## 8) Summary of This Audit Pass

Implemented (surgical, non-architectural):
- Correctness fixes: policy history ordering, TI feed parser hardening, TI metadata bounds, rate limiter stale-key pruning.
- Runtime stability fix: direct-script import-path bootstrap and startup/shutdown hardening.
- Readability cleanup: removed unused import and duplicate assignment.

Validation:
- Regression suite remains green at **143 passed**.
