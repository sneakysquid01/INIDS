# SYSTEM_DEEP_REVIEW

## Scope
Full deep review focused on correctness and runtime integrity (not architecture redesign): ingestion, detection, event propagation, scoring/policy/enforcement, false-positive handling, TI integration, streaming, HA/health, persistence, and observability.

## Current Maturity Stage
Late prototype / pre-production hardening.

Strengths:
- End-to-end pipeline exists and is mostly wired.
- Strong test footprint in repository.
- Core detection -> risk -> policy -> enforcement flow is present.

Gaps:
- Several partially wired demo/module endpoints.
- Some runtime correctness mismatches between legacy and event-driven paths.
- Streaming backpressure and queue durability primitives exist but are not fully exercised in request paths.

## Phase 1: Architecture Summary (Runtime Mental Model)

### Ingestion paths
- HTTP direct prediction: `web_app\app.py` -> `api_predict` -> `DetectionService.predict_from_features`.
- HTTP queue ingestion: `api_ingest`, `api_ingest_log`, `api_ingest_process`.
- Streaming ingestion: `api_ingest`/`api_ingest_log` -> Redis stream (`_stream_ingest_records`) -> `src\pipeline\stream_processor.py`.
- In-memory queue fallback: `src\ingestion_service.py:InMemoryIngestionQueue`.

### Detection and aggregation
- Engines registered in `web_app\app.py`: ML, signature, threshold, anomaly, threat intel.
- Engine execution: `src\detection\engine_registry.py:evaluate_all`.
- Verdict fusion: `src\detection\aggregator.py` (default `ANY_TRIGGER`).

### Event bus routing
- Core event types: `src\core\event_bus.py`.
- Runtime chain: `DetectionEvent -> RiskScoreEvent -> PolicyDecisionEvent -> ActionEvent`.
- Subscriptions registered in `web_app\app.py` for SIEM and realtime outputs.

### Risk -> policy -> enforcement
- Risk scoring: `src\ips\risk_engine.py`.
- Policy decisions: `src\ips\policy_engine.py`.
- Action execution/idempotency/approval/cleanup: `src\ips\action_executor.py`.
- Cleanup/reconciliation scheduler: `src\ips\scheduler.py`.

### TI, FP management, escalation
- TI cache/feeds: `src\threat_intel\feed_manager.py`.
- TI engine: `src\threat_intel\ti_engine.py` (ready only when cache populated).
- FP suppression: `src\prevention\false_positive_manager.py`.
- Escalation state machine: `src\prevention\escalation_tracker.py`.

### Health/HA/observability/persistence
- Health probes and readiness: `src\ha\health_check.py`, wiring in `web_app\app.py`.
- Leader election: `src\ha\leader_election.py`.
- Metrics and SIEM buffering/export: `src\metrics_service.py`, `src\observability\siem_exporter.py`.
- Persistent operational store: `src\ops_store.py`.

## Phase 2: Runtime Path Verification Findings

### Startup sequence
- `web_app\app.py` main block initializes model loading, scheduler, pipeline, module broadcaster.
- Health probes and signal handlers registered during module import.

### Engine registration and activation
- Engines are registered at startup; anomaly and TI readiness gates are honored.
- Anomaly engine activation can occur dynamically from normal traffic samples.

### Streaming worker lifecycle
- `_ensure_pipeline_started` creates `StreamProcessor` + `PipelineWorker`.
- Worker starts lag monitor + processing thread.
- Shutdown path attempts clean stop (`_shutdown_runtime`).

### Event propagation chain
- Verified event bus subscribers for detection/risk/policy/action + SIEM + realtime.

### Action execution conditions
- Enforcement only on `{BLOCK, TEMP_BLOCK, RATE_LIMIT, PENDING_BLOCK}`.
- `ActionExecutor` idempotency checks `ops_store.has_active_block`.

### FP suppression effect
- Signature engine accepts FP manager and can suppress rules.
- FP feedback endpoints call manager directly.

## Phase 3: Bugs / Risks Found

### Fixed in this pass
1. **BUG-F1 (High): allowlist bypassed entire detection/risk/policy chain**
   - Location: `web_app\app.py:_on_detection_event`.
   - Impact: allowlisted sources produced no risk/policy telemetry (observability and scoring blind spot).
   - Fix: keep full detection/risk flow; bypass only enforcement in policy-decision handler.

2. **BUG-F2 (High): `/api/predict` ran legacy direct prevention path in parallel with event-bus path**
   - Location: `web_app\app.py:api_predict`.
   - Impact: potential double-action persistence and divergent behavior.
   - Fix: removed direct `prevention_service.evaluate` path; response now derives recent action from `ops_store` for same source and request window.

3. **BUG-F3 (Medium): engine playground ignored explicit `enabled` input and lacked 404 handling**
   - Location: `web_app\app.py:api_module_engine_playground`.
   - Impact: API behavior mismatch, silent wrong toggles, ambiguous failures.
   - Fix: respect `enabled` when provided, parse string booleans, return `404 engine_not_found` when missing.

4. **BUG-F4 (Low): allowlist enforcement bypass audit path could throw and break handler**
   - Location: `web_app\app.py:_on_policy_decision_event`.
   - Fix: wrapped bypass-audit write with explicit exception logging.

### Existing risks not fully fixed (tracked)
1. **RISK-R1 (Medium): backpressure sampling logic is implemented but not applied to ingestion decisions**
   - `BackpressureController.should_process()` is currently unused.

2. **RISK-R2 (Medium): `IngestionService.redis_stream_queue` path is not used in app wiring**
   - App swaps `queue` to Redis-backed queue instead of passing `redis_stream_queue`, leaving fallback logic dormant.

3. **RISK-R3 (Low/Medium): module/demo endpoints include simulated/static values not tied to runtime engines**
   - Example: policy tuning simulator and synthetic multi-engine-voting payloads.

4. **RISK-R4 (Low): several utility/training modules appear operationally dormant in serving runtime**
   - e.g., `src\imbalance_handler.py` not imported in runtime path.

## Phase 4: Dead Code / Partial Wiring Inventory

### Safe to remove (candidate)
- None removed in this pass (conservative due unknown external usage).

### Needs wiring
- `src\pipeline\backpressure.py:should_process` (currently unused in request ingestion).
- `IngestionService.redis_stream_queue` branch (not exercised by current app wiring).

### Intentionally dormant / tooling-only
- `src\imbalance_handler.py` (training utility style, no runtime imports found).
- CLI/demo scripts in `src\` and `tools\` (expected operational tooling, not app runtime).

### Partially implemented feature surfaces
- `/api/modules/policy-tuning` returns simulation payload rather than applying policy.
- `/api/modules/multi-engine-voting` uses synthetic engine labels/attributes separate from runtime engine registry.

## Phase 5: Safe Fixes Applied

### Files changed
- `web_app\app.py`
  - Allowlist behavior corrected to enforcement-only bypass.
  - Added allowlist bypass audit event (`allowlist_enforcement_bypass`) in policy-decision handler.
  - Removed direct legacy prevention execution from `/api/predict`.
  - Wired `prevention_action` response from persisted actions for same source + request time.
  - Hardened module engine playground toggle behavior.

- `tests\test_api_detection.py`
  - Added regression test: allowlisted source still predicts attack but no prevention action is executed/persisted.

## Phase 6: Testing Integrity

### Status
Automated test execution was **not possible in this runtime** because required `pwsh` is unavailable in tool environment.

### Required verification command (to run locally)
`python -m pytest -q`

### Minimum focused suites to validate these fixes
- `tests\test_api_detection.py`
- `tests\test_prevention_runtime.py`
- `tests\test_pipeline_runtime.py`
- `tests\test_phase_g_integration.py`

## Enforcement Safety Assessment
Improved from prior state: allowlist now no longer suppresses risk/policy/audit generation; enforcement bypass is explicit and auditable.

Remaining caution: module/demo endpoints still permit behavior that is not strict production control-plane semantics.

## ML Reliability Assessment
Core ML inference path is stable and guarded with schema defaults.

Remaining caution: multiple runtime paths (legacy direct predict flow vs multi-engine streaming flow) still coexist and require strict consistency testing.

## Runtime Stability Assessment
Core service is generally stable with cleanup/shutdown/leader-aware scheduler patterns.

Remaining caution: some runtime branches are present but underused (backpressure sampling, alternate ingestion queue modes), increasing drift risk.

## Final Status
- Phase 1: Complete
- Phase 2: Complete
- Phase 3: Complete
- Phase 4: Complete
- Phase 5: Complete (safe incremental fixes applied)
- Phase 6: Pending local test execution due environment limitation
- Phase 7: Complete (this report updated)
