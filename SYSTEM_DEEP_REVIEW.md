# SYSTEM_DEEP_REVIEW

## Scope
Principal-level deep review across architecture, runtime path integrity, bug detection, dead code inventory, safe fix implementation, and regression validation.

## Runtime Architecture (Validated)
- Ingress paths:
  - HTTP synchronous path: `/api/predict` and `/api/ingest/process`
  - Redis streaming path: `StreamProcessor` + `PipelineWorker` + callback publishing `DetectionEvent`
- Core event flow:
  - `DetectionEvent -> RiskScoreEvent -> PolicyDecisionEvent -> ActionEvent`
- Prevention and persistence:
  - Policy and risk decisions audited in `ops_store.audits`
  - Actions persisted in `ops_store.actions`
  - Alerts persisted in `ops_store.alerts`
- Enforcement stack:
  - `RiskEngine` (weights-aware scoring)
  - `PolicyEngine` (ALLOW/ALERT/RATE_LIMIT/TEMP_BLOCK/BLOCK/PENDING_BLOCK)
  - `ActionExecutor` (idempotent active-block checks, approval flow, cleanup)

## Confirmed Issues Found
1. BUG-001 (Critical): actions cleanup endpoint bypassed full action lifecycle cleanup.
2. BUG-002 (High): policy rollback omitted approval gate and risk-weight fields.
3. BUG-003 (Medium): dead/unused import in detection event path.
4. BUG-004 (Medium): escalation endpoints accessed internal tracker state directly.
5. BUG-005 (High): ingest processing used legacy direct prevention path, bypassing event-driven controls.
6. BUG-006 (High): stream/event-bus detections were not persisted as alerts.
7. BUG-007 (Low): stream lag logic carried unused decoded variable.
8. BUG-008 (Low): anomaly model publication in `fit()` was not lock-protected.

## Fixes Implemented
### web_app/app.py
- Fixed BUG-006:
  - `_on_detection_event` now persists `attack/suspicious` detections into `ops_store.alerts`.
  - Added `alerts_total` metric increment in event-driven path.
- Fixed BUG-003:
  - Removed dead inline import in detection handler.
- Fixed BUG-005:
  - `api_ingest_process` no longer calls legacy `prevention_service.evaluate()`.
  - Removed duplicate direct alert/action persistence from this handler; relies on event pipeline.
- Fixed BUG-001:
  - `api_actions_cleanup` now uses `action_executor.cleanup_expired_actions()` (includes unblock + state transition + deletion).
- Fixed BUG-002:
  - `api_policy_rollback` now restores:
    - `block_requires_approval`
    - `risk_weight_confidence`
    - `risk_weight_severity`
    - `risk_weight_frequency`
- Fixed BUG-004:
  - Escalation endpoints now compute counts via `summary()` and avoid direct `_states` access.
- Restored/added endpoint surface required by current platform behavior:
  - `PATCH /api/alerts/<id>`
  - FP suppression endpoints (`GET/POST/DELETE`)
  - pending action list and approve endpoints
  - observability endpoints (`/api/detections/history`, `/api/anomaly/status`, `/api/escalation/summary`, `/api/escalation/evict`, `/api/fp-stats`)
- Strengthened policy update path:
  - `/api/policy` now accepts approval gate + risk weight fields.
- Alert querying:
  - `/api/alerts` now supports `status` filter.

### src/pipeline/stream_processor.py
- Fixed BUG-007:
  - Removed unused `last-delivered-id` decode variable from lag calculation path.
  - Retained explicit rough estimate behavior.

### src/detection/engines/anomaly_engine.py
- Fixed BUG-008:
  - `fit()` now builds and trains model in local variable.
  - Publishes `self._model` and `self._fitted` under `_buffer_lock` to avoid split visibility.

## Dead Code / Partial Wiring Notes
- `EscalationLevel.ALERT` remains defined but not naturally reached by current event-to-escalation progression.
- Legacy direct-prevention path still exists in `PreventionService.evaluate` for compatibility, but ingest processing now avoids it.
- `src/integrations/` package remains minimal/placeholder.

## Validation
- Targeted regression tests:
  - `tests/test_new_api_endpoints.py`
  - `tests/test_api_detection.py`
  - `tests/test_anomaly_and_escalation.py`
  - Result: `34 passed`
- Full suite:
  - Result: `318 passed`

## Final Status
- Phase 1: Complete
- Phase 2: Complete
- Phase 3: Complete
- Phase 4: Complete
- Phase 5: Complete (fixes applied)
- Phase 6: Complete (tests passed)
- Phase 7: Complete (`SYSTEM_DEEP_REVIEW.md` created)
