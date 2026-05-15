INIDS STABILIZATION ENGINE — FULL EXECUTION

PHASE 1 — GLOBAL ARCHITECTURE RECONSTRUCTION
1.1 Subsystem Inventory
Runtime-Active Subsystems (participate in live request processing):
IDSubsystemRuntime RoleState OwnershipS1Flask HTTP LayerRequest ingress, routing, middleware enforcementStateless per-requestS2SecurityMiddlewareStackAuth, rate-limit, CSRF, IP-block, auditPer-request + shared rate countersS3DetectionServiceML inference, Alert generation, EventBus publishStateless inference; AlertStore writeS4EventBusSynchronous event dispatch, handler registryShared mutable handler dictS5RiskEngineConfidence/severity/frequency compositeIn-memory IP event deques (mutable, bounded)S6PolicyEngineThreshold decision treePure function — reads policy objectS7ActionExecutorAdapter dispatch, DB persist, EventBus publishWrites to OpsStore; firewall adapter callsS8FirewallAdaptersSystem-level firewall enforcementExternal system stateS9OpsStoreSQLite/PG persistenceAuthoritative durable stateS10AlertStoreIn-memory alert ring buffer (maxlen=1000)In-memory; ephemeralS11EngineRegistryMulti-engine evaluate_all dispatchRegistry dict + enabled mapS12EngineAggregatorResult fusionPure functionS13RealTimeStreamerEventBus → SocketIO WebSocketStateless emitterS14SocketIO/WebSocketClient push channelConnection mapS15PreventionScheduler30s periodic cleanup, adapter unblockDaemon threadS16IngestionQueueIn-memory bounded queue (maxlen=10000)In-memory; ephemeral
Initialized-But-Disconnected Subsystems (present in process, not in execution paths):
IDSubsystemDisconnect TypeD1EscalationTrackerInstantiated, never calledD2ThreatIntelManagerInstantiated, feeds never loadedD3TIEngineRegistered in EngineRegistry but is_ready()=False alwaysD4FalsePositiveManagerLoaded, only wired to SignatureEngineD5SiemExporterInstantiated, not subscribed to EventBusD6StreamProcessorExists in src/pipeline/, never imported by app.pyD7PolicyStore/versionsPersists, but runtime policy object never reloaded from itD8LeaderElectionInstantiated with redis_client=None; single-node mode
1.2 Runtime Execution Topology
HTTP INGRESS
     │
     ▼
[S2: SecurityMiddlewareStack]
  RateLimitMiddleware → IPBlockingMiddleware → SecurityHeadersMiddleware
  → AuditLogMiddleware → CSRFProtection → CorrelationTracer → JWTAuthManager
     │
     ▼
[S1: Flask Route Handler]  — POST /api/predict
     │
     ▼
[S3: DetectionService.predict_from_features()]
  model.predict_proba(df)
  alert_store.add(Alert)        ← S10 write
  ops_store.save_alert(Alert)   ← S9 write
  EventBus.publish(DetectionEvent) ──────────────────────────────────┐
     │                                                               │
     ▼                                                               │ SYNC
  return PredictionResult ← (BLOCKED until all EventBus handlers    │
                              complete — see S4 below)              │
                                                                     │
──────────────────────────────────────────────────────────────────── │
S4: EventBus.publish(DetectionEvent)                               ◄─┘
  Acquires _lock, copies handler list, releases _lock
  Invokes handlers SERIALLY in calling thread:
  │
  ├─ Handler: _on_detection_event()
  │    S5: RiskEngine.calculate(event)
  │      reads/writes self._events_by_source[source_ip]  ← LOCK required
  │      returns risk_score float
  │    S9: ops_store.add_audit("risk_score")             ← DB write
  │    S4: EventBus.publish(RiskScoreEvent)              ← REENTRANT into S4
  │         Acquires RLock again (permitted)
  │         Invokes _on_risk_event() serially:
  │         │
  │         ├─ S6: PolicyEngine.decide(risk_event, policy_obj)
  │         │    pure function on policy_obj
  │         │    policy_obj = prevention_service.policy  ← RUNTIME OBJECT
  │         │    returns PolicyDecisionEvent
  │         │
  │         ├─ S9: ops_store.add_audit("policy_decision")
  │         │
  │         └─ S4: EventBus.publish(PolicyDecisionEvent) ← REENTRANT again
  │                  Invokes _on_policy_decision_event():
  │                  │
  │                  ├─ if decision in {BLOCK, TEMP_BLOCK, RATE_LIMIT}:
  │                  │    S7: ActionExecutor.execute()
  │                  │      S8: adapter.block(ip, ttl)  ← ⚠️ NO TIMEOUT
  │                  │      S9: ops_store.save_action()
  │                  │      S4: EventBus.publish(ActionEvent) ← 4th reentrance
  │                  │           _on_action_realtime()
  │                  │           S13: socketio.emit(ActionEvent)
  │                  │
  │                  └─ _on_risk_realtime()
  │                       S13: socketio.emit(RiskScoreEvent)
  │
  └─ Handler: _on_detection_realtime()
       S13: socketio.emit(DetectionEvent)
1.3 Execution Threading Model
Thread: Flask Worker (WSGI thread from ThreadPoolExecutor max_workers=4)
  ├─ Handles entire /api/predict call
  ├─ Enters EventBus.publish() synchronously
  ├─ Executes 4-level deep reentrant EventBus dispatch
  ├─ Calls adapter.block() — blocks on subprocess/socket
  ├─ DOES NOT RELEASE until entire chain completes
  └─ IF adapter hangs → thread pinned indefinitely

Thread: PreventionScheduler (daemon)
  ├─ Wakes every 30s
  ├─ Calls ActionExecutor.cleanup_expired_actions()
  ├─ Also calls adapter.unblock() — NO TIMEOUT
  └─ Independent of Flask worker threads

Thread: RealTimeStreamer (daemon)
  ├─ Already driven by EventBus (same worker thread as /api/predict)
  └─ No independent thread — emission is synchronous in EventBus chain

Thread: PerceptionIntegration (worker threads)
  ├─ Independent daemon pool
  └─ Accesses AlertStore / OpsStore
1.4 State Ownership Map
StateOwnerAccess PatternConcurrency Riskalert_store._alertsS10 AlertStoreR/W, multiple Flask threadsdeque with maxlen; append is GIL-safe but not fully atomic under contentionops_store._dbS9 OpsStoreR/W, multiple threadsSQLite WAL mode or connection-per-thread; needs verificationevent_bus._handlersS4 EventBusR/W with RLockSafe — RLock + copy-before-dispatchrisk_engine._events_by_sourceS5 RiskEngineR/W with LockSafe — Lock held during mutationprevention_service.policyS6 PolicyEngine reads thisWritten only at startup; never updated at runtimeStale after POST /api/policyallowlist._entriesAllowlistR/W with Lock, no persistenceSafe in-memory; LOST on restartengine_registry._enginesS11 EngineRegistryR/W with LockSafeingestion_queueS16In-memory queueEphemeral; lost on crash
1.5 Startup/Shutdown Ordering
Critical startup dependencies:
1. OpsStore → must init before: AlertFilterEngine, Allowlist, FalsePositiveManager, 
              ActionExecutor, PolicyStore
2. ML Models (.pkl) → must load before: DetectionService, MLEngine registration
3. EngineRegistry populated → must complete before: /api/detect usable
4. EventBus subscriptions wired → must complete before: /api/predict usable
5. PreventionScheduler.start() → must start before: action expiration works
6. RealTimeStreamer.start() → must start before: WebSocket updates work
7. socketio.run() → LAST — accepts connections
No shutdown ordering defined. PreventionScheduler daemon thread terminates on process exit — any in-flight adapter.unblock() calls are abandoned.
1.6 Hot-Path Execution Chain
/api/predict CRITICAL PATH (blocks HTTP response):
  S2 middleware (~2ms)
  → S3 ML inference (~3ms)
  → S4 EventBus chain:
      → S5 RiskEngine (~3ms)
      → S6 PolicyEngine (~1ms)
      → S7 ActionExecutor + S8 adapter (~1ms-200ms depending on adapter)
      → S9 3x OpsStore writes (~5-15ms)
      → S13 3x socketio.emit (~3ms)
  TOTAL: 18ms (mock adapter) → 200ms+ (UFW) → ∞ (hung adapter)
1.7 Failure Propagation Paths
FAILURE: S8 adapter.block() hangs
  → S7 ActionExecutor thread pinned
  → S4 EventBus handler in-flight
  → S3 DetectionService blocked waiting for EventBus return
  → S1 Flask worker thread pinned
  → After 4 concurrent requests: ALL 4 workers pinned
  → HTTP 502/504 from gateway
  → Detection effectively stopped

FAILURE: S9 OpsStore DB locked/corrupted
  → ops_store.save_alert() throws → alert lost
  → ops_store.save_action() throws → action not persisted but firewall was called
  → State desync: firewall blocked IP; DB has no record; scheduler can't unblock

FAILURE: S4 EventBus handler exception
  → Exception caught, logged, next handler continues
  → If _on_detection_event throws: RiskEngine never runs
  → If _on_risk_event throws: PolicyEngine never runs, no action
  → Silent partial failure — HTTP response returns as if normal

FAILURE: Process crash
  → S10 AlertStore: all in-memory alerts lost (DB persisted ones survive)
  → Allowlist._entries: LOST (not persisted to S9)
  → S16 IngestionQueue: all queued items LOST
  → S5 RiskEngine frequency state: LOST
  → D1 EscalationTracker state: LOST

PHASE 2 — FINDING CORRELATION + VALIDATION
2.1 Finding Classification Matrix
Finding IDSourceDescriptionClassificationRoot vs SymptomF-01R1+R2Allowlist not persisted to OpsStoreVERIFIEDROOT CAUSE — missing OpsStore method implsF-02R1+R2TI feeds never loaded at startupVERIFIEDROOT CAUSE — explicit TODO comment; no loaderF-03R1+R2EscalationTracker instantiated but never calledVERIFIEDROOT CAUSE — missing call site in EventBus handlerF-04R1+R2FalsePositiveManager only wired to SignatureEngineVERIFIEDROOT CAUSE — other engines never receive fp_manager refF-05R1+R2Policy rollback updates DB but not runtime policy objectVERIFIEDROOT CAUSE — prevention_service.policy never reloadedF-06R1+R2SiemExporter not subscribed to EventBusVERIFIEDROOT CAUSE — subscription call absent from app.py wiringF-07R1+R2StreamProcessor not integrated with web_appVERIFIEDARCHITECTURAL GAP — separate module, never importedF-08R1+R2LeaderElection has no Redis, no failoverVERIFIEDARCHITECTURAL INCOMPLETENESS — single-node onlyF-09R1+R2EventBus dispatch synchronous — adapter latency blocks HTTPVERIFIEDROOT CAUSE — design choice with no timeout isolationF-10R1+R2No timeout/circuit-breaker on adapter.block() callsVERIFIEDROOT CAUSE — ActionExecutor calls adapter nakedF-11R1Frontend GlobalState singleton race conditionsHIGH-CONFIDENCE INFERENCESYMPTOM of concurrent WebSocket message processingF-12R2RiskEngine memory eviction under botnet load degrades frequency scoresHIGH-CONFIDENCE INFERENCEROOT CAUSE is bounded dict; symptom is accuracy dropF-13R1/api/predict and /api/detect are divergent paths — only predict triggers EventBus chainVERIFIEDARCHITECTURAL — intentional but creates operational gapF-14R1+R2ops_store.save_alert() and adapter.block() can desync (action executed but DB write fails)HIGH-CONFIDENCE INFERENCECASCADING FAILURE RESULT of no transactional boundaryF-15R2PreventionScheduler calls adapter.unblock() with no timeoutVERIFIEDROOT CAUSE — same issue as F-10, different call siteF-16R1AlertStore maxlen=1000 — in-memory only; lost on crashVERIFIEDARCHITECTURAL DECISION with documented riskF-17R1No request timeout on /api/predict end-to-endHIGH-CONFIDENCE INFERENCESYMPTOM of F-09/F-10 combinedF-18R2IngestionQueue maxlen=10000 — in-memory, lost on crashVERIFIEDARCHITECTURAL GAP — no durable queueF-19R1ThreadPoolExecutor max_workers=4 — under attack load, all 4 pinned by blocked adapter callsHIGH-CONFIDENCE INFERENCECASCADING FAILURE RESULT of F-10
2.2 Finding Correlation: Shared Root Causes
Root Cause Cluster A — Missing Persistence Implementation:
F-01 → Allowlist OpsStore methods never implemented
→ Directly enables: restart-induced allowlist wipe → blocked trusted IPs re-blocked
Root Cause Cluster B — Incomplete Startup Wiring:
F-02 (TI feeds) + F-03 (Escalation) + F-06 (SIEM) + F-05 (Policy reload)
→ All are app.py initialization gaps — subsystems created but integration calls absent
Root Cause Cluster C — Unprotected Synchronous Adapter Calls:
F-09 + F-10 + F-15 + F-17 + F-19
→ All cascade from: adapter.block() in calling thread, no timeout, no circuit-breaker
→ This is the single most dangerous operational failure mode
Root Cause Cluster D — Partial Engine Integration:
F-04 (FP Manager) + F-03 (Escalation) + F-13 (predict vs detect divergence)
→ Features implemented but never connected to detection decision path
2.3 Conflicting Findings
None detected. Both reports are consistent and complementary — Report 1 provides architectural overview; Report 2 provides subsystem depth. No contradictions.
2.4 Findings Rejected / Reclassified
FindingOriginal AssessmentReclassificationReason"Global state race conditions" (F-11)R1: Medium severityLOW-CONFIDENCE production impactJavaScript is single-threaded; race requires specific WebSocket message ordering that's unlikely in practice but real under high concurrency"No JS bundling"MediumP3 NON-CRITICALNo runtime correctness impact; pure performance"Circular dependencies — none detected"✓ GoodCONFIRMEDEventBus handler chain is reentrant but not circular

PHASE 3 — SYSTEM SURVIVAL PRIORITIZATION
P0 — SYSTEM SURVIVAL
P0-A: Adapter Blocking → Thread Pool Exhaustion → System Death

Finding: F-09, F-10, F-15, F-19
Mechanism: adapter.block() hangs in Flask worker thread → 4 concurrent blocked requests exhaust ThreadPoolExecutor → system stops processing detections
Production Risk: Under active attack (exactly when adapter load is highest), detection system becomes inoperable
Classification: CASCADING CRASH PATTERN

P0-B: OpsStore DB Write Failure After Adapter Execution

Finding: F-14
Mechanism: adapter.block() succeeds (IP is firewalled) → ops_store.save_action() throws → action not recorded → PreventionScheduler never unblocks IP → permanent phantom block
Classification: STATE CORRUPTION (firewall vs DB desync)

P1 — OPERATIONAL STABILITY
P1-A: Allowlist Data Loss on Restart

Finding: F-01
Mechanism: allowlist._entries not persisted → restart empties allowlist → previously approved IPs re-blocked
Classification: PERSISTENCE DEFECT

P1-B: Policy Runtime Disconnect

Finding: F-05
Mechanism: POST /api/policy updates DB + returns 200 OK → runtime policy_obj unchanged → operator believes they changed thresholds; system still uses old ones
Classification: RUNTIME ORCHESTRATION DEFECT

P1-C: EscalationTracker Never Called

Finding: F-03
Mechanism: Repeat attacker hits system 10x → each time gets only ALERT → no auto-escalation → operator must manually intervene
Classification: INTEGRATION DEFECT

P2 — SCALABILITY + RESILIENCE
P2-A: TI Engine Permanently Disabled

Finding: F-02
Mechanism: is_ready()=False → TI engine skipped in every evaluate_all() → known-bad IPs not flagged by TI data
Classification: INTEGRATION DEFECT

P2-B: FalsePositiveManager Partial Coverage

Finding: F-04
Mechanism: Suppressions only checked by SignatureEngine → ML/Threshold engines fire on suppressed patterns → FP suppression partially effective
Classification: INTEGRATION DEFECT

P2-C: SiemExporter Manual Pull

Finding: F-06
Mechanism: SIEM export is manual endpoint → events not pushed in real-time → SIEM data always stale
Classification: INTEGRATION DEFECT

P2-D: RiskEngine Memory Eviction Under DDoS

Finding: F-12
Mechanism: >10,000 unique source IPs → LRU eviction → frequency scores reset for evicted IPs → risk scores artificially low → escalation thresholds not reached
Classification: SCALABILITY DEFECT

P3 — CLEANUP / NON-CRITICAL

P3-A: StreamProcessor not integrated (standalone, no active path)
P3-B: Frontend GlobalState singleton (low practical impact in JS single-thread)
P3-C: No JS bundling
P3-D: HA/LeaderElection incomplete (single-node acceptable for current deployment)


PHASE 4 — SURGICAL FIX DESIGN
FIX-01: Adapter Timeout + Circuit Breaker (P0-A)
Exact problem origin: src/ips/action_executor.py — ActionExecutor.execute() calls self.adapter.block(target, ttl) with no timeout. For UFW/Nftables, this wraps subprocess.run(). For Webhook, requests.post() with its own timeout but exception not caught upstream.
Exact runtime failure mechanism: Flask worker thread enters subprocess.run() in UfwFirewallAdapter.block(). If UFW service is hung or system is under load, subprocess.run() blocks indefinitely (no timeout= passed). Thread pinned. With max_workers=4, 4 concurrent attack detections exhaust the pool. Subsequent HTTP requests queue behind blocked workers until client timeout (30s+). System stops detecting.
Exact subsystem interaction: S1 Flask Worker → S7 ActionExecutor → S8 FirewallAdapter → subprocess → OS. No return path until OS responds.
Exact root cause: block_ip() in ActionExecutor has try/except but no timeout enforcement. subprocess.run() in UfwFirewallAdapter.block() has no timeout= parameter.
Exact operational impact: System processes 0 detections under attack when adapter is degraded. Detection stops exactly when it matters most.
Exact production risk: Complete service unavailability. 4-worker thread pool exhaustion in under 4 seconds under sustained attack.
Exact fix strategy:
Step 1 — Add timeout to subprocess calls in firewall_adapters.py:
python# UfwFirewallAdapter.block() and .unblock() and .rate_limit()
result = subprocess.run(
    args,
    capture_output=True,
    timeout=5,          # ADD THIS — was missing
    check=False
)
# NftablesFirewallAdapter — same change, same locations
Step 2 — Wrap ActionExecutor.block_ip() with a concurrent.futures timeout:
python# src/ips/action_executor.py

import concurrent.futures as cf

ADAPTER_CALL_TIMEOUT_S = 3.0  # Configurable via Settings

def _call_adapter_with_timeout(self, fn, *args) -> tuple[bool, str]:
    """
    Execute adapter call in isolated thread with hard timeout.
    Returns (success, status_string).
    Never raises — failure is returned as status.
    """
    with cf.ThreadPoolExecutor(max_workers=1) as _exec:
        future = _exec.submit(fn, *args)
        try:
            return future.result(timeout=ADAPTER_CALL_TIMEOUT_S)
        except cf.TimeoutError:
            self.logger.error(
                "adapter_timeout target=%s timeout_s=%s",
                args[0] if args else "unknown",
                ADAPTER_CALL_TIMEOUT_S
            )
            return False, "adapter_timeout"
        except Exception as exc:
            self.logger.exception("adapter_call_failed target=%s", args[0] if args else "unknown")
            return False, f"adapter_exception:{type(exc).__name__}"

def block_ip(self, ip: str, ttl: int) -> tuple[bool, str]:
    """Replaced — now timeout-protected"""
    ok, status = self._call_adapter_with_timeout(
        self.adapter.block, ip, ttl
    )
    return bool(ok), "blocked" if ok else status
Step 3 — Add circuit breaker state to ActionExecutor.__init__():
python# src/ips/action_executor.py __init__
self._cb_failure_count: int = 0
self._cb_open_until: float = 0.0
self._cb_failure_threshold: int = 3
self._cb_open_duration_s: float = 60.0
self._cb_lock = Lock()
Step 4 — Add circuit breaker check at entry of execute():
pythondef _circuit_open(self) -> bool:
    with self._cb_lock:
        if self._cb_open_until == 0.0:
            return False
        if time.time() > self._cb_open_until:
            self._cb_open_until = 0.0
            self._cb_failure_count = 0
            self.logger.info("circuit_breaker_closed adapter=%s", type(self.adapter).__name__)
            return False
        return True

def _record_adapter_result(self, success: bool) -> None:
    with self._cb_lock:
        if success:
            self._cb_failure_count = 0
        else:
            self._cb_failure_count += 1
            if self._cb_failure_count >= self._cb_failure_threshold:
                self._cb_open_until = time.time() + self._cb_open_duration_s
                self.logger.error(
                    "circuit_breaker_open adapter=%s for %ss",
                    type(self.adapter).__name__, self._cb_open_duration_s
                )

# In execute(), before calling block_ip():
if self._circuit_open():
    action_event = ActionEvent(..., status="CIRCUIT_OPEN", executed=False)
    self.ops_store.save_action(action_event.to_dict())
    return action_event
Exact runtime behavior change: adapter calls now have hard 3s timeout. After 3 consecutive failures/timeouts, adapter calls fast-fail for 60s returning "CIRCUIT_OPEN" status. HTTP request processing continues. Flask workers never pinned.
Exact integration points: src/ips/action_executor.py (block_ip, execute), src/firewall_adapters.py (subprocess timeout). src/ips/scheduler.py PreventionScheduler also calls adapter — same wrapper must be applied to cleanup path.
Exact dependency implications: ADAPTER_CALL_TIMEOUT_S must be added to src/settings.py. Circuit breaker state is per-process; not shared across nodes.
Exact regression risks: A 3s timeout on subprocess is generous for UFW (typical ~100ms). Risk of false timeout if system is under extreme CPU load. Mitigated by making timeout configurable.
Exact rollback strategy: Remove _call_adapter_with_timeout wrapper from block_ip(). Revert to direct self.adapter.block(target, ttl). No DB schema changes. No deployment dependencies.

FIX-02: Allowlist Persistence (P1-A)
Exact problem origin: src/ops_store.py — three methods absent: list_allowlist(), add_allowlist_entry(), remove_allowlist_entry(). src/prevention/allowlist.py — add() and remove() methods call self._persist_add() and self._persist_remove() which call the missing ops_store methods.
Exact runtime failure mechanism: Allowlist._load() is called at init time; calls self._ops_store.list_allowlist() which doesn't exist or returns []. All subsequent allowlist.add() calls succeed in-memory. On process restart, _load() returns empty set. Allowlist is blank.
Exact root cause: OpsStore schema already has allowlist table (confirmed in schema). Methods were never implemented in the OpsStore class.
Exact operational impact: Any allowlist entry added after startup is lost on any process restart (crash, deploy, OOM kill). Trusted IPs get re-blocked post-restart. Incident response recovery is broken.
Exact production risk: Blocked trusted services resume being blocked after every restart. Compliance gap if allowlist is part of security policy.
Exact fix strategy:
Add to src/ops_store.py:
pythondef list_allowlist(self) -> list[dict]:
    """
    Return all allowlist entries from persistent store.
    Returns: [{"entry": "192.168.1.1", "reason": "...", "added_by": "...", "added_at": "..."}, ...]
    """
    query = """
        SELECT entry, reason, added_by, added_at
        FROM allowlist
        ORDER BY added_at DESC
    """
    return self._fetchall(query)

def add_allowlist_entry(
    self,
    entry: str,
    reason: str = "",
    added_by: str = "system"
) -> None:
    """
    Insert allowlist entry. Silently ignores duplicate (UNIQUE constraint).
    """
    query = """
        INSERT OR IGNORE INTO allowlist (id, entry, reason, added_by, added_at)
        VALUES (?, ?, ?, ?, ?)
    """
    self._execute(query, (
        str(uuid4()),
        entry,
        reason,
        added_by,
        datetime.now(timezone.utc).isoformat()
    ))

def remove_allowlist_entry(self, entry: str) -> None:
    """Delete allowlist entry by normalized IP/CIDR string."""
    query = "DELETE FROM allowlist WHERE entry = ?"
    self._execute(query, (entry,))
Modify src/prevention/allowlist.py — _persist_add and _persist_remove already call these methods. No change to allowlist.py required IF the ops_store method signatures match. Verify _persist_add passes added_by from caller context. If not, add parameter threading from add(entry, reason, added_by="system").
Modify Allowlist._load() to explicitly iterate loaded entries:
pythondef _load(self) -> None:
    if self._ops_store is None:
        return
    try:
        rows = self._ops_store.list_allowlist()
        for row in rows:
            normalized = self._normalize(row["entry"])
            if normalized:
                self._entries.add(normalized)
        self._rebuild_networks()
        self.logger.info("Allowlist loaded %d entries from store", len(self._entries))
    except Exception:
        self.logger.exception("Allowlist failed to load from store — starting empty")
Exact runtime behavior change: All allowlist.add() calls now atomically write to SQLite. All restarts hydrate from DB. The in-memory set remains the authoritative read path (fast). DB is write-ahead and read at startup only.
Exact regression risks: INSERT OR IGNORE semantics: if a duplicate entry is added, no error, no duplicate row. Correct. PostgreSQL equivalent is INSERT ... ON CONFLICT (entry) DO NOTHING. If using PG, verify OpsStore _execute uses parameterized query. Low risk.
Exact rollback: Remove the three OpsStore methods. Allowlist reverts to in-memory only. No data loss — DB table retains data but is not read/written.

FIX-03: Policy Runtime Reload (P1-B)
Exact problem origin: web_app/app.py — prevention_service.policy is set at initialization. POST /api/policy calls policy_store.update() which persists to DB but does not reassign prevention_service.policy. PolicyEngine.decide() receives policy as parameter from _on_risk_event(), which reads prevention_service.policy — the stale runtime object.
Exact runtime failure mechanism: Operator calls POST /api/policy with new thresholds. API returns 200. DB updated. prevention_service.policy object unchanged. All subsequent PolicyEngine.decide() calls use old thresholds. Operator believes change is live; it is not.
Exact root cause: No callback/hook in the POST /api/policy route handler to reload the runtime policy from the updated store.
Exact fix strategy:
In web_app/app.py, locate the POST /api/policy route handler. After the call to policy_store.update(new_config), add:
python# After: policy_store.update(new_config, changed_by=..., description=...)
# Add immediately after:
reloaded = policy_store.load_current()
if reloaded is not None:
    prevention_service.policy = reloaded
    logger.info("policy_runtime_reloaded version=%s", getattr(reloaded, 'version', 'unknown'))
else:
    logger.error("policy_reload_failed: store returned None after update")
In POST /api/policy/rollback route handler, after policy_store.rollback():
pythonreloaded = policy_store.load_current()
if reloaded is not None:
    prevention_service.policy = reloaded
    logger.info("policy_rolled_back_and_reloaded")
policy_store.load_current() must exist and return a policy object (not a dict). Verify its return type matches prevention_service.policy's expected type. If it returns a dict, convert via PolicyConfig(**reloaded_dict) or equivalent.
Exact regression risks: prevention_service.policy is read by _on_risk_event() which can be executing concurrently on another Flask worker thread. The assignment prevention_service.policy = reloaded is a Python attribute assignment — GIL-protected for the assignment itself, but the old policy object may be mid-use in another thread. This is acceptable: PolicyEngine.decide() takes the policy as parameter at call time; a concurrent call using the old object completes normally, subsequent calls use the new object. No corruption risk — the old object is immutable during decide().
Exact rollback: Remove the two reload blocks from the route handlers. Revert to original behavior (DB updated, runtime stale).

FIX-04: EscalationTracker Integration (P1-C)
Exact problem origin: web_app/app.py — escalation_tracker = EscalationTracker(cooldown_seconds=300.0) is instantiated. No call to escalation_tracker.record_hit() exists anywhere in the codebase. The integration point is _on_detection_event() in app.py.
Exact runtime failure mechanism: Every detection from the same attacker IP is independently evaluated with the same risk score. No escalation occurs. A repeat attacker generating low-confidence detections receives ALERT indefinitely and is never blocked.
Exact fix strategy:
In web_app/app.py, function _on_detection_event(event: DetectionEvent):
pythondef _on_detection_event(event: DetectionEvent) -> None:
    # Existing: Calculate risk
    risk_event = risk_engine.calculate(event)
    
    # ADD: Record hit in escalation tracker
    if event.suspicious and event.source_ip:
        escalation_level = escalation_tracker.record_hit(
            source_ip=event.source_ip,
            severity=event.severity
        )
        # Attach escalation level to risk event for PolicyEngine consumption
        # PolicyEngine must receive escalation context
        if escalation_level is not None:
            risk_event = _apply_escalation_to_risk(risk_event, escalation_level)
    
    # Existing: publish, audit
    event_bus.publish(risk_event)
    ops_store.add_audit("risk_score", ...)
Add helper in app.py:
pythondef _apply_escalation_to_risk(risk_event: RiskScoreEvent, level: int) -> RiskScoreEvent:
    """
    Adjust risk score based on escalation level.
    Level 0: no change
    Level 1 (repeat alert): +0.10 to risk_score
    Level 2 (rate_limit candidate): +0.20
    Level 3 (temp_block candidate): +0.35
    Level 4 (perm_block): +0.50
    All additions clamped to 1.0 max.
    """
    ESCALATION_BOOST = {0: 0.0, 1: 0.10, 2: 0.20, 3: 0.35, 4: 0.50}
    boost = ESCALATION_BOOST.get(level, 0.0)
    if boost == 0.0:
        return risk_event
    new_score = min(1.0, risk_event.risk_score + boost)
    return RiskScoreEvent(
        detection=risk_event.detection,
        risk_score=new_score,
        components={**risk_event.components, "escalation_boost": boost, "escalation_level": level},
        timestamp=risk_event.timestamp
    )
Exact regression risks: escalation_tracker.record_hit() writes to an internal dict and optionally to ops_store.escalations table. This adds ~1-2ms to the hot path. Acceptable. Risk: if record_hit() throws, wrap in try/except to prevent chain break.
Exact rollback: Remove the record_hit() call block and _apply_escalation_to_risk(). Restore original _on_detection_event().

FIX-05: TI Feed Loader at Startup (P2-A)
Exact problem origin: web_app/app.py line ~310 — explicit # TODO: Load threat intel feeds comment. ti_manager.load_feeds(SETTINGS.ti_feed_dir) never called.
Exact fix strategy:
In web_app/app.py, in load_models() function, after model loading completes:
python# After: engine_registry.register(ml_engine)
# Add:
_load_threat_intel_feeds()
Add function in app.py:
pythondef _load_threat_intel_feeds() -> None:
    """
    Load TI feeds from configured directory.
    Non-fatal: logs warning if feeds missing or malformed.
    TI engine stays disabled if feeds fail to load.
    """
    feed_dir = getattr(SETTINGS, 'ti_feed_dir', None)
    if not feed_dir:
        logger.info("ti_feeds_skipped: ti_feed_dir not configured")
        return
    if not os.path.isdir(feed_dir):
        logger.warning("ti_feeds_skipped: directory not found path=%s", feed_dir)
        return
    try:
        ti_manager.load_feeds(feed_dir)
        count = ti_manager.cache.size() if hasattr(ti_manager, 'cache') else '?'
        logger.info("ti_feeds_loaded count=%s from=%s", count, feed_dir)
    except Exception:
        logger.exception("ti_feeds_load_failed path=%s — TI engine will stay disabled", feed_dir)
Exact regression risks: If ti_feed_dir contains malformed files, load_feeds() may throw. The try/except above ensures TI failure is non-fatal. TI engine stays at is_ready()=False. Existing behavior preserved on failure.

FIX-06: FalsePositiveManager Full Engine Integration (P2-B)
Exact problem origin: MLEngine, ThresholdEngine, AnomalyEngine in src/detection/engines/ — these classes receive no reference to fp_manager. SignatureEngine already has integration.
Exact fix strategy:
Modify engine constructors in src/detection/engines/ml_engine.py, threshold_engine.py, anomaly_engine.py:
pythonclass MLEngine(DetectionEngine):
    def __init__(self, model, engine_id="ml_primary", fp_manager=None):
        self._fp_manager = fp_manager
        ...

    def evaluate(self, features: dict) -> EngineResult:
        result = self._detect(features)
        # ADD: Check FP suppression
        if self._fp_manager and result.verdict == "attack":
            rule_id = f"ml_{result.attack_type or 'generic'}"
            if self._fp_manager.is_suppressed(self.engine_id, rule_id):
                return EngineResult(
                    engine_id=self.engine_id,
                    verdict="normal",
                    confidence=0.0,
                    severity="low",
                    reason=f"fp_suppressed:{rule_id}"
                )
        return result
In web_app/app.py where engines are instantiated, pass fp_manager:
pythonml_engine = MLEngine(model, engine_id='ml_primary', fp_manager=fp_manager)
threshold_engine = ThresholdEngine(fp_manager=fp_manager)
anomaly_engine = AnomalyEngine(model_path, fp_manager=fp_manager)
Exact regression risks: If fp_manager.is_suppressed() has a different signature, adapt the call. The suppression only fires if the result is "attack" — safe to add without changing normal-traffic behavior.

FIX-07: SiemExporter EventBus Subscription (P2-C)
Exact problem origin: web_app/app.py EventBus wiring section — siem_exporter is created but event_bus.subscribe() is never called for it.
Exact fix strategy:
In web_app/app.py, in the EventBus subscription wiring block (after all other subscriptions), add:
python# Existing wiring: 6 subscriptions for detection→risk→policy→action chain
# ADD — SIEM export subscriptions:
if siem_exporter is not None:
    event_bus.subscribe(DetectionEvent, siem_exporter.on_detection)
    event_bus.subscribe(ActionEvent, siem_exporter.on_action)
Verify siem_exporter.on_detection and siem_exporter.on_action exist in SiemExporter. If they don't, add them:
python# src/observability/siem_exporter.py
def on_detection(self, event: DetectionEvent) -> None:
    """Called by EventBus on every detection — push to SIEM."""
    try:
        self._export_detection(event)
    except Exception:
        logger.exception("siem_export_detection_failed")
        # Do not re-raise — must not break EventBus chain

def on_action(self, event: ActionEvent) -> None:
    try:
        self._export_action(event)
    except Exception:
        logger.exception("siem_export_action_failed")
Exact regression risks: SIEM export is now on the hot path for every detection. If SIEM export is slow (HTTP call), this adds to EventBus chain latency. Ensure _export_detection() is either fast (<5ms) or is queued internally with a buffer. If SIEM endpoint is unavailable, the try/except ensures chain continues.

PHASE 5 — FILE-LEVEL IMPLEMENTATION PLANNING
IMPL-01: Adapter Timeout + Circuit Breaker
Affected subsystem: S7 ActionExecutor, S8 FirewallAdapters, S15 PreventionScheduler
Exact files:
FileModificationsrc/firewall_adapters.pyAdd timeout=5 to all subprocess.run() calls in UfwFirewallAdapter and NftablesFirewallAdaptersrc/ips/action_executor.pyAdd _call_adapter_with_timeout(), _circuit_open(), _record_adapter_result(), modify block_ip(), modify execute()src/ips/scheduler.pyWrap action_executor.adapter.unblock() call with same timeout patternsrc/settings.pyAdd adapter_call_timeout_s: float = 3.0 and adapter_cb_failure_threshold: int = 3
Exact modification targets:
src/firewall_adapters.py:

UfwFirewallAdapter.block(): subprocess.run([...]) → subprocess.run([...], timeout=5)
UfwFirewallAdapter.unblock(): same
NftablesFirewallAdapter.block(): same
NftablesFirewallAdapter.unblock(): same
Handle subprocess.TimeoutExpired in try/except, return False

src/ips/action_executor.py:

Add imports: import concurrent.futures, import time
Add __init__ params: adapter_timeout_s=3.0, cb_failure_threshold=3, cb_open_duration_s=60.0
Add 5 new methods (as specified in FIX-01)
Modify block_ip(): replace direct self.adapter.block() with self._call_adapter_with_timeout(self.adapter.block, target, ttl_seconds)
Modify execute(): add circuit breaker check at entry

Execution ordering: src/settings.py first, then src/firewall_adapters.py, then src/ips/action_executor.py, then src/ips/scheduler.py.
Validation requirements:

Unit test: Mock adapter that sleeps 10s → verify block_ip() returns (False, "adapter_timeout") within 3.5s
Unit test: Mock adapter that fails 3x → verify circuit opens → 4th call returns (False, "circuit_open") immediately
Unit test: Circuit auto-closes after cb_open_duration_s
Integration test: UfwAdapter with timeout=5 subprocess → verify subprocess.TimeoutExpired returns False

Rollback: Git revert of 4 files. No DB changes. No deployment dependencies.

IMPL-02: Allowlist Persistence
Exact files:
FileModificationsrc/ops_store.pyAdd 3 methods: list_allowlist(), add_allowlist_entry(), remove_allowlist_entry()src/prevention/allowlist.pyVerify _persist_add(), _persist_remove(), _load() call correct method signatures
Exact modification targets:
src/ops_store.py:

Locate the class body after existing CRUD methods
Insert the 3 methods from FIX-02 above
Use INSERT OR IGNORE for SQLite; parameterize for PG

src/prevention/allowlist.py:

_load(): verify it calls self._ops_store.list_allowlist() and iterates result
_persist_add(entry, reason): verify it calls self._ops_store.add_allowlist_entry(entry, reason)
_persist_remove(entry): verify it calls self._ops_store.remove_allowlist_entry(entry)

Validation requirements:

Integration test: allowlist.add("10.0.0.1", "test") → ops_store.list_allowlist() returns entry
Integration test: New Allowlist(ops_store) instance → contains("10.0.0.1") returns True
Integration test: allowlist.remove("10.0.0.1") → new instance → contains("10.0.0.1") returns False
API test: POST /api/allowlist → restart server → GET /api/allowlist returns entry


IMPL-03: Policy Runtime Reload
Exact files:
FileModificationweb_app/app.pyModify POST /api/policy handler and POST /api/policy/rollback handler
Exact modification targets:
web_app/app.py — POST /api/policy route:

After policy_store.update(...) call: add reload block (FIX-03)
Verify policy_store.load_current() return type

web_app/app.py — POST /api/policy/rollback route:

After policy_store.rollback(...) call: add reload block

Validation requirements:

Test: POST /api/policy with risk_block_threshold=0.90 → verify prevention_service.policy.risk_block_threshold == 0.90 in-process
Test: Submit detection → verify PolicyEngine uses new threshold
Test: POST /api/policy/rollback → verify runtime policy reverts to prior version


IMPL-04: EscalationTracker Integration
Exact files:
FileModificationweb_app/app.pyModify _on_detection_event() handler, add _apply_escalation_to_risk() helper
Exact modification targets:
web_app/app.py — _on_detection_event(event):

After risk_event = risk_engine.calculate(event):
Add escalation call block (FIX-04)
Wrap in try/except to prevent chain break

Validation requirements:

Test: Send 3 detections from same IP within 300s → verify 3rd detection risk_score elevated
Test: escalation_tracker.get_level(ip) returns 2 after 2 hits
Test: 5+ hits → verify BLOCK decision reached for IP that was previously only getting ALERT


IMPL-05: TI Feed Loader
Exact files:
FileModificationweb_app/app.pyAdd _load_threat_intel_feeds() function, call from load_models()
Validation requirements:

Test: Create SETTINGS.ti_feed_dir pointing to directory with a valid feed file → after load_models(), ti_engine.is_ready() == True
Test: Missing directory → load_models() completes without exception, ti_engine.is_ready() == False
Test: Malformed feed file → load_models() completes without exception


IMPL-06: FP Manager Engine Integration
Exact files:
FileModificationsrc/detection/engines/ml_engine.pyAdd fp_manager param to __init__, add suppression check in evaluate()src/detection/engines/threshold_engine.pySamesrc/detection/engines/anomaly_engine.pySameweb_app/app.pyPass fp_manager=fp_manager to engine constructors

IMPL-07: SiemExporter Subscription
Exact files:
FileModificationsrc/observability/siem_exporter.pyAdd on_detection(), on_action() methods if absentweb_app/app.pyAdd EventBus subscribe calls in wiring block

PHASE 6 — DEPENDENCY-AWARE EXECUTION ROADMAP
WAVE 1 — SYSTEM SURVIVAL (Days 1-2)
Objective: Eliminate thread-pool exhaustion path. Prevent system from going down during active attack.
Systems touched: S7 ActionExecutor, S8 FirewallAdapters, S15 PreventionScheduler, Settings
Prerequisites: None. This is the first change.
Implementation sequence:

Modify src/settings.py — add timeout/CB config fields
Modify src/firewall_adapters.py — add subprocess timeout
Modify src/ips/action_executor.py — add timeout wrapper + CB
Modify src/ips/scheduler.py — add timeout to unblock calls
Deploy as single commit

Runtime risks: ThreadPoolExecutor(max_workers=1) created per adapter call. Under 4 concurrent requests, this creates 4 additional threads beyond the worker threads. With max_workers=4 Flask workers, total additional threads = 4. Acceptable for standard thread limits.
Deployment strategy: Rolling restart. One worker at a time. Validate via health endpoint between restarts.
Rollback: Revert 4 files. Restart.
Validation:

Load test: Send 20 concurrent /api/predict requests while adapter sleeps 10s → verify no thread exhaustion, responses return within 5s
Verify circuit breaker opens after 3 failures
Verify circuit auto-closes after 60s

Post-deployment monitoring:

Watch: adapter_timeout count in logs
Watch: circuit_breaker_open log events
Watch: Flask worker thread count (should not grow unboundedly)


WAVE 2 — DATA INTEGRITY (Days 3-4)
Objective: Fix allowlist persistence. Fix policy runtime reload.
Systems touched: S9 OpsStore, Allowlist, S6 PolicyEngine (via prevention_service.policy)
Prerequisites: Wave 1 complete (prevents race where hung adapter holds thread during DB write)
Implementation sequence:

Modify src/ops_store.py — add 3 allowlist methods
Verify src/prevention/allowlist.py — add/verify _load(), _persist_add(), _persist_remove()
Modify web_app/app.py — policy reload in POST /api/policy and POST /api/policy/rollback
Deploy

Runtime risks: Low. DB writes are atomic per SQLite. Policy reload is single assignment.
Deployment strategy: Standard restart.
Rollback: Remove 3 OpsStore methods. Allowlist data in DB persists but is ignored. Remove reload blocks from route handlers.
Validation:

Add allowlist entry via API → restart → GET /api/allowlist returns entry
POST /api/policy with changed threshold → verify detection uses new threshold without restart
POST /api/policy/rollback → verify detection uses prior threshold


WAVE 3 — DETECTION CAPABILITY RESTORATION (Days 5-7)
Objective: Wire escalation tracker. Load TI feeds. Integrate FP manager to all engines.
Systems touched: D1 EscalationTracker, D2 ThreatIntelManager, D3 TIEngine, S11 EngineRegistry
Prerequisites: Wave 2 complete (OpsStore methods available — escalation tracker may write to DB)
Implementation sequence:

Modify web_app/app.py — _on_detection_event() + _apply_escalation_to_risk() + _load_threat_intel_feeds()
Modify src/detection/engines/ml_engine.py, threshold_engine.py, anomaly_engine.py — add fp_manager
Modify web_app/app.py — pass fp_manager to engine constructors
Deploy

Runtime risks: Escalation boost changes risk scores for repeat IPs. First deployment: IPs that had accumulated repeat hits but no escalation record will start fresh (escalation_tracker state was empty). They will escalate on future hits. This is correct behavior.
Deployment strategy: Deploy during low-traffic window. Monitor BLOCK decisions for unexpected spikes in first 30 minutes.
Rollback: Remove escalation call from _on_detection_event(). Remove _load_threat_intel_feeds() call from load_models(). Revert engine constructors.
Validation:

5 rapid detections from same IP → verify risk escalation → verify BLOCK or TEMP_BLOCK decision
Verify TI engine is_ready() returns True if feeds directory configured
Add FP suppression for ML engine → verify detection returns "normal" for suppressed pattern


WAVE 4 — OBSERVABILITY + INTEGRATION (Days 8-10)
Objective: Connect SiemExporter to EventBus. Verify all 8 gaps resolved.
Systems touched: D5 SiemExporter, S4 EventBus
Prerequisites: Wave 3 complete
Implementation sequence:

Add on_detection() and on_action() to src/observability/siem_exporter.py
Add subscribe calls in web_app/app.py EventBus wiring block
Deploy

Runtime risks: SIEM export now on hot path. If SIEM endpoint is slow, this adds latency. Wrap in internal queue if SIEM export is non-trivial.
Deployment strategy: Standard restart. Verify SIEM receives events within 1s of detection.

PHASE 7 — FINAL OUTPUT STRUCTURE
7.1 Reconstructed Architecture Model
INIDS is an event-driven monolith. The EventBus is the architectural spine. The detection → risk → policy → action chain runs synchronously within a single Flask worker thread. All four operational phases (inference, risk calculation, policy decision, firewall action) complete before the HTTP response is returned.
7.2 Runtime Execution Flow Analysis
Normal case (~20ms): Request → S2 middleware → S3 ML inference → S4 EventBus chain (RiskEngine, PolicyEngine, ActionExecutor mock) → S9 3 DB writes → S13 3 WebSocket emits → response
Attack case with UFW (~200ms): Same as above but adapter.block() is a subprocess call to UFW taking 100-200ms. All this latency is added to HTTP response time.
Failure case (hung adapter): adapter.block() never returns. Worker thread pinned. After 4 concurrent detections, system stops processing. No detection occurs until adapter recovers or timeout triggers (currently: never, without FIX-01).
7.3 Subsystem Dependency Graph
[Settings] → [OpsStore] → [Allowlist, AlertFilterEngine, FalsePositiveManager, ActionExecutor]
[OpsStore] → [PolicyStore] → [prevention_service.policy] → [PolicyEngine]
[ML Models] → [DetectionService] → [MLEngine] → [EngineRegistry]
[EngineRegistry] → [EngineAggregator] → [/api/detect]
[EventBus] → [DetectionService, RealTimeStreamer, SiemExporter(AFTER FIX-07)]
[ActionExecutor] → [FirewallAdapters] → [OS firewall]
[ActionExecutor] → [OpsStore]
[PreventionScheduler] → [ActionExecutor] → [FirewallAdapters]
7.4 Event Propagation Analysis
DetectionEvent triggers 2 parallel chains in series:

Chain A: _on_detection_event → RiskScoreEvent → _on_risk_event → PolicyDecisionEvent → _on_policy_decision_event → ActionEvent → _on_action_realtime
Chain B: _on_detection_realtime → WebSocket emit

Both run synchronously in the same thread. Chain B is ~1ms. Chain A is 20-200ms.
7.5 Failure Propagation Analysis
adapter hang → thread pinned → pool exhaustion → 502/504 → detection stops
DB write fail → alert/action not persisted → firewall state diverges from DB
EventBus handler exception → logged, chain continues to next handler → SILENT PARTIAL FAILURE
allowlist add → restart → allowlist empty → re-block of trusted IP → incident re-opens
POST /api/policy → returns 200 → runtime unchanged → operator confusion
7.6 Verified vs Inferred Findings Matrix
FindingReport 1Report 2ClassificationF-01 Allowlist persistence✓✓VERIFIEDF-02 TI feeds not loaded✓✓VERIFIEDF-03 EscalationTracker disconnected✓✓VERIFIEDF-04 FP Manager partial✓✓VERIFIEDF-05 Policy runtime stale✓✓VERIFIEDF-06 SIEM not subscribed✓✓VERIFIEDF-09 Synchronous EventBus✓✓VERIFIEDF-10 No adapter timeout✓✓VERIFIEDF-11 Frontend race✓—HIGH-CONFIDENCE INFERENCEF-12 RiskEngine eviction—✓HIGH-CONFIDENCE INFERENCEF-14 DB/adapter desync——HIGH-CONFIDENCE INFERENCE (derived)F-19 Thread pool exhaustion—✓HIGH-CONFIDENCE INFERENCE
7.7 Root Cause Correlation Matrix
Root CauseDownstream EffectsNo subprocess timeout in firewall_adapters.pyF-09, F-10, F-15, F-17, F-19OpsStore methods not implementedF-01, F-14app.py startup wiring incompleteF-02, F-03, F-06prevention_service.policy never reloadedF-05Engine constructors don't accept fp_managerF-04
7.8 Production Risk Classification
RiskP-LevelSystem Survival ImpactThread pool exhaustion from hung adapterP0System stops detecting under attackDB/firewall desync from failed write after actionP0Phantom blocks; unresolvable without manual interventionAllowlist wipe on restartP1Trusted IPs re-blocked; incident re-opensPolicy runtime stale after updateP1Policy changes silently ineffectiveEscalation not wiredP1Repeat attackers never auto-blockedTI engine disabledP2Known-bad IPs missedFP suppression incompleteP2Suppressed patterns still fire in ML/Threshold enginesSIEM pull-onlyP2SIEM data perpetually stale
7.9 Cascading Failure Chains
Chain 1 — Attack under adapter failure:
UFW unresponsive → adapter.block() hangs → Flask worker pinned
→ 4 workers all pinned → API returns 504
→ Detection pipeline stops → attacker continues undetected
→ Alerts queue in clients → clients show stale state
→ Dashboard appears functional (last WebSocket messages cached)
→ Operator unaware detection is stopped
Chain 2 — False positive + restart:
FP detection for trusted IP → BLOCK action → IP blocked
→ Analyst adds to allowlist → works in-memory
→ Deployment/crash → restart → allowlist empty
→ Trusted IP hits system → re-detected → re-blocked
→ Analyst adds to allowlist again → works until next restart
→ Repeat indefinitely
Chain 3 — Policy change failure:
Security policy change: lower thresholds for aggressive blocking
→ POST /api/policy → 200 OK
→ DB updated → version stored
→ Runtime policy object unchanged
→ Attacks continue being under-responded to
→ Operator escalates to "something is wrong"
→ Root cause: restart required (not documented)
→ Restart loses in-memory state
7.10 Stabilization Priorities

FIX-01 (Adapter timeout + CB) — P0, Days 1-2
FIX-02 (Allowlist persistence) — P1, Days 3-4
FIX-03 (Policy runtime reload) — P1, Days 3-4
FIX-04 (Escalation wiring) — P1, Days 5-7
FIX-05 (TI feed loader) — P2, Days 5-7
FIX-06 (FP manager integration) — P2, Days 5-7
FIX-07 (SIEM subscription) — P2, Days 8-10

7.11 Surgical Remediation Specifications
Documented in full in Phase 4 above. Each fix specifies: exact origin, mechanism, subsystems, root cause, operational impact, production risk, fix strategy, behavior change, integration points, dependency implications, regression risks, rollback strategy.
7.12 File-Level Modification Targets
FileFixesChange Typesrc/settings.pyFIX-01Add 2 config fieldssrc/firewall_adapters.pyFIX-01Add timeout=5 to subprocess calls; handle TimeoutExpiredsrc/ips/action_executor.pyFIX-01Add 5 methods, modify block_ip() and execute()src/ips/scheduler.pyFIX-01Wrap unblock() with timeoutsrc/ops_store.pyFIX-02Add 3 methodssrc/prevention/allowlist.pyFIX-02Verify/complete _load(), _persist_add(), _persist_remove()web_app/app.pyFIX-03, FIX-04, FIX-05, FIX-07Modify 2 route handlers, modify _on_detection_event(), modify load_models(), add EventBus subscriptionssrc/detection/engines/ml_engine.pyFIX-06Add fp_manager param + checksrc/detection/engines/threshold_engine.pyFIX-06Add fp_manager param + checksrc/detection/engines/anomaly_engine.pyFIX-06Add fp_manager param + checksrc/observability/siem_exporter.pyFIX-07Add on_detection(), on_action()
7.13 Dependency-Aware Execution Roadmap
Detailed in Phase 6: Wave 1 (Days 1-2), Wave 2 (Days 3-4), Wave 3 (Days 5-7), Wave 4 (Days 8-10). Each wave has prerequisites, risks, deployment strategy, rollback, validation.
7.14 Rollback + Safety Strategy
WaveRollback ScopeRollback MethodData ImpactWave 14 source filesGit revert + restartNoneWave 23 source filesGit revert + restartAllowlist entries remain in DB but ignoredWave 35 source filesGit revert + restartEscalationTracker DB entries benignWave 42 source filesGit revert + restartSIEM events stop flowing; no data loss
No wave requires DB schema changes. All waves are independently rollbackable.
7.15 Runtime Validation Strategy
TestTargetPass CriteriaAdapter timeoutFIX-013s max for hung adapter, returns adapter_timeoutCircuit breaker openFIX-01After 3 failures, next call returns immediatelyCircuit breaker closeFIX-01After 60s, next call attempts adapterAllowlist persistFIX-02Entry survives process restartAllowlist loadFIX-02New instance loads entries from DBPolicy live reloadFIX-03Threshold change takes effect on next detection without restartEscalation activationFIX-043 hits in 5 min → escalated risk scoreTI engine readyFIX-05ti_engine.is_ready() returns True when feeds dir existsFP suppression MLFIX-06Suppressed ML pattern returns "normal"SIEM pushFIX-07Detection event arrives at SIEM within 2s
7.16 Production Hardening Plan
Post-Wave-4 hardening actions:

Add DB/adapter reconciliation metric: After each save_action(), verify action count in DB matches expected. Log any discrepancy as state_desync_detected.
Add EventBus handler latency metrics: Instrument each handler with time.perf_counter() before/after invoke. Emit to MetricsService. Alert if any handler exceeds 50ms.
Add thread pool utilization alert: len(executor._threads) vs max_workers. Alert at 75% utilization.
Add allowlist integrity check: On startup, compare allowlist._entries size to ops_store.list_allowlist() count. Log mismatch.
Add policy version mismatch detection: On startup, compare prevention_service.policy hash to policy_store.load_current() hash. Alert if different.

7.17 Operational Monitoring Requirements
MetricSourceAlert Thresholdadapter_timeout_totalActionExecutor log/metrics>3 in 60s → PagerDutycircuit_breaker_openActionExecutor log/metricsAny → immediate alertflask_worker_threads_activeThread pool instrumentation>3 of 4 workers active → warningeventbus_handler_duration_msHandler instrumentation>100ms → warningriskengine_tracked_sourcesRiskEngine metrics>9000 (90% of 10000) → warningallowlist_load_countStartup log0 when entries expected → warningti_feeds_loaded_countStartup log0 when dir configured → warningescalation_level_perm_blockEscalationTrackerAny Level-4 → notification
7.18 Final Direct Implementation Execution List
DAY 1 — MORNING:
─────────────────
1. src/settings.py
   + adapter_call_timeout_s: float = 3.0
   + adapter_cb_failure_threshold: int = 3
   + adapter_cb_open_duration_s: float = 60.0

2. src/firewall_adapters.py
   + Add timeout=5 to subprocess.run() in UfwFirewallAdapter.block()
   + Add timeout=5 to subprocess.run() in UfwFirewallAdapter.unblock()
   + Add timeout=5 to subprocess.run() in NftablesFirewallAdapter.block()
   + Add timeout=5 to subprocess.run() in NftablesFirewallAdapter.unblock()
   + Add except subprocess.TimeoutExpired → return False in each

3. src/ips/action_executor.py
   + Add imports: concurrent.futures, time, threading.Lock
   + Add to __init__: _cb_failure_count, _cb_open_until, _cb_lock, _cb_failure_threshold, _cb_open_duration_s
   + Add method: _call_adapter_with_timeout(fn, *args)
   + Add method: _circuit_open()
   + Add method: _record_adapter_result(success)
   + Modify block_ip(): replace adapter.block() with _call_adapter_with_timeout()
   + Modify execute(): add circuit open check at entry

4. src/ips/scheduler.py
   + Locate cleanup loop that calls adapter.unblock()
   + Replace direct call with _call_adapter_with_timeout() pattern

DAY 1 — AFTERNOON:
───────────────────
5. Run tests:
   - test_adapter_timeout (mock adapter sleeping 10s → returns False within 3.5s)
   - test_circuit_breaker_opens (3 failures → opens)
   - test_circuit_breaker_closes (60s elapses → attempts again)
   - Load test: 20 concurrent /api/predict with hung mock adapter → no thread exhaustion

6. Deploy Wave 1 to production
7. Monitor: adapter_timeout count, circuit_breaker log events, Flask worker count

DAY 3 — MORNING:
─────────────────
8. src/ops_store.py
   + Add list_allowlist() → list[dict]
   + Add add_allowlist_entry(entry, reason, added_by)
   + Add remove_allowlist_entry(entry)

9. src/prevention/allowlist.py
   + Audit _load(): verify calls list_allowlist(), iterates, normalizes
   + Audit _persist_add(): verify calls add_allowlist_entry()
   + Audit _persist_remove(): verify calls remove_allowlist_entry()
   + If any are incomplete/wrong: fix to match method signatures

DAY 3 — AFTERNOON:
───────────────────
10. web_app/app.py — POST /api/policy handler
    + After policy_store.update(): add prevention_service.policy = policy_store.load_current()
    + Null-check on load_current() result
    + Log successful reload

11. web_app/app.py — POST /api/policy/rollback handler
    + After policy_store.rollback(): add same reload block

12. Run tests:
    - test_allowlist_persist (add → restart → load → contains)
    - test_allowlist_remove_persist (add → remove → restart → not contains)
    - test_policy_reload (POST /api/policy → verify runtime threshold changed)
    - test_policy_rollback (rollback → verify prior threshold in runtime)

13. Deploy Wave 2
14. Monitor: allowlist load count at startup, policy version in logs

DAY 5 — MORNING:
─────────────────
15. web_app/app.py — _on_detection_event()
    + Import: RiskScoreEvent
    + After risk_event = risk_engine.calculate(event):
    + Add try block: escalation_tracker.record_hit(source_ip, severity) → level
    + Add _apply_escalation_to_risk(risk_event, level) → risk_event
    + Add except Exception: log, continue

16. web_app/app.py — add _apply_escalation_to_risk() function
    (as specified in FIX-04)

17. web_app/app.py — load_models() function
    + Add: _load_threat_intel_feeds() call at end of function
    + Add: _load_threat_intel_feeds() function definition
    (as specified in FIX-05)

DAY 5 — AFTERNOON:
───────────────────
18. src/detection/engines/ml_engine.py
    + __init__: add fp_manager=None param
    + evaluate(): add suppression check before return

19. src/detection/engines/threshold_engine.py
    + Same pattern as ml_engine

20. src/detection/engines/anomaly_engine.py
    + Same pattern as ml_engine

21. web_app/app.py — engine constructors
    + Pass fp_manager=fp_manager to MLEngine, ThresholdEngine, AnomalyEngine

22. Run tests:
    - test_escalation_5_hits (5 rapid detections → verify BLOCK decision)
    - test_ti_feed_loading (valid feeds dir → ti_engine.is_ready() == True)
    - test_ti_feed_missing_dir (missing dir → no exception, is_ready() == False)
    - test_fp_suppression_ml (suppress pattern → ML engine returns normal)
    - test_fp_suppression_threshold (same)

23. Deploy Wave 3 during low-traffic window
24. Monitor: BLOCK decision rate for first 30 min (escalation may increase blocking)

DAY 8:
───────
25. src/observability/siem_exporter.py
    + Add on_detection(event: DetectionEvent) → wraps _export_detection in try/except
    + Add on_action(event: ActionEvent) → wraps _export_action in try/except

26. web_app/app.py — EventBus wiring block
    + Add: event_bus.subscribe(DetectionEvent, siem_exporter.on_detection)
    + Add: event_bus.subscribe(ActionEvent, siem_exporter.on_action)
    + Guard with: if siem_exporter is not None

27. Run tests:
    - test_siem_push (detection event → siem_exporter.on_detection called)
    - test_siem_push_nonfatal (siem export fails → EventBus chain continues)

28. Deploy Wave 4
29. Verify SIEM receives events within 2s of detection

DAY 10:
────────
30. Full system validation:
    - All 7 FIX items verified via tests
    - Load test: 500 req/min for 30 min, no thread exhaustion
    - Restart test: allowlist, escalation state, TI feeds all correct post-restart
    - Policy change test: threshold update live without restart
    - SIEM streaming active
    - Circuit breaker functional
    - Escalation ladder functional: 5 hits → BLOCK

Total files modified: 12
Database schema changes: 0
New dependencies: 0 (concurrent.futures is stdlib)
Estimated implementation time: 8 working days
Risk level: Low — all changes are additive or isolated modifications to existing functions; each wave independently rollbackable within 5 minutes