ARCHITECTURE MAP — INTERNAL SUMMARY
Subsystems reconstructed: 16 named components (3 auth systems, prevention pipeline, ML pipeline, 6 storage layers, alert lifecycle, 2 rate limiters + dead code, middleware chain, Docker runtime, Redis/leader election, 4 firewall adapters, threading model, persistence boundaries, configuration loading, WebSocket/event system).

Target architecture: 10 solutions replacing the above with unified auth (OpsStore-backed, RS256), centralized RBAC (single decorator), hardened config (startup fail-closed), frozen concurrency model, Docker secrets, stateless adapters, SHA-256 model verification, OpsStore-only storage, unified Redis rate limiter, non-root container.

44-step roadmap across 6 phases (A–F): Steps 1–7 (Phase A, 0–72h), Steps 8–15 (Phase B, Week 1–2), Steps 16–22 (Phase C, Week 2–4), Steps 23–30 (Phase D, Month 2), Steps 31–38 (Phase E, Month 2–3), Steps 39–44 (Phase F, Quarter 2).

Critical ordering constraint (recovery document, final sentence): Step 2 (mock TI removal) must be deployed and confirmed before Step 26 (disabling dry_run / enabling live blocking). This constraint supersedes all other scheduling flexibility.

EXECUTION PHASE 1 — ARCHITECTURAL VALIDATION

┌──────────────────────────────────────────────────────────────────────┐
│ SUBSYSTEM: AUTHENTICATION REDESIGN (Solution 1)                      │
├──────────────────────────────────────────────────────────────────────┤
│ VALIDATION STATUS: GAPS IDENTIFIED                                   │
│                                                                      │
│ GAPS:                                                                │
│  G-AUTH-1: OpsStore is now the backing store for every auth check    │
│  (revoked_tokens query on every token validation). OpsStore is a     │
│  SQLite/PostgreSQL dependency. The plan does not specify an index     │
│  on revoked_tokens(jti) — without it, every auth check is a full     │
│  table scan on the revocation table, which grows with every issued   │
│  token until the cleanup job runs.                                   │
│  RESOLUTION: Add CREATE INDEX idx_revoked_jti ON revoked_tokens(jti) │
│  in the same migration that creates the revoked_tokens table.        │
│                                                                      │
│  G-AUTH-2: The INIDS_AUTH_COMPAT compatibility window is described   │
│  as "2-sprint" but the recovery document does not define when old    │
│  systems (auth_service.py, auth_jwt.py) are removed. This creates    │
│  an open-ended compat window that risks becoming permanent debt.     │
│  RESOLUTION: Define explicit removal in Phase F Step F-AUTH-REMOVE   │
│  (see Phase 3 implementation targets below). Compat flag removed in  │
│  the same sprint that Phase E regression tests confirm no auth       │
│  failures for the full observation window.                           │
│                                                                      │
│  G-AUTH-3: The INIDS_JWT_PUBLIC_KEY is referenced as a separate      │
│  secret but is absent from the REQUIRED_SECRETS list in Solution 3's │
│  validate_config_at_startup(). Without it, a deployment where the    │
│  public key is missing starts but fails all token verifications.     │
│  RESOLUTION: Add INIDS_JWT_PUBLIC_KEY to REQUIRED_SECRETS list.      │
│                                                                      │
│ MIGRATION HAZARDS:                                                   │
│  — INIDS_AUTH_COMPAT=true means both old bypass-capable auth and     │
│  new auth run simultaneously. If the compat flag is active and       │
│  ALLOW_UNAUTHENTICATED is not yet fully removed from code, the old   │
│  system can still bypass auth while the new system correctly rejects. │
│  MITIGATION: Compat flag activates only the API-key compatibility     │
│  shim (env-var keys → api_keys table lookup). The bypass variable    │
│  must be removed from code (Step 1) before compat mode is enabled.  │
│                                                                      │
│ ROLLBACK HAZARDS:                                                    │
│  — If new-format RS256 tokens have been issued and the rollback      │
│  re-enables the old HS256 system, all RS256 tokens are invalid.     │
│  Forced re-authentication required for all active sessions.          │
│  — The users table in OpsStore is additive. Rolling back the auth   │
│  system does not require dropping it — old env-var auth still works. │
│                                                                      │
│ HIDDEN COUPLING:                                                     │
│  — OpsStore must be fully initialized before any auth check can      │
│  succeed. Auth initialization and OpsStore initialization are        │
│  coupled at startup. If OpsStore fails, all routes fail, not just    │
│  database routes. This is acceptable (fail-closed) but must be       │
│  documented in the runbook.                                          │
│  — require_roles() startup validation depends on all route           │
│  registrations being complete before the validation loop runs.       │
│  Flask blueprints registered after the validation loop would be      │
│  invisible to it. Validation must run after all blueprint             │
│  registrations — after create_app() is fully complete.              │
│                                                                      │
│ PRODUCTION RISKS:                                                    │
│  — The 1-hour token expiry is non-negotiable per the plan. Services │
│  that cache tokens for longer than 1 hour will break after the       │
│  cutover. All API consumers must be audited for token caching.       │
│  — HS256 → RS256 cutover invalidates all existing tokens. Any       │
│  session in flight at cutover time fails. Plan a low-traffic         │
│  maintenance window for this step (Step 18).                        │
│                                                                      │
│ RECOMMENDATION: RESOLVE GAPS G-AUTH-1 through G-AUTH-3 BEFORE       │
│ PROCEEDING TO STEP 16. G-AUTH-1 and G-AUTH-3 are implementation      │
│ details resolvable in the same sprint as Step 16. G-AUTH-2 requires │
│ a Phase F removal entry (see Phase 3 targeting below). These gaps    │
│ do not block Phase A (emergency steps) or Phase B.                  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SUBSYSTEM: RBAC INTEGRATION (Solution 2)                             │
├──────────────────────────────────────────────────────────────────────┤
│ VALIDATION STATUS: GAPS IDENTIFIED                                   │
│                                                                      │
│ GAPS:                                                                │
│  G-RBAC-1: During the transition period (Steps 3–16), Phase A       │
│  applies legacy @require_auth decorators to the 25 undecorated       │
│  routes. The startup validation_all_routes_have_auth_decorator()     │
│  is described in Solution 2 as checking for @require_roles or        │
│  @public_route. It will FAIL at startup in Phase A because the       │
│  bridge decorators are @require_auth, not @require_roles.            │
│  RESOLUTION: The startup validator must accept @require_auth during  │
│  the Phase A–B transition. Introduce a transitional decorator        │
│  @require_auth_legacy that is explicitly recognized by the validator  │
│  and auto-warns: "This route uses legacy auth — migrate to           │
│  @require_roles by [date]." Remove @require_auth_legacy recognition  │
│  from validator in Phase F.                                          │
│                                                                      │
│  G-RBAC-2: The @public_route whitelist is implied (only /health,    │
│  login, refresh endpoints) but the exact list is not enumerated in   │
│  the recovery document. An incomplete whitelist causes startup       │
│  failure; an overly broad whitelist creates auth deserts.            │
│  RESOLUTION: The implementation of Step 3 must produce an explicit   │
│  PUBLIC_ROUTES list: ["/health", "/api/health", "/api/auth/login",  │
│  "/api/auth/refresh", "/api/auth/revoke"]. This list must be        │
│  reviewed by the security lead before Step 16 deployment.           │
│                                                                      │
│  G-RBAC-3: The audit log format for authorization decisions is not   │
│  specified. The audits table schema in OpsStore needs a new event   │
│  type column or a dedicated authorization_decisions table.           │
│  RESOLUTION: Extend OpsStore.audits with event_type column (values: │
│  "auth_success", "auth_failure", "authz_denied", "authz_allowed",   │
│  "audit"). This is a schema migration in Step 13 infrastructure.    │
│                                                                      │
│ MIGRATION HAZARDS:                                                   │
│  — Bootstrap lockout risk: If Step 16 (unified auth) is deployed    │
│  without service accounts for the existing API keys, all API key     │
│  authentication fails immediately. The recovery plan addresses this  │
│  via "create service accounts for existing placeholder keys" but the │
│  order of operations must be: (1) create service accounts in users   │
│  table, (2) hash existing API keys into api_keys table, (3) enable  │
│  UnifiedAuthService. The plan implies but does not sequence these    │
│  three sub-steps explicitly.                                         │
│                                                                      │
│ ROLLBACK HAZARDS:                                                    │
│  — Startup route coverage validation is fail-closed (RuntimeError    │
│  on missing decorator). If a route is accidentally left without a    │
│  decorator after Phase A Step 3, the service fails to start. This   │
│  is intentional but means a missed route in Step 3 is a production  │
│  outage, not a silent gap. Test startup validation in staging first. │
│                                                                      │
│ HIDDEN COUPLING:                                                     │
│  — The RBAC validator runs inside validate_config_at_startup().      │
│  This means RBAC validation runs before models are loaded and before │
│  the prevention scheduler is started. If the validator raises during │
│  a deployment, model files and scheduler never start — clean failure. │
│  — RBAC's role schema must be stable before any route is decorated.  │
│  Adding a new role after routes are decorated requires a re-audit    │
│  of all route decorators to ensure the new role has appropriate      │
│  access. Document the role schema as frozen until Phase C.          │
│                                                                      │
│ PRODUCTION RISKS:                                                    │
│  — Applying @require_auth (Step 3) to 25 routes simultaneously is   │
│  a high-blast-radius change. Any API consumer using those endpoints  │
│  without credentials breaks. A consumer audit BEFORE Step 3         │
│  deployment is non-optional.                                        │
│                                                                      │
│ RECOMMENDATION: RESOLVE G-RBAC-1 BEFORE STEP 3 (it affects         │
│ Phase A). G-RBAC-2 and G-RBAC-3 are pre-Step-16 requirements.      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SUBSYSTEM: PREVENTION ORCHESTRATION (Solution 6)                     │
├──────────────────────────────────────────────────────────────────────┤
│ VALIDATION STATUS: GAPS IDENTIFIED                                   │
│                                                                      │
│ GAPS:                                                                │
│  G-PREV-1: The plan eliminates InMemoryPreventionStore but does not  │
│  define the procedure for handling in-flight prevention operations   │
│  at the moment of the switchover. If an ActionExecutor is in the     │
│  middle of a block_ip() call when the removal deploys, the OpsStore  │
│  write may succeed but the in-memory store reference fails. The      │
│  drain_to_ops_store() is a startup drain, not a live-traffic guard. │
│  RESOLUTION: In Step 12, the removal of InMemoryPreventionStore      │
│  must be sequenced as: (1) stop all writes to InMemoryPreventionStore│
│  (redirect to OpsStore only), (2) drain any remaining in-memory     │
│  state to OpsStore, (3) remove the InMemoryPreventionStore object.  │
│  Step (1) must deploy first with a short observation window (1 hour) │
│  before steps (2)+(3) deploy.                                        │
│                                                                      │
│  G-PREV-2: UFW's behavior when adding a duplicate rule is           │
│  version-dependent. Some UFW versions report an error on duplicate  │
│  block; others silently ignore it. The plan asserts idempotency      │
│  ("adding the same rule twice must not error") but does not provide  │
│  implementation guidance for handling the UFW duplicate case.        │
│  RESOLUTION: In UfwFirewallAdapter.block(), catch subprocess         │
│  CalledProcessError; check if the error message contains "Skipping" │
│  or "already exists" (UFW's success messages for existing rules) —  │
│  treat these as success. Add a test with mocked subprocess output.  │
│                                                                      │
│  G-PREV-3: WebhookFirewallAdapter state recovery after restart is    │
│  explicitly unaddressed. The plan marks reconcile() as skipped for  │
│  stateless adapters. This means after a restart, OpsStore may show  │
│  active blocks that the webhook endpoint no longer has in effect.    │
│  The recovery plan accepts this by making webhook adapters stateless │
│  — but this means webhook-enforced blocks are NOT guaranteed to      │
│  survive restarts. This policy decision is implicit, not stated.     │
│  RESOLUTION: Add explicit documentation: "WebhookFirewallAdapter is  │
│  best-effort. Blocks enforced via webhook are not guaranteed to       │
│  persist across restarts. If persistence is required, use UFW or     │
│  nftables adapters." Add a startup log WARNING if webhook adapter    │
│  is selected: "Webhook adapter does not support restart persistence."│
│                                                                      │
│ MIGRATION HAZARDS:                                                   │
│  — The circuit breaker (3 failures → 60s open) is inherited from    │
│  the current codebase and is not changed by Solution 6. During the  │
│  migration window (Step 12), if ActionExecutor's circuit breaker     │
│  opens due to OpsStore failures, no blocks are applied — fail-open  │
│  for prevention during a database outage. This is the existing       │
│  behavior and is acceptable, but should be documented.              │
│                                                                      │
│ ROLLBACK HAZARDS:                                                    │
│  — If InMemoryPreventionStore is removed and OpsStore has a partial │
│  outage, active blocks are not visible to the prevention system      │
│  until OpsStore recovers. With the old in-memory store, blocks       │
│  would survive a brief OpsStore outage. This is a trade-off the     │
│  plan makes (correct behavior) — document it.                       │
│                                                                      │
│ HIDDEN COUPLING:                                                     │
│  — reconcile() is gated by leader election (is_leader() == True).   │
│  After Step 11 (fail-closed leader election), if Redis fails,       │
│  reconcile() never runs. Active blocks in OpsStore may diverge from │
│  OS firewall state during a Redis outage. This is the correct        │
│  behavior (fail-closed) but operators must be alerted to Redis       │
│  failure immediately.                                               │
│                                                                      │
│ PRODUCTION RISKS:                                                    │
│  — Enabling prevention (disabling dry_run, Step 26) while any mock  │
│  TI indicators remain is a self-DoS risk. Step 2 (mock TI removal)  │
│  is a hard prerequisite confirmed as the "single rule that governs  │
│  the entire sequence."                                              │
│                                                                      │
│ RECOMMENDATION: PROCEED. G-PREV-1 is a deployment procedure gap     │
│ (resolve in Step 12 implementation). G-PREV-2 and G-PREV-3 are      │
│ implementation details resolvable during Step 21.                   │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SUBSYSTEM: STATE OWNERSHIP BOUNDARIES (Solutions 6, 8)               │
├──────────────────────────────────────────────────────────────────────┤
│ VALIDATION STATUS: GAPS IDENTIFIED                                   │
│                                                                      │
│ GAPS:                                                                │
│  G-STATE-1: After consolidation, OpsStore owns alerts, actions,      │
│  audit log, auth data, and RBAC. OpsStore is a single point of       │
│  failure for all security-critical data domains. The plan specifies  │
│  SQLite (dev) or PostgreSQL (prod) but does not specify connection   │
│  pooling, retry behavior, or PostgreSQL failover configuration.      │
│  RESOLUTION: For SQLite path: document that SQLite is single-writer  │
│  and unsuitable for multi-instance deployments without a distributed │
│  lock. For PostgreSQL path: specify minimum connection pool size      │
│  (min=2, max=10) and connection timeout (5 seconds) in OpsStore      │
│  initialization. These are operational decisions, not architecture   │
│  changes, but they must be documented before Phase C deployment.    │
│                                                                      │
│  G-STATE-2: The inids_rbac.db migration (Step 17) occurs while both │
│  databases coexist. Between Step 16 (unified auth deployed) and      │
│  Step 17 (RBAC migrated), rbac_manager.py continues writing to       │
│  inids_rbac.db. New RBAC records created in this window are NOT      │
│  migrated to OpsStore by the existing migration script.             │
│  RESOLUTION: Step 17 must include a final synchronization step:      │
│  after the bulk migration, diff inids_rbac.db against already-       │
│  migrated records and apply any delta. Only then disable             │
│  rbac_manager.py writes.                                            │
│                                                                      │
│  G-STATE-3: Rate limit state is specified as Redis-backed with       │
│  in-memory fallback. Under in-memory fallback, rate limit state is  │
│  not shared across instances — each instance has independent         │
│  counters. This is documented as "acceptable degradation" but the    │
│  plan does not specify the behavior of the fallback (does it fail    │
│  silently or log a warning?). Security decisions based on rate limit  │
│  state during Redis failure should be logged at WARNING level.       │
│                                                                      │
│ MIGRATION HAZARDS:                                                   │
│  — The RBAC data migration (Step 17) cannot use cross-database       │
│  transactions. If the migration script fails midway, OpsStore has    │
│  partial RBAC data and inids_rbac.db is still the authoritative      │
│  source. The migration must be idempotent: re-running it on already- │
│  migrated data must be safe (INSERT OR IGNORE / upsert semantics).  │
│                                                                      │
│ ROLLBACK HAZARDS:                                                    │
│  — After InMemoryAlertStore and InMemoryPreventionStore are removed  │
│  (Step 12), rollback requires re-injecting them as secondary write   │
│  targets. The re-injection path must be tested in staging before    │
│  Step 12 deploys to production.                                     │
│                                                                      │
│ HIDDEN COUPLING:                                                     │
│  — All dashboard queries now go to OpsStore. At high alert volume,  │
│  dashboard pagination queries compete with write traffic on the same │
│  database. The alert deduplication index (Step 28) partially         │
│  mitigates read query cost. Dashboard queries must use indexed paths │
│  only — no unindexed full scans on the alerts table.               │
│                                                                      │
│ PRODUCTION RISKS:                                                    │
│  — Low. All changes are additive (new tables, indexes). No existing │
│  data is deleted until the old stores are explicitly removed.        │
│                                                                      │
│ RECOMMENDATION: PROCEED. All gaps are resolvable at implementation  │
│ time. G-STATE-2 is the highest priority — resolve before Step 17.  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SUBSYSTEM: CONCURRENCY MODEL (Solution 4)                            │
├──────────────────────────────────────────────────────────────────────┤
│ VALIDATION STATUS: CONFIRMED                                         │
│                                                                      │
│ GAPS:                                                                │
│  G-CONC-1 (Minor): The EventBus delivers events synchronously on    │
│  the publisher's thread. With OpsStore now as the sole writer for   │
│  all domains, a slow OpsStore write in an EventBus handler blocks   │
│  the detection pipeline for that request. The plan does not address  │
│  EventBus delivery latency bounding.                                 │
│  RESOLUTION: This is a performance concern, not a correctness or    │
│  security concern. Acceptable for Phase B/C. Add to Phase E          │
│  observability: monitor ml_inference_time and ops_store_write_time  │
│  as separate metrics. If p99 exceeds thresholds, convert EventBus   │
│  to async delivery in a future sprint.                              │
│                                                                      │
│ MIGRATION HAZARDS:                                                   │
│  — PolicyConfig frozen dataclass: all callers of self.policy.X must │
│  be updated to config_manager.get().X. A missed call site retains   │
│  the old mutable reference. The plan explicitly says "grep for all  │
│  .policy. references" — this grep must be run on the entire         │
│  codebase, not just prevention_service.py.                          │
│                                                                      │
│ ROLLBACK HAZARDS:                                                    │
│  — Reverting PolicyConfigManager is clean (revert the file). No data│
│  migration required. Race condition re-emerges on rollback.         │
│                                                                      │
│ HIDDEN COUPLING:                                                     │
│  — CPython GIL provides atomicity for the AnomalyEngine single      │
│  reference assignment. If INIDS is ever run under PyPy or Jython    │
│  (no GIL), this guarantee breaks. The plan notes this and suggests  │
│  using threading.local or RLock for non-CPython. Document this       │
│  assumption in the code comment.                                    │
│  — The prevention scheduler's cleanup_expired_actions() and         │
│  reconcile() both run on the same background thread (leader-gated). │
│  These cannot be concurrent with each other, which is correct. But  │
│  block_ip() on the request thread IS concurrent with the scheduler  │
│  thread. The DB-level unique constraint (Step 20) handles this.     │
│                                                                      │
│ PRODUCTION RISKS:                                                    │
│  — Low after implementation. The concurrency fixes are localized and │
│  well-scoped.                                                       │
│                                                                      │
│ RECOMMENDATION: PROCEED.                                            │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SUBSYSTEM: STORAGE CONSOLIDATION (Solutions 6, 8)                    │
├──────────────────────────────────────────────────────────────────────┤
│ VALIDATION STATUS: GAPS IDENTIFIED                                   │
│                                                                      │
│ GAPS:                                                                │
│  G-STORE-1: The drain_to_ops_store() procedure at startup drains    │
│  InMemoryAlertStore to OpsStore. But the plan does not specify a    │
│  verification step confirming the drain completed before the in-     │
│  memory store is removed. If OpsStore is slow during startup and    │
│  the drain times out, alerts are lost.                              │
│  RESOLUTION: drain_to_ops_store() must return a count of drained    │
│  records and a count of failures. If failures > 0, log CRITICAL and │
│  halt the removal of InMemoryAlertStore for this startup cycle.     │
│  The drain failure is non-fatal to startup (OpsStore is available)  │
│  but must be operationally visible.                                 │
│                                                                      │
│  G-STORE-2: The plan requires that version-gated migrations apply   │
│  only migrations with version > current_db_version. But the current │
│  database (before Step 13 deployment) has no schema_version table.  │
│  The first deployment of Step 13 must bootstrap the schema_version  │
│  table and set the initial version to the current schema state.     │
│  RESOLUTION: The first version-gated migration (version 1) creates  │
│  the schema_version table and records version=1. The startup logic  │
│  checks for schema_version table existence; if absent, creates it   │
│  with version=0 and runs all migrations ≥ 1. If present, runs      │
│  only migrations > current version. This is a standard migration    │
│  bootstrap pattern.                                                 │
│                                                                      │
│ MIGRATION HAZARDS:                                                   │
│  — The unique constraint on actions (Step 20) requires SQLite 3.8.9+│
│  for partial index support. The plan notes this. Verify SQLite       │
│  version in the container before deploying Step 20.                 │
│  Command: python3 -c "import sqlite3; print(sqlite3.sqlite_version)"│
│  Minimum acceptable: 3.8.9. Recommended: 3.35+ (for RETURNING       │
│  clause support which simplifies the save_action() duplicate logic). │
│                                                                      │
│ ROLLBACK HAZARDS:                                                    │
│  — Each migration must have a corresponding down-migration script.   │
│  The plan acknowledges this for column ADD operations. The unique    │
│  constraint (Step 20) rollback is DROP INDEX — confirm the index name│
│  is consistent across environments.                                 │
│  — The most dangerous rollback is from Step 17 (RBAC migration).    │
│  If OpsStore RBAC data is partially migrated and rbac_manager.py    │
│  is restored, the two systems will diverge. The rollback procedure  │
│  must restore inids_rbac.db from the pre-migration backup, not from │
│  the current (partially-migrated) OpsStore state.                   │
│                                                                      │
│ HIDDEN COUPLING:                                                     │
│  — OpsStore migrations run unconditionally on startup (current       │
│  behavior). Step 13 changes this to version-gated. But between      │
│  Step 12 (stores consolidated) and Step 13 (migrations version-     │
│  gated), the startup migration risk is highest because OpsStore     │
│  now holds all alert and action data. Step 13 should deploy before  │
│  Step 12 per the plan's own sequencing — confirmed in the roadmap   │
│  summary: Step 13 is a dependency for Step 12.                      │
│                                                                      │
│ PRODUCTION RISKS:                                                    │
│  — The deduplication index (Step 28) locks the alerts table during  │
│  index creation. On large tables in SQLite, this is a blocking       │
│  write-lock. Schedule Step 28 during a maintenance window.          │
│                                                                      │
│ RECOMMENDATION: PROCEED. G-STORE-1 and G-STORE-2 are implementation │
│ details resolvable in Steps 12 and 13 respectively. All other gaps  │
│ are documented operational procedures.                              │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SUBSYSTEM: DEPLOYMENT HARDENING (Solution 10)                        │
├──────────────────────────────────────────────────────────────────────┤
│ VALIDATION STATUS: GAPS IDENTIFIED                                   │
│                                                                      │
│ GAPS:                                                                │
│  G-DEPLOY-1: The health check in the Dockerfile uses `curl` but     │
│  python:3.11-slim does not include curl. The health check command    │
│  will fail silently (container appears unhealthy immediately).       │
│  RESOLUTION: Either add to Dockerfile:                              │
│    RUN apt-get update && apt-get install -y --no-install-recommends  │
│    curl && rm -rf /var/lib/apt/lists/*                              │
│  Or replace the health check with a Python-native approach:          │
│    CMD ["python", "-c",                                              │
│    "import urllib.request; urllib.request.urlopen(                  │
│    'http://localhost:5000/health', timeout=5)"]                      │
│  Python-native is preferred as it adds no extra attack surface.     │
│                                                                      │
│  G-DEPLOY-2: The Dockerfile CMD specifies gunicorn but the         │
│  application uses Flask-SocketIO. Flask-SocketIO requires specific  │
│  worker configuration. The plan specifies --worker-class eventlet   │
│  and -w 1 (one worker) which is correct for SocketIO, but gunicorn  │
│  must also be in requirements.txt and eventlet must be installed.   │
│  Verify both are in requirements.txt before Step 5 deploys.         │
│                                                                      │
│  G-DEPLOY-3: The `read_only: true` filesystem requires all write    │
│  paths to be on tmpfs or mounted volumes. The application may write  │
│  to paths not covered by /tmp or /data. Specifically: Python .pyc   │
│  compilation, log file writes (if file-based logging is configured), │
│  and any libraries that write to their own directories. An audit of  │
│  all file write operations must be performed before enabling         │
│  read_only.                                                         │
│  RESOLUTION: Run the container WITHOUT read_only first, capture all  │
│  write operations with `strace -e trace=write,open,creat -f`, then  │
│  enumerate all non-/tmp and non-/data paths. Add tmpfs mounts for  │
│  any Python cache paths (e.g., /app/__pycache__ → tmpfs). Then      │
│  enable read_only.                                                   │
│                                                                      │
│  G-DEPLOY-4: Service startup ordering between containers (if Redis  │
│  and the app container are separate) is not specified with Docker   │
│  Compose health check dependencies. Without `depends_on:            │
│  redis: { condition: service_healthy }`, the app may start before   │
│  Redis is ready, causing leader election to fail on first attempt.  │
│  RESOLUTION: Add depends_on with service_healthy condition for Redis │
│  in docker-compose.yml. Redis must have its own healthcheck defined.│
│                                                                      │
│ MIGRATION HAZARDS:                                                   │
│  — Removing the source tree volume mount (../../:/app) requires all  │
│  file paths in the application that reference /app/... to be        │
│  re-mapped to /data/... or /models/.... Any hardcoded path to the   │
│  source tree causes a startup failure with read_only filesystem.    │
│  INIDS_DB_URL must point to /data/inids_ops.db, not to a path       │
│  within /app.                                                       │
│                                                                      │
│ ROLLBACK HAZARDS:                                                    │
│  — Container rebuild is required for Step 5. Rollback requires       │
│  rebuilding the previous image. Ensure the previous image is tagged  │
│  and retained in the registry before deploying the hardened image.  │
│                                                                      │
│ HIDDEN COUPLING:                                                     │
│  — Non-root user (inids) must have read permissions on /models and  │
│  write permissions on /data. Volume mounts must be owned by the      │
│  inids user (UID/GID from the container). On the host, the volume   │
│  directories must be chowned to the container's UID before the      │
│  container starts. This is an operational requirement.              │
│                                                                      │
│ PRODUCTION RISKS:                                                    │
│  — High for Step 5. Container rebuild and volume reconfiguration is │
│  a significant operational change. Validate in a staging environment │
│  before production deployment. The rollback is a container rebuild  │
│  (not instant).                                                     │
│                                                                      │
│ RECOMMENDATION: RESOLVE G-DEPLOY-1 BEFORE STEP 5. G-DEPLOY-2       │
│ through G-DEPLOY-4 are implementation-time verifications. None of   │
│ these gaps make the deployment plan UNSAFE AS WRITTEN — they are     │
│ operational details that must be confirmed during implementation.   │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SUBSYSTEM: ML MODEL TRUST CHAIN (Solution 7)                         │
├──────────────────────────────────────────────────────────────────────┤
│ VALIDATION STATUS: GAPS IDENTIFIED                                   │
│                                                                      │
│ GAPS:                                                                │
│  G-ML-1: The "trusted artifact store" for model download is          │
│  referenced but not defined. The trust chain is: artifact store →   │
│  TLS verification → SHA-256 checksum → joblib.load(). But if the    │
│  artifact store itself is compromised, the checksum stored in the    │
│  repo may not match the malicious model, which is the desired        │
│  behavior — the plan's integrity check IS the defense. The gap is   │
│  that the download script and its authentication method are not      │
│  specified. An unauthenticated download from a misconfigured S3      │
│  bucket is not a "trusted" artifact store.                           │
│  RESOLUTION: The model download script (used in container startup)  │
│  must: (1) connect to the artifact store via TLS with certificate    │
│  verification enabled, (2) use an authenticated download (IAM role, │
│  token, or key), (3) download to a tempdir first, (4) verify        │
│  checksum, (5) move to /models only after verification. Document    │
│  the artifact store type (S3, GCS, registry) in the runbook.        │
│                                                                      │
│  G-ML-2: The inference timeout for model.predict() is not bounded.  │
│  An adversarially crafted feature vector with extreme cardinality   │
│  could cause a slow inference path. The plan specifies graceful      │
│  degradation on exception but does not bound execution time.        │
│  RESOLUTION: Wrap inference in a ThreadPoolExecutor future with a   │
│  timeout: future.result(timeout=INIDS_INFERENCE_TIMEOUT_S, default  │
│  0.5). On TimeoutError, return the unknown verdict (same as other   │
│  inference failures). This bounds DoS surface via slow inference.   │
│                                                                      │
│  G-ML-3: The ACTIVE_MODELS manifest is committed to the repo. If   │
│  an attacker gains repo write access, they can update the manifest  │
│  to reference a different model version. The checksums.sha256 file  │
│  is also in the repo. Updating both files in one commit bypasses    │
│  the integrity check.                                               │
│  RESOLUTION: The checksums.sha256 file must be signed with a        │
│  separate key (e.g., GPG signing by the ML team lead, verified at   │
│  startup). Alternatively, store checksums in a separate, more        │
│  tightly access-controlled location (e.g., the artifact store, not  │
│  the main repo). For the current recovery sprint, treat repo commit  │
│  signing (enforced via branch protection rules) as the mitigation.  │
│  Document this as a known residual risk until signing is implemented.│
│                                                                      │
│ MIGRATION HAZARDS:                                                   │
│  — Generating checksums.sha256 against the current models requires  │
│  verifying those models are the known-good versions. If a model has  │
│  been modified (even legitimately) since last checksum generation,   │
│  Step 6 deployment fails at startup. Run checksum generation in a   │
│  controlled environment against models downloaded from the artifact  │
│  store, not from the currently mounted volume.                      │
│                                                                      │
│ ROLLBACK HAZARDS:                                                    │
│  — INIDS_MODEL_VERIFY=warn allows the service to start even with a  │
│  bad checksum (logs a warning but proceeds). This is the specified  │
│  rollback mechanism for Step 6. However, running with a           │
│  INIDS_MODEL_VERIFY=warn in production re-opens the attack surface   │
│  that Step 6 closes. The warn mode is for rollback only — not for   │
│  extended operation. Add a CRITICAL log entry every 5 minutes while │
│  warn mode is active to prevent it from being forgotten.           │
│                                                                      │
│ HIDDEN COUPLING:                                                     │
│  — load_models() and AnomalyEngine each call joblib.load()           │
│  independently. Step 6 must apply load_model_with_verification() to │
│  BOTH call sites. A missed call site means the AnomalyEngine model  │
│  path remains unverified.                                           │
│                                                                      │
│ PRODUCTION RISKS:                                                    │
│  — Medium. Step 6 adds startup failure on bad checksum. Test with   │
│  production model files before deployment to avoid surprise startup  │
│  failures.                                                          │
│                                                                      │
│ RECOMMENDATION: PROCEED. G-ML-1 must be resolved before Step 6      │
│ (as an operational procedure document, not a code change). G-ML-2   │
│ and G-ML-3 are resolvable in Phases D and F respectively.           │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SUBSYSTEM: RATE LIMITING UNIFICATION (Solution 9)                    │
├──────────────────────────────────────────────────────────────────────┤
│ VALIDATION STATUS: GAPS IDENTIFIED — SPECIFICATION INCONSISTENCY     │
│                                                                      │
│ CRITICAL GAP:                                                        │
│  G-RATE-1: The recovery plan states the unified rate limiter is      │
│  "registered as a Flask before_request hook, after auth (so user    │
│  identity is available for per-user limits)." This is internally     │
│  inconsistent. Flask before_request hooks execute BEFORE route       │
│  handler decorators. Auth decorators (@require_roles) execute as     │
│  part of the route handler dispatch, NOT in before_request. At       │
│  before_request time, flask.g.auth does not yet exist — auth has    │
│  not run. Per-user rate limiting in a before_request hook cannot    │
│  access the authenticated user identity.                             │
│                                                                      │
│  This is a specification error. As written, the per-user rate limit  │
│  (200/min for authenticated users) will fail silently at runtime     │
│  because flask.g.auth is None during before_request execution.       │
│                                                                      │
│  OPERATIONAL FAILURE: Per-user rate limits never engage. All traffic │
│  falls through to the global IP limit (1000/min). Authenticated       │
│  users can exceed their per-user limit with no enforcement.          │
│                                                                      │
│  CORRECTION (preserving architectural intent):                       │
│  Split the unified rate limiter into two tiers, both at the correct  │
│  execution points:                                                   │
│    TIER 1 — before_request hook: Global per-IP limit (1000/min).    │
│    No user identity required. Runs before auth. Protects against     │
│    unauthenticated DDoS.                                            │
│    TIER 2 — inside require_roles() decorator: Per-user and per-route │
│    limit applied after AuthContext is constructed. require_roles()   │
│    already has the AuthContext — add rate_limiter.check(user_id, ...)│
│    call after auth succeeds. Route-specific @rate_limit() decorators │
│    apply here as well.                                              │
│  This preserves the "unified rate limiter" architectural intent: one │
│  RateLimiter class, one Redis-backed RateLimitStore, two enforcement │
│  points positioned correctly in the request lifecycle. The RateLimiter│
│  class interface from Solution 9 is unchanged.                       │
│                                                                      │
│  SEQUENCE CHANGE: This correction does not change the step           │
│  sequencing. Step 16 (unified auth) remains the prerequisite for    │
│  per-user limits because require_roles() is the enforcement point.  │
│  The before_request global IP limit can be deployed in Step 19 as   │
│  planned.                                                           │
│                                                                      │
│ ADDITIONAL GAPS:                                                     │
│  G-RATE-2: The dead code rate limiter in production_hardening.py    │
│  (SecurityHardeningManager.enforce_rate_limit with non-resetting     │
│  counter) is addressed in Step 23 (delete the module). But between  │
│  Step 19 (rate limiter unified) and Step 23 (dead code removed),    │
│  the dead code remains. Ensure no developer imports it during this  │
│  window. Add a CI check: grep -r "from.*production_hardening" → fail │
│  if any import found (after Step 19 confirms it's safe to remove).  │
│                                                                      │
│ MIGRATION HAZARDS:                                                   │
│  — Deploying the new rate limiter BEFORE removing the old ones is   │
│  the correct sequence (per the plan). During the overlap window,    │
│  requests are rate-limited by both the old and new systems. This    │
│  is more restrictive than intended but not unsafe. Monitor for       │
│  false-positive 429 responses during the overlap window.            │
│                                                                      │
│ ROLLBACK HAZARDS:                                                    │
│  — If the new rate limiter is removed after the old ones are deleted,│
│  the system has no rate limiting. The minimum safe rollback state    │
│  is: new limiter active OR both old limiters active. Never remove    │
│  both old limiters before confirming the new limiter is operational. │
│                                                                      │
│ RECOMMENDATION: RESOLVE G-RATE-1 BEFORE STEP 19. The split-tier    │
│ correction preserves the architectural intent and does not change   │
│ the step sequencing or class interface. Implementation engineers     │
│ must apply this correction in Step 19 implementation.              │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SUBSYSTEM: STARTUP VALIDATION LOGIC (Solution 3)                     │
├──────────────────────────────────────────────────────────────────────┤
│ VALIDATION STATUS: GAPS IDENTIFIED                                   │
│                                                                      │
│ GAPS:                                                                │
│  G-START-1: INIDS_JWT_PUBLIC_KEY is absent from REQUIRED_SECRETS    │
│  (already flagged as G-AUTH-3). Critical — public key missing means │
│  all token verifications fail at runtime rather than at startup.    │
│  RESOLUTION: Add "INIDS_JWT_PUBLIC_KEY" to REQUIRED_SECRETS.        │
│                                                                      │
│  G-START-2: The route coverage validator (validate_all_routes_have_ │
│  auth_decorator) must run AFTER all blueprints are registered. In   │
│  Flask's application factory pattern, this means calling the         │
│  validator at the END of create_app(), after all blueprint.register  │
│  calls. If called too early, not all routes are visible.            │
│  RESOLUTION: Place validate_all_routes_have_auth_decorator(app)     │
│  as the LAST call inside create_app(), after all blueprint           │
│  registrations. Add a comment explicitly documenting this constraint.│
│                                                                      │
│  G-START-3: Database readiness is not verified with retry logic.    │
│  OpsStore.__init__() raises if the database is unavailable. In       │
│  Docker Compose environments where the database container starts     │
│  before the app container is fully ready, the app exits immediately  │
│  and Docker restarts it. Without exponential backoff, crash-looping │
│  restarts can trigger repeated pip install (if not moved to build   │
│  time — which Step 5 addresses) or model downloads.                 │
│  RESOLUTION: After Step 5, pip is at build time. But model download │
│  is still at startup time. Add a readiness retry for OpsStore with  │
│  max_retries=5, backoff_seconds=[1, 2, 4, 8, 16]. If all retries    │
│  fail, raise RuntimeError (fail-closed).                            │
│                                                                      │
│  G-START-4: The startup validator checks ALLOW_UNAUTHENTICATED env  │
│  var and raises if found set to true. But it does not check if the  │
│  variable exists and is set to any value other than "true" — e.g.,  │
│  ALLOW_UNAUTHENTICATED=1 or ALLOW_UNAUTHENTICATED=yes. The check   │
│  must be case-insensitive and handle truthy variants.               │
│  RESOLUTION: Check: if os.environ.get("ALLOW_UNAUTHENTICATED","")  │
│  .lower() in {"true", "1", "yes", "on"}: raise RuntimeError(...)   │
│                                                                      │
│ MIGRATION HAZARDS:                                                   │
│  — The startup validator is fail-closed. A misconfigured secret     │
│  name (e.g., environment variable not injected by Docker secrets    │
│  correctly) causes the container to fail to start and never become  │
│  healthy. This is the desired behavior but means deployment failures │
│  surface as startup failures, not runtime errors. Monitor container │
│  exit codes and docker logs after every deployment.                 │
│                                                                      │
│ ROLLBACK HAZARDS:                                                    │
│  — None specific. Reverting the validation logic restores the fail- │
│  open behavior.                                                     │
│                                                                      │
│ HIDDEN COUPLING:                                                     │
│  — validate_config_at_startup() is called early in app initialization│
│  (before model load). If this call raises, models are never loaded.  │
│  This is correct (fail-closed). But the error message from a missing │
│  secret should clearly identify WHICH secret is missing to aid      │
│  diagnosis.                                                         │
│                                                                      │
│ PRODUCTION RISKS:                                                    │
│  — Low for Step 1 (ALLOW_UNAUTHENTICATED check is minimal code).   │
│  Higher for Solution 3 full implementation (Step 16 prerequisite):  │
│  missing JWT key in the secrets injection causes production outage.  │
│                                                                      │
│ RECOMMENDATION: PROCEED. G-START-1 through G-START-4 are all        │
│ implementation-time details resolvable in Phase B/C.                │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SUBSYSTEM: REGRESSION PREVENTION STRATEGY (Phase 5)                  │
├──────────────────────────────────────────────────────────────────────┤
│ VALIDATION STATUS: GAPS IDENTIFIED                                   │
│                                                                      │
│ GAPS:                                                                │
│  G-REG-1: Attack chain AC-3 (supply chain compromise at container   │
│  boot) has no automated regression test. The operational mitigation  │
│  (Step 7 hash verification) is correct, but there is no CI gate     │
│  confirming the mitigation is in place.                              │
│  RESOLUTION: Add to CI pre-merge checks:                            │
│    # Verify --hash= entries are present in requirements.txt          │
│    grep '\-\-hash=sha256' requirements.txt | wc -l | \              │
│    xargs -I{} bash -c '[ {} -gt 0 ] || (echo "No hash pins found"; exit 1)'│
│  Add as tests/security/test_supply_chain_mitigations.py:            │
│    def test_requirements_are_pinned_with_hashes(): ...              │
│    def test_requirements_has_no_yanked_packages(): ... (pip-audit)  │
│                                                                      │
│  G-REG-2: Attack chain AC-3 also includes the vector of the repo    │
│  volume mount. A test should verify the container configuration does │
│  NOT mount the source repository:                                   │
│    def test_container_does_not_mount_source_tree():                 │
│      # Parse docker-compose.yml, assert no volume contains ../../   │
│      ...                                                            │
│                                                                      │
│  G-REG-3: Chain C6 (SECRET_KEY compromise → JWT forgery) has no    │
│  regression test. The placeholder rejection test (test_placeholder_ │
│  secret_key_rejected) only verifies that weak keys are rejected at  │
│  startup. It does not verify that a correct-format but compromised   │
│  HS256 key cannot be used to forge a JWT token after the RS256      │
│  migration.                                                         │
│  RESOLUTION: Add test_hs256_token_rejected_after_rs256_migration():  │
│    hs256_token = jwt.encode({"sub": "admin", "roles": ["admin"]},   │
│    "any-secret", algorithm="HS256")                                 │
│    resp = client.get("/api/alerts",                                 │
│    headers={"Authorization": f"Bearer {hs256_token}"})              │
│    assert resp.status_code == 401  # RS256-only service rejects it  │
│                                                                      │
│  G-REG-4: Test suite placement in Phase F (Step 40) means all       │
│  Tier 0 and Tier 1 fixes (Phases A, B, C) are deployed WITHOUT      │
│  regression tests covering their security properties. The tests       │
│  that verify AC-1 through AC-4 are not written until Quarter 2.     │
│  This violates quality criterion 8: "every attack chain has a        │
│  corresponding regression test specification."                       │
│  RESOLUTION: The test SUITE (full coverage, CI gate) is Phase F.    │
│  But the security regression tests for each Phase A step MUST be    │
│  written IN THE SAME SPRINT as the step they validate. Security      │
│  tests are not deferred. Implementation sequencing:                 │
│    Step 1 → write test_auth_bypass_disabled() in same PR            │
│    Step 4 → write test_jwt_no_password_no_token() in same PR        │
│    Step 6 → write test_model_load_rejects_bad_checksum() in same PR │
│  The Phase F Step 40 assembles these into a unified suite with      │
│  coverage gates. Individual tests are written per-step.             │
│                                                                      │
│ MIGRATION HAZARDS:                                                   │
│  — CI gates that fail build before merge require the test            │
│  infrastructure (test runner, test database, mock Redis) to be       │
│  available in CI. Ensure CI environment is configured before         │
│  enforcing the gate.                                                │
│                                                                      │
│ ROLLBACK HAZARDS:                                                    │
│  — Tests are non-breaking additions. Rollback of test code does not │
│  affect production behavior.                                        │
│                                                                      │
│ RECOMMENDATION: RESOLVE G-REG-4 IMMEDIATELY. This changes the       │
│ implementation procedure: every Phase A, B, and C step must include  │
│ its corresponding security regression test in the same PR.           │
│ G-REG-1, G-REG-2, G-REG-3 are resolvable in Phase E/F.            │
└──────────────────────────────────────────────────────────────────────┘
EXECUTION PHASE 2 — DEPENDENCY GRAPH & MIGRATION SEQUENCING

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Auth Bypass Removal (Step 1)                               │
│ TIER: 0                                                              │
│ REQUIRES: Nothing                                                    │
│ ENABLES: Steps 3, 4, 5, 15 (all auth fixes now have effect)         │
│ COMPATIBILITY WINDOW REQUIRED: NO                                    │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Before .env commit is deployed (immediate revert) │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing — this is the first step     │
│ WHAT BREAKS IF DONE TOO LATE: Every hour of delay = full auth bypass │
│   on all deployed instances                                          │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Mock Threat Intel Removal (Step 2)                         │
│ TIER: 0                                                              │
│ REQUIRES: Nothing                                                    │
│ ENABLES: Step 26 (safe to enable prevention mode)                   │
│ COMPATIBILITY WINDOW REQUIRED: NO                                    │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Immediate (revert load_threat_intel())            │
│ WHAT BREAKS IF DONE TOO EARLY: TI engine produces no matches until  │
│   feed is configured (acceptable — false negatives preferred)        │
│ WHAT BREAKS IF DONE TOO LATE: Enabling auto_block before this step  │
│   causes immediate self-DoS on internal infrastructure               │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Auth All Undecorated Routes (Step 3)                       │
│ TIER: 0                                                              │
│ REQUIRES: Step 1 (auth bypass disabled — decorators must have effect)│
│ ENABLES: Steps 24, 25 (input hardening now meaningful)               │
│ COMPATIBILITY WINDOW REQUIRED: NO                                    │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: git revert of decorator additions                 │
│ WHAT BREAKS IF DONE TOO EARLY: Decorators exist but bypass is active│
│   — false sense of security                                          │
│ WHAT BREAKS IF DONE TOO LATE: FP suppression endpoint remains open  │
│   as a detection kill switch                                         │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Break Passwordless JWT Login (Step 4)                      │
│ TIER: 0                                                              │
│ REQUIRES: Step 1 (API keys are now genuine after rotation)           │
│ ENABLES: Authenticated JWT tokens carry real identity; Step 16       │
│   (permanent auth system can build on this bridge)                  │
│ COMPATIBILITY WINDOW REQUIRED: YES (bridge behavior active until     │
│   Step 16 replaces it)                                               │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Revert api_auth_login handler (immediate)         │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing — purely additive security   │
│ WHAT BREAKS IF DONE TOO LATE: Admin JWT issuance remains open to    │
│   any caller with HTTP access                                        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Docker Deployment Hardening (Step 5)                       │
│ TIER: 0                                                              │
│ REQUIRES: Step 1 (secrets are outside repo — safe to remove repo     │
│   mount)                                                             │
│ ENABLES: Steps 6, 7 (model security and supply chain security now    │
│   meaningful — attack surface for model substitution is closed)      │
│ COMPATIBILITY WINDOW REQUIRED: NO (container rebuild, no traffic     │
│   coexistence needed)                                                │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Docker image tag retention (previous image        │
│   available for immediate re-deploy)                                 │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing (can be done any time after  │
│   Step 1, but secrets must be outside repo first)                   │
│ WHAT BREAKS IF DONE TOO LATE: Volume mount + root = host-level      │
│   blast radius for any exploit in Steps 3-4 gap window              │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Model Integrity Verification (Step 6)                      │
│ TIER: 0                                                              │
│ REQUIRES: Step 5 (model directory is now a controlled volume)        │
│ ENABLES: Step 8 (safe to trust ML pipeline integrity), Step 41       │
│   (hot reload safety depends on integrity check)                     │
│ COMPATIBILITY WINDOW REQUIRED: YES (INIDS_MODEL_VERIFY=warn as       │
│   rollback/fallback during adoption window)                          │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Set INIDS_MODEL_VERIFY=warn (no code change)      │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing — but must be after Step 5   │
│   or the model directory is still the full repo mount               │
│ WHAT BREAKS IF DONE TOO LATE: Malicious model substitution remains   │
│   viable through the /models volume                                  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Pin Requirements With Hashes (Step 7)                      │
│ TIER: 0                                                              │
│ REQUIRES: Step 5 (pip install moved to build time — hash             │
│   verification now covers the single remaining fetch point)          │
│ ENABLES: Step 42 (CVE remediation builds on pinned baseline)         │
│ COMPATIBILITY WINDOW REQUIRED: NO                                    │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Revert requirements.txt and rebuild image         │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing — can be done any time after │
│   Step 5                                                             │
│ WHAT BREAKS IF DONE TOO LATE: Every image build is a supply chain   │
│   attack surface                                                     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: ML Inference Graceful Degradation (Step 8)                 │
│ TIER: 1                                                              │
│ REQUIRES: Step 6 (model integrity established — safe to trust         │
│   inference is operating on clean models)                            │
│ ENABLES: Reliable multi-engine pipeline; Step 41 (hot reload safety) │
│ COMPATIBILITY WINDOW REQUIRED: NO                                    │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Revert ml_engine.py and anomaly_engine.py         │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing                               │
│ WHAT BREAKS IF DONE TOO LATE: Model transient failures crash         │
│   the detection pipeline for individual requests                     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: PolicyConfig Race Fix (Step 9)                             │
│ TIER: 1                                                              │
│ REQUIRES: Nothing (pure concurrency fix, standalone)                 │
│ ENABLES: Step 11 (clean prevention scheduler pause), Step 26         │
│   (policy enforcement hardening), Step 16 (auth config uses same    │
│   immutability pattern)                                              │
│ COMPATIBILITY WINDOW REQUIRED: NO                                    │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Revert prevention_service.py                      │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing                               │
│ WHAT BREAKS IF DONE TOO LATE: Race condition persists in prevention  │
│   pipeline while blocking is active                                  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: AnomalyEngine Model Race Fix (Step 10)                     │
│ TIER: 1                                                              │
│ REQUIRES: Nothing (self-contained)                                   │
│ ENABLES: Step 41 (hot reload depends on atomic model reference swap) │
│ COMPATIBILITY WINDOW REQUIRED: NO                                    │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Revert anomaly_engine.py                          │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing                               │
│ WHAT BREAKS IF DONE TOO LATE: Active data race between fit() and     │
│   evaluate() under concurrent high traffic                           │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Leader Election Fail-Closed (Step 11)                      │
│ TIER: 1                                                              │
│ REQUIRES: Step 9 (PolicyConfigManager — prevention scheduler pause   │
│   is clean; config not partially written when scheduler stops)       │
│ ENABLES: Step 16 (safe multi-instance prevention), Step 21           │
│   (reconcile runs reliably — spurious DESYNCED must be eliminated   │
│   before reliable reconciliation can be trusted)                    │
│ COMPATIBILITY WINDOW REQUIRED: NO (but single-instance deployments  │
│   without Redis need INIDS_REDIS_REQUIRED=false flag)                │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Revert leader_election.py                         │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing                               │
│ WHAT BREAKS IF DONE TOO LATE: Redis outage causes split-brain →      │
│   duplicate firewall rules → audit log flood                         │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Version-Gated Schema Migrations (Step 13)                  │
│ TIER: 1                                                              │
│ REQUIRES: Nothing (standalone OpsStore change)                       │
│ ENABLES: Steps 12 (alert store consolidation), 20 (unique            │
│   constraint), 28 (alert dedup), 29 (health probe), 38 (public API) │
│ COMPATIBILITY WINDOW REQUIRED: NO                                    │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Keep old migration functions as fallback for      │
│   one sprint; revert by re-enabling them                             │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing                               │
│ WHAT BREAKS IF DONE TOO LATE: Unconditional UPDATE migrations cause  │
│   startup latency spikes as OpsStore write volume increases post-12  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Eliminate Duplicate Alert/Action Stores (Step 12)          │
│ TIER: 1                                                              │
│ REQUIRES: Steps 1-3 (auth enforced — OpsStore now trusted authority),│
│   Step 13 (migration infrastructure for future schema changes)       │
│ ENABLES: Steps 28, 30 (alert dedup and pagination now safe), clean  │
│   dashboard reads                                                    │
│ COMPATIBILITY WINDOW REQUIRED: YES (drain window before removal)     │
│ DUAL-WRITE REQUIRED: YES — Phase 1: write to both stores + read     │
│   from in-memory; Phase 2: write to both + read from OpsStore;      │
│   Phase 3: write to OpsStore only + read from OpsStore              │
│ ROLLBACK BOUNDARY: Re-inject InMemoryAlertStore as secondary write  │
│   target                                                             │
│ WHAT BREAKS IF DONE TOO EARLY: API routes that read from in-memory  │
│   stores are not yet updated → they return empty results             │
│ WHAT BREAKS IF DONE TOO LATE: Dashboard shows divergent data;        │
│   security audit is based on incomplete record                       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Register CORS Middleware (Step 14)                         │
│ TIER: 1                                                              │
│ REQUIRES: Steps 1-3 (auth establishes what to protect with CORS)    │
│ ENABLES: Step 32 (CSP unsafe-inline removal — CORS must work first) │
│ COMPATIBILITY WINDOW REQUIRED: NO                                    │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Remove CORS registration (CORS reverts to dead   │
│   code state — no enforcement)                                       │
│ WHAT BREAKS IF DONE TOO EARLY: Without auth (Step 1-3), CORS        │
│   enforcement on open endpoints is ineffective                       │
│ WHAT BREAKS IF DONE TOO LATE: Cross-origin requests to authenticated │
│   endpoints from any webpage remain possible                         │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Fix Sensor Key Role (Step 15)                              │
│ TIER: 1                                                              │
│ REQUIRES: Step 1 (auth is enabled — role assignment has effect)      │
│ ENABLES: Principle of least privilege for sensor nodes               │
│ COMPATIBILITY WINDOW REQUIRED: NO                                    │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Revert auth_service.py:53 (single line)          │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing                               │
│ WHAT BREAKS IF DONE TOO LATE: Compromised sensors retain analyst     │
│   access to all read endpoints                                       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Unified Authentication System (Step 16)                    │
│ TIER: 1                                                              │
│ REQUIRES: Steps 1-4 (bridge auth), Step 13 (migration infra),        │
│   Step 9 (config immutability for auth config)                       │
│ ENABLES: Steps 17 (RBAC migration), 18 (JWT hardening), 19           │
│   (rate limiter unification — user identity now available), 22       │
│   (audit trail integrity)                                            │
│ COMPATIBILITY WINDOW REQUIRED: YES — INIDS_AUTH_COMPAT=true for 2   │
│   sprints while old auth systems handle legacy callers               │
│ DUAL-WRITE REQUIRED: YES — Both auth systems validate requests during │
│   the compat window; new auth logs decisions separately for audit    │
│ ROLLBACK BOUNDARY: Set INIDS_AUTH_COMPAT=true (old system takes     │
│   over immediately); new users table is additive (no data loss)      │
│ WHAT BREAKS IF DONE TOO EARLY: Bootstrap lockout if service accounts │
│   not pre-created in users table                                     │
│ WHAT BREAKS IF DONE TOO LATE: Bridge fixes (Steps 1, 4) remain the  │
│   permanent auth architecture — accumulating security debt           │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Migrate RBAC Into Unified Auth (Step 17)                   │
│ TIER: 1                                                              │
│ REQUIRES: Step 16 (unified auth must exist as the target)            │
│ ENABLES: Step 22 (unified audit trail), Step 23 (dead code removal   │
│   safe), single authorization truth                                  │
│ COMPATIBILITY WINDOW REQUIRED: YES — parallel audit writes during    │
│   migration transition                                               │
│ DUAL-WRITE REQUIRED: YES — RBAC writes go to both inids_rbac.db and │
│   OpsStore during migration; read from OpsStore after validation     │
│ ROLLBACK BOUNDARY: Restore rbac_manager.py and inids_rbac.db from   │
│   pre-migration backup; re-add RBAC auth checks                      │
│ WHAT BREAKS IF DONE TOO EARLY: No unified auth target to migrate into│
│ WHAT BREAKS IF DONE TOO LATE: Two disconnected auth sources persist  │
│   — system is still vulnerable despite Step 16                       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: RS256 + Revocation + No Expired Refresh (Step 18)          │
│ TIER: 1                                                              │
│ REQUIRES: Step 16 (JWT infrastructure must exist)                    │
│ ENABLES: Correct JWT security posture; Step 40 (test for HS256       │
│   token rejection)                                                   │
│ COMPATIBILITY WINDOW REQUIRED: YES — all existing HS256 tokens       │
│   invalid after cutover; forced re-auth for all active sessions      │
│ DUAL-WRITE REQUIRED: NO (hard cutover — not a gradual migration)     │
│ ROLLBACK BOUNDARY: Fall back to HS256 with non-placeholder key;      │
│   old tokens must be re-accepted                                     │
│ WHAT BREAKS IF DONE TOO EARLY: HS256 → RS256 cutover invalidates    │
│   all existing tokens — timing is a maintenance window decision      │
│ WHAT BREAKS IF DONE TOO LATE: HS256 with symmetric key remains the  │
│   JWT algorithm — C-19 vulnerability persists                        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Unify Rate Limiters (Step 19)                              │
│ TIER: 1                                                              │
│ REQUIRES: Step 16 (user identity in request context for per-user     │
│   limits via require_roles() enforcement point — see G-RATE-1        │
│   correction)                                                        │
│ ENABLES: Predictable, auditable rate enforcement system-wide         │
│ COMPATIBILITY WINDOW REQUIRED: YES — new limiter active before old   │
│   limiters removed; overlap window of 1 deployment cycle            │
│ DUAL-WRITE REQUIRED: NO — rate state is ephemeral; overlap is        │
│   additive (more restrictive, not incorrect)                         │
│ ROLLBACK BOUNDARY: Re-register old middleware if new limiter fails   │
│ WHAT BREAKS IF DONE TOO EARLY: Removing both old limiters before    │
│   new limiter is verified → no rate limiting for brief window        │
│ WHAT BREAKS IF DONE TOO LATE: Two conflicting limiters remain;       │
│   per-user limits remain unavailable                                 │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: DB-Level Idempotency for Prevention (Step 20)              │
│ TIER: 1                                                              │
│ REQUIRES: Step 13 (migration infrastructure for schema change)       │
│ ENABLES: Safe multi-instance prevention operation; TOCTOU race       │
│   removed                                                            │
│ COMPATIBILITY WINDOW REQUIRED: NO (schema change is additive — index │
│   added, not data modified)                                          │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: DROP INDEX uq_active_block; restore               │
│   has_active_block() TOCTOU check in execute()                       │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing                               │
│ WHAT BREAKS IF DONE TOO LATE: TOCTOU remains under concurrent        │
│   prevention operations from multiple instances                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Webhook Adapter State + Reconcile Fix (Step 21)            │
│ TIER: 1                                                              │
│ REQUIRES: Steps 11 (reconcile runs reliably after fail-closed        │
│   leader election), 13 (migration infra)                             │
│ ENABLES: Step 27 (nftables handle parsing), Step 37 (webhook TLS),  │
│   clean reconciliation for all adapter types                         │
│ COMPATIBILITY WINDOW REQUIRED: NO                                    │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Revert firewall_adapters.py and action_executor.py│
│ WHAT BREAKS IF DONE TOO EARLY: Nothing                               │
│ WHAT BREAKS IF DONE TOO LATE: Spurious DESYNCED records from webhook │
│   adapter trigger false remediation after every restart              │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SOLUTION: Sanitize Correlation ID + Audit Timestamps (Step 22)       │
│ TIER: 1                                                              │
│ REQUIRES: Step 17 (unified audit trail in OpsStore — the trail       │
│   being hardened must be the canonical one)                          │
│ ENABLES: Step 44 (ContextVar isolation builds on sanitized IDs),     │
│   forensically reliable audit trail                                  │
│ COMPATIBILITY WINDOW REQUIRED: NO                                    │
│ DUAL-WRITE REQUIRED: NO                                              │
│ ROLLBACK BOUNDARY: Revert correlation_tracing.py                     │
│ WHAT BREAKS IF DONE TOO EARLY: Nothing                               │
│ WHAT BREAKS IF DONE TOO LATE: Log injection continues as an active  │
│   attack path after real auth is in place (audit log has legal       │
│   standing post-Step 16)                                             │
└──────────────────────────────────────────────────────────────────────┘
FULL DEPENDENCY GRAPH — ORDERED SOLUTION GROUPS
Group 0 — No dependencies (can begin immediately, parallel within group):

Step 2 (mock TI removal)
Step 9 (PolicyConfig race)
Step 10 (AnomalyEngine race)
Step 13 (version-gated migrations)
Group 1 — Depends only on Group 0 baseline or is independent:

Step 1 (auth bypass removal) — no dependencies, but logically first in Group 0 timeline
Group 2 — Depends on Step 1 (parallel within group):

Step 3 (auth all undecorated routes)
Step 4 (break passwordless JWT)
Step 5 (Docker hardening)
Step 15 (sensor key role fix)
Group 3 — Depends on Step 5 (parallel within group):

Step 6 (model integrity)
Step 7 (pin requirements)
Group 4 — Depends on Step 9 (parallel within group):

Step 11 (leader election fail-closed)
Step 26 (prevention defaults + TI feed) [also depends on Step 2]
Group 5 — Depends on Steps 1+13 (parallel within group):

Step 12 (eliminate duplicate stores) [also depends on Steps 1-3]
Step 14 (register CORS) [depends on Steps 1-3]
Group 6 — Depends on Step 6:

Step 8 (ML graceful degradation)
Group 7 — Depends on Steps 11+13:

Step 21 (webhook adapter + reconcile)
Step 20 (DB-level idempotency) [depends only on Step 13]
Group 8 — Depends on Steps 1-4 + 13 + 9:

Step 16 (unified auth system)
Group 9 — Depends on Step 16:

Step 17 (RBAC migration)
Step 18 (RS256 + revocation)
Step 19 (unify rate limiters)
Group 10 — Depends on Step 17:

Step 22 (correlation ID sanitization)
Step 23 (delete dead code)
Group 11 — Depends on Step 3:

Step 24 (input sanitizer)
Group 12 — Depends on Step 24:

Step 25 (sanitize packet capture)
Group 13 — Depends on Steps 12+28:

Step 30 (alert pagination + retention)
Group 14 — Depends on Steps 12+13:

Step 28 (full alert IDs + dedup)
Step 29 (health check probe → read-only)
Group 15 — Depends on Step 21:

Step 27 (nftables JSON handle parsing)
Group 16 — No dependencies:

Step 31 (localhost whitelist)
Step 33 (fix hardcoded metrics)
Step 35 (rate counter deque)
Step 36 (honeypot confidence)
Step 38 (OpsStore public API) [depends on Step 13]
Group 17 — Depends on Step 14:

Step 32 (remove unsafe-inline CSP)
Group 18 — Depends on Step 21:

Step 37 (enforce webhook TLS)
Group 19 — Depends on Step 29:

Step 34 (remove OPS_DB_PATH from health)
Group 20 — Depends on all Phase A-E:

Step 39 (decompose app.py)
Step 40 (full regression test suite)
Group 21 — Depends on Steps 6+8+10:

Step 41 (ML model versioning + hot reload)
Group 22 — Depends on Step 7:

Step 42 (dependency CVE audit)
Group 23 — No dependencies:

Step 43 (suspicious label inversion)
Group 24 — Depends on Steps 22+40:

Step 44 (ContextVar greenlet isolation)
CRITICAL PATH

Step 1 (auth bypass)
  → Step 3 + Step 4 [PARALLEL]
    → Step 5 (Docker)
      → Step 6 (model integrity)
        → Step 8 (ML degradation)
  → Step 13 (migrations)
    → Step 9 (PolicyConfig) → Step 11 (leader election)
      → Step 16 (unified auth)
        → Step 17 (RBAC migration)
          → Step 22 (audit integrity)
        → Step 18 (RS256)
        → Step 19 (rate limiting)
      → Step 12 (store consolidation)
        → Step 28 (alert dedup)
          → Step 30 (retention)
      → Step 20 (DB idempotency)
      → Step 21 (webhook adapter)
        → Step 27 (nftables)
→ [Phase E completes] → Step 39 + Step 40 [PARALLEL]
Critical path length: 13 sequential decision points from Step 1 to Phase F entry. Zero float on: Steps 1, 5, 6, 13, 9, 11, 16, 17, 22.

PARALLEL TRACKS (independent of critical path once dependencies satisfied)
Track A: Step 2 → Step 26 (TI hardening — independent of auth)
Track B: Step 10 → Step 41 (ML versioning — independent of auth)
Track C: Step 7 → Step 42 (CVE audit — independent of auth)
Track D: Steps 31, 33, 35, 36 (Tier 3 fixes — no dependencies)
Track E: Step 3 → Step 24 → Step 25 (input hardening — depends on auth but not on RBAC migration)
Track F: Step 14 → Step 32 (CORS/CSP — depends on auth, independent of RBAC)
EXECUTION PHASE 3 — FILE-LEVEL IMPLEMENTATION TARGETING

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: A-01                                                      │
│ SOLUTION: Auth Bypass Removal (Step 1)                               │
│ PHASE: A                                                             │
│ TARGET MODULE(S):                                                    │
│   .env                                                               │
│   src/auth_service.py (AuthService._bypass_enabled())                │
│   src/settings.py (load_settings() validation block)                │
│   .gitignore, .dockerignore                                          │
│ CHANGE TYPE: MODIFY                                                  │
│ SCOPE OF CHANGE:                                                     │
│   .env: Change ALLOW_UNAUTHENTICATED=true → ALLOW_UNAUTHENTICATED=  │
│   false. Change SECRET_KEY=change-me-now → newly generated 32-byte  │
│   hex value. Change all three API key vars from placeholder values   │
│   to newly generated strong random keys (≥ 32 chars).               │
│   src/auth_service.py: Remove _bypass_enabled() method entirely.     │
│   In require_auth decorator, remove the conditional block that calls │
│   _bypass_enabled() and returns synthetic AuthContext. The decorator │
│   must always perform the key lookup.                                │
│   src/settings.py: In load_settings() or immediately after Settings │
│   construction, add:                                                 │
│     if os.environ.get("ALLOW_UNAUTHENTICATED","").lower() in         │
│     {"true","1","yes","on"}:                                         │
│       raise RuntimeError("Auth bypass is enabled. Cannot start.")   │
│   .gitignore: Add .env, *.pem, *.key, *.p12, *.pfx, models/*.pkl,  │
│   data/*.db                                                          │
│   .dockerignore: Add .env, .env.*, *.pem, *.key                     │
│ SIDE EFFECTS:                                                        │
│   All automated tests or scripts that set ALLOW_UNAUTHENTICATED=true │
│   will break. These must be audited and converted to use test API    │
│   keys before deployment.                                            │
│ COMPATIBILITY SHIM REQUIRED: NO                                      │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO                                          │
│ VALIDATION CHECKPOINT:                                               │
│   1. curl -X GET http://localhost:5000/api/alerts → must return 401  │
│   2. curl -X GET -H "X-API-Key: [new-admin-key]"                    │
│      http://localhost:5000/api/alerts → must return 200              │
│   3. Application starts with ALLOW_UNAUTHENTICATED not set → OK     │
│   4. Application starts with ALLOW_UNAUTHENTICATED=true → exits      │
│      immediately with RuntimeError                                   │
│   5. grep -r "ALLOW_UNAUTHENTICATED" . → must show only the startup │
│      check code, not any bypass-enabling usage                       │
│ ROLLBACK PROCEDURE:                                                  │
│   1. Revert .env to ALLOW_UNAUTHENTICATED=false (not true — the     │
│      goal is disabled bypass, not re-enabled bypass; only revert     │
│      API key values if the new keys are wrong)                       │
│   2. Revert src/auth_service.py to restore _bypass_enabled() IF the │
│      removal causes unexpected cascading failures; set               │
│      ALLOW_UNAUTHENTICATED=false to keep it non-functional           │
│   3. Revert src/settings.py startup check if it falsely fires        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: A-02                                                      │
│ SOLUTION: Mock Threat Intel Removal (Step 2)                         │
│ PHASE: A                                                             │
│ TARGET MODULE(S):                                                    │
│   web_app/app.py — load_threat_intel() function                      │
│   src/settings.py — Settings dataclass (add ti_feed_path field)     │
│ CHANGE TYPE: MODIFY                                                  │
│ SCOPE OF CHANGE:                                                     │
│   web_app/app.py: Replace the body of load_threat_intel() with:     │
│     feed_path = os.environ.get("INIDS_TI_FEED_PATH", "")           │
│     if not feed_path:                                                │
│       logger.warning("No threat intel feed configured.")             │
│       return []   # TI engine starts with no indicators             │
│     # Load from feed_path (CSV or JSON/STIX parsing)                │
│     # Validate: reject all RFC-1918 and loopback ranges             │
│     # Return list of valid indicators                                │
│   Remove all hardcoded mock_indicators / RFC-1918 IP lists from     │
│   the function body.                                                 │
│ SIDE EFFECTS:                                                        │
│   ThreatIntelEngine produces no matches until INIDS_TI_FEED_PATH is │
│   set. This is safe — false negatives from TI are preferable to     │
│   self-blocking. Log warning on every startup until feed is set.    │
│ COMPATIBILITY SHIM REQUIRED: NO                                      │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO                                          │
│ VALIDATION CHECKPOINT:                                               │
│   1. Start application without INIDS_TI_FEED_PATH → log shows       │
│      "No threat intel feed configured" at WARNING level              │
│   2. Submit a detection request with source_ip matching one of the  │
│      former RFC-1918 mock indicators → ThreatIntelEngine returns     │
│      PASS (not flagged)                                              │
│   3. grep -r "192.168\|10.0.0\|172.16" web_app/app.py →             │
│      no RFC-1918 IPs in load_threat_intel()                         │
│ ROLLBACK PROCEDURE:                                                  │
│   Restore the original load_threat_intel() body (with empty list    │
│   at minimum — do NOT restore RFC-1918 mock indicators)             │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: A-03                                                      │
│ SOLUTION: Auth All Undecorated Routes (Step 3)                       │
│ PHASE: A                                                             │
│ TARGET MODULE(S):                                                    │
│   web_app/app.py — approximately 25 route handler functions          │
│ CHANGE TYPE: MODIFY                                                  │
│ SCOPE OF CHANGE:                                                     │
│   Pre-work: Enumerate all routes via app.url_map.iter_rules() in a  │
│   script; identify those with no decorator. Categorize each by       │
│   required role per the recovery plan's priority order:             │
│     - /api/fp-suppressions POST/DELETE: @require_auth(roles=         │
│       ["analyst","admin"])                                           │
│     - Policy config endpoints: @require_auth(roles=["admin"])        │
│     - /api/forensics/timeline: @require_auth(roles=["analyst",       │
│       "admin"])                                                      │
│     - /api/escalation/*: @require_auth(roles=["analyst","operator",  │
│       "admin"])                                                      │
│     - Behavioral profiling, drift monitor, anomaly learning,         │
│       network topology: @require_auth(roles=["analyst","admin"])     │
│     - All others: @require_auth(roles=["viewer","analyst",           │
│       "operator","admin"])                                           │
│   Apply decorators to all 25 routes. For each route: confirm the    │
│   current @require_auth implementation accepts the roles list or    │
│   create a role-checking variant.                                    │
│   Add startup validation: validate_all_routes_have_auth_decorator   │
│   (app) — accepts @require_auth, @jwt_required, or @public_route.   │
│   Define PUBLIC_ROUTES = ["/health", "/api/health",                  │
│   "/api/auth/login", "/api/auth/refresh"].                          │
│ SIDE EFFECTS:                                                        │
│   All existing callers of the 25 newly-decorated routes must send   │
│   valid credentials. Audit ALL API consumers and scripts before      │
│   deploying.                                                         │
│ COMPATIBILITY SHIM REQUIRED: YES — @require_auth_legacy is accepted │
│   by startup validator during Phase A-B transition (see G-RBAC-1)   │
│   LIFETIME: Removed in Phase F along with INIDS_AUTH_COMPAT flag    │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO                                          │
│ VALIDATION CHECKPOINT:                                               │
│   For each of the 25 routes:                                         │
│   1. curl without credentials → 401 or 403                          │
│   2. curl with viewer-level key → correct response or 403 for       │
│      admin-only routes                                               │
│   3. Application starts successfully (startup validator passes)      │
│   4. test_all_routes_require_auth() passes (write this test in the  │
│      same PR per G-REG-4 resolution)                                 │
│ ROLLBACK PROCEDURE:                                                  │
│   git revert the decorator additions (targeted revert per route)     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: A-04                                                      │
│ SOLUTION: Break Passwordless JWT Login (Step 4)                      │
│ PHASE: A                                                             │
│ TARGET MODULE(S):                                                    │
│   web_app/app.py — api_auth_login route handler                      │
│   web_app/app.py — api_auth_refresh route handler                    │
│   src/auth_jwt.py — JWTManager.create_token()                        │
│ CHANGE TYPE: MODIFY                                                  │
│ SCOPE OF CHANGE:                                                     │
│   api_auth_login: Currently accepts any username and issues JWT.     │
│   Change to:                                                         │
│     1. Require X-API-Key header in the login request                │
│     2. Validate the API key against AuthService._principals          │
│     3. If key is valid: derive the principal's username/role from   │
│        the key lookup                                                │
│     4. Issue JWT for that principal only (not for arbitrary username)│
│     5. If key is invalid or absent: return 401 immediately           │
│   api_auth_refresh: Currently accepts expired tokens via             │
│   allow_expired=True.                                                │
│   Change to:                                                         │
│     1. Reject already-expired tokens outright (no allow_expired)    │
│     2. Accept tokens within 5-minute "refresh window" before expiry │
│     3. Reject tokens older than 5 minutes after expiry              │
│   src/auth_jwt.py: Remove allow_expired parameter from              │
│   JWTManager.verify_token() call chain. This is a permanent removal │
│   — the parameter enables an infinite token lifetime.                │
│ SIDE EFFECTS:                                                        │
│   Services that called /api/auth/login with just a username (no key)│
│   receive 401. All such services must be updated to send their       │
│   API key with the login request.                                    │
│ COMPATIBILITY SHIM REQUIRED: YES — Bridge behavior until Step 16.   │
│   LIFETIME: api_auth_login bridge removed when Step 16 deploys      │
│   the permanent credential verification flow.                        │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO                                          │
│ VALIDATION CHECKPOINT:                                               │
│   1. POST /api/auth/login {"username":"admin"} → 401 (no key)       │
│   2. POST /api/auth/login with X-API-Key:[valid-key] → 200 + JWT    │
│   3. POST /api/auth/login with X-API-Key:[invalid-key] → 401        │
│   4. POST /api/auth/refresh with expired-by-10min token → 401       │
│   5. POST /api/auth/refresh with about-to-expire (within 5min) → 200│

│ ROLLBACK PROCEDURE:                                                  │
│   Revert api_auth_login and api_auth_refresh handlers in app.py.    │
│   Revert allow_expired removal in auth_jwt.py.                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: A-05                                                      │
│ SOLUTION: Docker Deployment Hardening (Step 5)                       │
│ PHASE: A                                                             │
│ TARGET MODULE(S):                                                    │
│   Dockerfile (create or replace)                                     │
│   deploy/compose/docker-compose.yml (or docker-compose.yml)         │
│   .dockerignore (create)                                             │
│ CHANGE TYPE: MODIFY / CREATE                                         │
│ SCOPE OF CHANGE:                                                     │
│   Dockerfile — full replacement with hardened version:              │
│     FROM python:3.11-slim                                            │
│     RUN groupadd -r inids && useradd -r -g inids inids               │
│     WORKDIR /app                                                     │
│     COPY src/ ./src/                                                 │
│     COPY web_app/ ./web_app/                                         │
│     COPY rules/ ./rules/                                             │
│     COPY requirements.txt .                                          │
│     RUN pip install --no-cache-dir --require-hashes -r requirements.txt │
│     RUN apt-get update && apt-get install -y --no-install-recommends │
│         python3 && rm -rf /var/lib/apt/lists/*                       │
│     USER inids                                                       │
│     HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \    │
│       --retries=3 CMD python3 -c \                                   │
│       "import urllib.request; urllib.request.urlopen(                │
│       'http://localhost:5000/health', timeout=4)"                    │
│     EXPOSE 5000                                                      │
│     CMD ["gunicorn","--worker-class","eventlet","-w","1",            │
│          "-b","0.0.0.0:5000","web_app.app:app"]                      │
│   docker-compose.yml — remove env_file pointing to .env.example;   │
│   replace volume ../../:/app with:                                   │
│     - inids-data:/data                                               │
│     - inids-models:/models                                           │
│   Add: user: "inids:inids", read_only: true, tmpfs: [/tmp]          │
│   Add: deploy.resources.limits: {memory: 2g, cpus: "2.0"}           │
│   Add: secrets block and environment referencing _FILE variants      │
│   .dockerignore: .env, .env.*, *.pem, *.key, *.db, models/*.pkl     │
│ SIDE EFFECTS:                                                        │
│   All file paths in the application referencing /app/data or paths  │
│   within the source tree must be re-mapped to /data/... or          │
│   /models/.... Audit INIDS_DB_URL and all open() calls in app.py.   │
│   Container rebuild required — previous image must be tagged and     │
│   retained for rollback.                                             │
│ COMPATIBILITY SHIM REQUIRED: NO                                      │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO (data volume contains existing db files) │
│ VALIDATION CHECKPOINT:                                               │
│   1. docker inspect <image> | jq '.[0].Config.User' → "inids"       │
│   2. docker exec <container> id → shows inids UID, not 0            │
│   3. docker exec <container> ls /app/src → NOT exposed (no repo mnt)│
│   4. docker exec <container> ls /data → inids_ops.db present        │
│   5. Application starts, /health returns 200                         │
│   6. docker exec <container> ls ../../ → permission denied or empty  │
│ ROLLBACK PROCEDURE:                                                  │
│   docker pull <previous-image-tag>                                   │
│   Update docker-compose.yml to reference previous image tag          │
│   Restore old docker-compose.yml volume and user configuration       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: A-06                                                      │
│ SOLUTION: Model Integrity Verification (Step 6)                      │
│ PHASE: A                                                             │
│ TARGET MODULE(S):                                                    │
│   web_app/app.py — load_models()                                     │
│   src/detection/engines/anomaly_engine.py — model load path         │
│   scripts/generate_model_checksums.py (CREATE NEW)                  │
│   models/checksums.sha256 (CREATE NEW)                               │
│ CHANGE TYPE: MODIFY / CREATE                                         │
│ SCOPE OF CHANGE:                                                     │
│   scripts/generate_model_checksums.py — new script:                 │
│     import hashlib, pathlib, sys                                     │
│     models_dir = pathlib.Path("models")                              │
│     out = models_dir / "checksums.sha256"                            │
│     with out.open("w") as f:                                         │
│       for pkl in sorted(models_dir.glob("*.pkl")):                   │
│         digest = hashlib.sha256(pkl.read_bytes()).hexdigest()         │
│         f.write(f"{digest}  {pkl.name}\n")                           │
│   Run script against known-good models; commit checksums.sha256.    │
│   src/detection/ml_utils.py (CREATE NEW):                            │
│     def load_model_with_verification(path, expected_sha256):         │
│       digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()   │
│       if digest != expected_sha256:                                  │
│         raise SecurityError(f"Model {path} failed integrity check") │
│       return joblib.load(path)                                       │
│   web_app/app.py load_models(): Replace joblib.load(path) calls with│
│   load_model_with_verification(path, lookup_checksum(path)).         │
│   lookup_checksum() reads models/checksums.sha256, raises if missing.│
│   Check INIDS_MODEL_VERIFY env var: "strict"=raise on bad checksum  │
│   (default); "warn"=log CRITICAL and continue; "disabled"=skip check.│
│   anomaly_engine.py: Same replacement for its joblib.load() call.   │
│ SIDE EFFECTS:                                                        │
│   If model files have been modified since checksum generation,       │
│   startup fails. Must regenerate checksums against production models │
│   before deploying.                                                  │
│ COMPATIBILITY SHIM REQUIRED: YES — INIDS_MODEL_VERIFY=warn for      │
│   rollback only. LIFETIME: Remove warn option in Phase F after       │
│   full observation window confirms no legitimate checksum mismatches.│
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO                                          │
│ VALIDATION CHECKPOINT:                                               │
│   1. Start with correct checksums → application starts normally      │
│   2. Modify any .pkl file by one byte; restart → RuntimeError or     │
│      CRITICAL log depending on INIDS_MODEL_VERIFY setting            │
│   3. python scripts/generate_model_checksums.py → checksums.sha256  │
│      produced; revert .pkl modification; restart → starts normally   │
│   4. test_model_load_rejects_bad_checksum() passes (written in PR)  │
│ ROLLBACK PROCEDURE:                                                  │
│   Set INIDS_MODEL_VERIFY=warn (env var change only, no restart       │
│   needed if health check passes; restart required to apply).         │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: A-07                                                      │
│ SOLUTION: Pin Requirements With Hashes (Step 7)                      │
│ PHASE: A                                                             │
│ TARGET MODULE(S):                                                    │
│   requirements.txt (full replacement)                                │
│   requirements.in (CREATE — unpinned source of truth)               │
│ CHANGE TYPE: MODIFY / CREATE                                         │
│ SCOPE OF CHANGE:                                                     │
│   In a clean virtualenv: pip install pip-tools                       │
│   pip-compile --generate-hashes requirements.in -o requirements.txt  │
│   Review each pinned version for CVEs (target: cryptography ≥        │
│   42.0.8, numpy ≥ 1.26.4, Werkzeug ≥ 3.0.6).                       │
│   If upgrade required: update requirements.in, re-compile.           │
│   Dockerfile already uses --require-hashes (from A-05). Confirm     │
│   --require-hashes is present in the COPY+RUN layer.                │
│   Add CI step: pip-audit -r requirements.txt --fail-on-vuln          │
│ SIDE EFFECTS:                                                        │
│   If any package upgrade changes API behavior, application code      │
│   may break. Run full test suite after upgrades.                     │
│ COMPATIBILITY SHIM REQUIRED: NO                                      │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO                                          │
│ VALIDATION CHECKPOINT:                                               │
│   1. docker build --no-cache . → succeeds with --require-hashes      │
│   2. Application starts and passes /health check                     │
│   3. grep '\-\-hash=sha256' requirements.txt | wc -l > 0             │
│   4. pip-audit -r requirements.txt → no known vulnerabilities        │
│ ROLLBACK PROCEDURE:                                                  │
│   Revert requirements.txt to previous version; rebuild image.        │
└──────────────────────────────────────────────────────────────────────┘
Continuing with Phase B implementation targets (Steps 8–15):


┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: B-01                                                      │
│ SOLUTION: Version-Gated Schema Migrations (Step 13) — done FIRST    │
│ PHASE: B                                                             │
│ TARGET MODULE(S):                                                    │
│   src/ops_store.py — _migrate_actions_table(), _migrate_alerts_      │
│   table(), __init__(), _verify_schema_version()                      │
│ CHANGE TYPE: MODIFY                                                  │
│ SCOPE OF CHANGE:                                                     │
│   Add schema_version table bootstrap logic: on first run, if the    │
│   table doesn't exist, create it with version=0.                     │
│   Assign integer version numbers to all existing migration steps:    │
│     version 1: original table CREATE statements                      │
│     version 2: existing column additions currently in               │
│       _migrate_actions_table() and _migrate_alerts_table()           │
│   __init__(): replace "run all migrations" with:                     │
│     current_version = self._get_schema_version()                    │
│     for migration in MIGRATIONS:                                     │
│       if migration.version > current_version:                        │
│         migration.up(self._conn)                                     │
│         self._set_schema_version(migration.version)                  │
│   Remove all unconditional UPDATE queries from startup path.         │
│   Move data-normalization UPDATEs into their version-gated functions.│
│   Fix _verify_schema_version() to re-raise all exceptions that      │
│   indicate startup should abort (not just RuntimeError).            │
│ SIDE EFFECTS:                                                        │
│   Existing databases are treated as version 2. New databases start  │
│   at version 0 and run all migrations.                               │
│ COMPATIBILITY SHIM REQUIRED: NO                                      │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO (migrations are the mechanism)           │
│ VALIDATION CHECKPOINT:                                               │
│   1. Fresh database: run app → all migrations apply, schema_version=2│
│   2. Existing database: run app → no migrations re-run,             │
│      schema_version unchanged                                        │
│   3. Startup time measurement: ≥ 30% improvement on large tables    │
│   4. test_schema_migration_idempotent() passes                       │
│ ROLLBACK PROCEDURE:                                                  │
│   Keep old _migrate_actions_table() and _migrate_alerts_table() as  │
│   commented-out functions for one sprint. Re-enable by calling them  │
│   from __init__ (adds them back to the startup sequence).            │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: B-02                                                      │
│ SOLUTION: ML Inference Graceful Degradation (Step 8)                 │
│ PHASE: B                                                             │
│ TARGET MODULE(S):                                                    │
│   src/detection/engines/ml_engine.py — predict(), predict_proba()   │
│   src/detection/engines/anomaly_engine.py — evaluate(), is_ready()  │
│ CHANGE TYPE: MODIFY                                                  │
│ SCOPE OF CHANGE:                                                     │
│   ml_engine.py: Wrap both predict() and predict_proba() calls in    │
│   try/except. On any exception: log engine_id + error, return        │
│   EngineResult(verdict="unknown", confidence=0.0,                    │
│   metadata={"error": str(e), "fallback": True}). Do NOT re-raise.   │
│   anomaly_engine.py evaluate(): Guard against _model is None         │
│   (A-10 fix): at top of evaluate(), call model = self._get_model(); │
│   if model is None: return EngineResult(verdict="unknown", ...).     │
│   Wrap inference block in try/except as above.                       │
│   Add metric increment: ml_unknown_verdict_total counter on fallback.│
│ SIDE EFFECTS:                                                        │
│   Inference failures now silently return "unknown" verdicts. Monitor │
│   ml_unknown_rate metric — spike indicates model/feature problems.   │
│ COMPATIBILITY SHIM REQUIRED: NO                                      │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO                                          │
│ VALIDATION CHECKPOINT:                                               │
│   1. Inject a feature vector that causes predict_proba() to raise   │
│      → response returns 200 with verdict="unknown", fallback=True   │
│   2. ml_unknown_verdict_total metric increments on each fallback     │
│   3. Normal requests still return attack/normal verdicts correctly   │
│ ROLLBACK PROCEDURE:                                                  │
│   Revert ml_engine.py and anomaly_engine.py.                         │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: B-03                                                      │
│ SOLUTION: PolicyConfig Race Fix (Step 9)                             │
│ PHASE: B                                                             │
│ TARGET MODULE(S):                                                    │
│   src/prevention_service.py — PolicyConfig dataclass,                │
│   PolicyConfigManager (CREATE NEW in same file or src/policy.py)    │
│   All callers of self.policy.* in prevention_service.py,            │
│   src/ips/action_executor.py, src/risk_engine.py (audit all)        │
│ CHANGE TYPE: MODIFY                                                  │
│ SCOPE OF CHANGE:                                                     │
│   PolicyConfig: Add frozen=True to @dataclass decorator.             │
│   PolicyConfigManager: New class with _config: PolicyConfig and      │
│   _lock: RLock. Methods: get() → returns current frozen config;     │
│   update(**kwargs) → acquires lock, calls dataclasses.replace(),     │
│   stores new frozen instance, returns it.                            │
│   Replace all direct self.policy.X reads with:                       │
│     cfg = self.config_manager.get(); cfg.X                           │
│   Replace all set_policy() calls with:                              │
│     self.config_manager.update(**kwargs)                             │
│   grep -rn "\.policy\." -- src/ to find ALL call sites before       │
│   writing the PR.                                                    │
│ SIDE EFFECTS:                                                        │
│   Any code that modifies PolicyConfig fields directly fails at       │
│   runtime (frozen dataclass raises FrozenInstanceError). These are  │
│   bugs — fix them, do not unfreeze.                                  │
│ COMPATIBILITY SHIM REQUIRED: NO                                      │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO                                          │
│ VALIDATION CHECKPOINT:                                               │
│   1. test_policy_config_concurrent_update_and_read() passes (from   │
│      Phase 5 regression suite — write in same PR)                    │
│   2. Manual: attempt direct policy.mode = "block" → FrozenInstanceError│
│   3. config_manager.update(mode="block") → succeeds, new config     │
│      visible to concurrent readers                                   │
│ ROLLBACK PROCEDURE:                                                  │
│   Revert prevention_service.py (removes frozen=True, removes        │
│   PolicyConfigManager). Re-introduces race condition.                │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: B-04                                                      │
│ SOLUTION: AnomalyEngine Model Swap Race (Step 10)                    │
│ PHASE: B                                                             │
│ TARGET MODULE(S):                                                    │
│   src/detection/engines/anomaly_engine.py                            │
│ CHANGE TYPE: MODIFY                                                  │
│ SCOPE OF CHANGE:                                                     │
│   Remove direct self._model attribute. Add:                          │
│     self._model_ref = None        # atomic reference                 │
│     self._model_lock = threading.Lock()                              │
│   Add _set_model(model): acquires _model_lock, assigns self._model_ref│
│   Add _get_model(): returns self._model_ref (single read — GIL-safe) │
│   In fit(): replace self._model = trained_model with               │
│     self._set_model(trained_model)                                   │
│   In evaluate(): replace all uses of self._model with:              │
│     model = self._get_model()   # capture snapshot into local var    │
│     # use `model` exclusively — never reference self._model_ref again│
│   Add code comment: "CPython GIL makes single ref assignment atomic. │
│   Non-CPython: wrap in RLock."                                       │
│ SIDE EFFECTS:                                                        │
│   Any external code accessing engine._model directly breaks. Grep   │
│   for ._model outside anomaly_engine.py and fix those callers.       │
│ COMPATIBILITY SHIM REQUIRED: NO                                      │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO                                          │
│ VALIDATION CHECKPOINT:                                               │
│   1. test_anomaly_engine_concurrent_fit_and_evaluate() passes        │
│   2. No AttributeError on model access during concurrent fit/eval    │
│ ROLLBACK PROCEDURE:                                                  │
│   Revert anomaly_engine.py.                                          │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: B-05                                                      │
│ SOLUTION: Leader Election Fail-Closed (Step 11)                      │
│ PHASE: B                                                             │
│ TARGET MODULE(S):                                                    │
│   src/ha/leader_election.py — is_leader(), __init__()                │
│ CHANGE TYPE: MODIFY                                                  │
│ SCOPE OF CHANGE:                                                     │
│   is_leader(): In the except RedisError block, change return True   │
│   to return False. Add:                                              │
│     self.logger.error("leader_election_redis_failure: assuming NOT  │
│     leader — prevention scheduler paused")                           │
│     self._emit_metric("leader_election_state", 0)                    │
│   On success: self._emit_metric("leader_election_state", 1)         │
│   Fix integer division: replace self._ttl // 3 with               │
│     max(1, int(self._ttl / 3.0))                                     │
│   Add INIDS_REDIS_REQUIRED env var support: if "false", retain       │
│   always-leader behavior (single-instance deployments without Redis).│
│     if not self._redis_required:                                     │
│       return True   # single-instance mode                           │
│ SIDE EFFECTS:                                                        │
│   Single-instance deployments without Redis must set                 │
│   INIDS_REDIS_REQUIRED=false or prevention scheduler stops.          │
│   This must be documented in the deployment runbook before deploy.  │
│ COMPATIBILITY SHIM REQUIRED: YES — INIDS_REDIS_REQUIRED=false for   │
│   single-instance mode. LIFETIME: Permanent operational config.      │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO                                          │
│ VALIDATION CHECKPOINT:                                               │
│   1. With Redis running: is_leader() returns True for one instance, │
│      False for others                                                │
│   2. With Redis stopped: is_leader() returns False for all instances │
│      (INIDS_REDIS_REQUIRED unset or true)                            │
│   3. leader_election_state metric = 0 when Redis unavailable         │
│   4. With INIDS_REDIS_REQUIRED=false: is_leader() returns True      │
│      regardless of Redis availability                                │
│ ROLLBACK PROCEDURE:                                                  │
│   Revert leader_election.py. Re-introduces fail-open behavior.       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: B-06                                                      │
│ SOLUTION: Eliminate Duplicate Alert/Action Stores (Step 12)          │
│ PHASE: B                                                             │
│ TARGET MODULE(S):                                                    │
│   src/detection_service.py — InMemoryAlertStore usage                │
│   src/prevention_service.py — InMemoryPreventionStore                │
│   web_app/app.py — all routes reading from in-memory stores         │
│   src/ops_store.py — list_alerts() (add offset param)               │
│ CHANGE TYPE: MODIFY / DEPRECATE                                      │
│ SCOPE OF CHANGE:                                                     │
│   Deployment sub-sequence (3 sub-deploys, not 1):                   │
│   SUB-DEPLOY 1 (redirect writes):                                    │
│     DetectionService: add OpsStore parameter; write alerts to both   │
│     InMemoryAlertStore AND OpsStore simultaneously.                  │
│     ActionExecutor already writes to OpsStore — confirm no remaining │
│     direct writes to InMemoryPreventionStore anywhere in codebase.  │
│   Observe for 1 hour: both stores agree.                             │
│   SUB-DEPLOY 2 (switch reads):                                       │
│     Update all API routes that read from InMemoryAlertStore to read  │
│     from OpsStore instead.                                           │
│     OpsStore.list_alerts() gets offset param (I-05 fix):            │
│       list_alerts(limit=50, offset=0, ...) → (rows, total_count)    │
│     Observe dashboards for 30 min — no data discrepancy.            │
│   SUB-DEPLOY 3 (remove in-memory stores):                            │
│     drain_to_ops_store(): flush remaining in-memory alerts to        │
│     OpsStore; log count of drained records; fail if drain_failures>0.│
│     Remove InMemoryAlertStore from DetectionService constructor.     │
│     Remove InMemoryPreventionStore from PreventionService.           │
│ SIDE EFFECTS:                                                        │
│   All code paths reading from alert_store or prevention_store vars   │
│   must be updated to use OpsStore. Full grep required.               │
│ COMPATIBILITY SHIM REQUIRED: YES (sub-deploy 1: dual-write).         │
│   LIFETIME: Removed in sub-deploy 3.                                 │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: YES — drain_to_ops_store() drains           │
│   InMemoryAlertStore to OpsStore. No data deleted from source until  │
│   drain verification passes.                                         │
│ VALIDATION CHECKPOINT:                                               │
│   1. After sub-deploy 2: GET /api/alerts returns same count from    │
│      dashboard as direct OpsStore query                              │
│   2. After sub-deploy 3: no InMemoryAlertStore or                   │
│      InMemoryPreventionStore references remain in codebase           │
│   3. OpsStore write latency monitored — no spike above baseline      │
│ ROLLBACK PROCEDURE:                                                  │
│   Sub-deploy 1: Remove dual-write, revert to in-memory only.         │
│   Sub-deploy 2: Re-route reads back to in-memory store.              │
│   Sub-deploy 3: Re-inject InMemoryAlertStore as secondary write      │
│   target; data already drained to OpsStore is preserved.            │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: B-07                                                      │
│ SOLUTION: Register CORS Middleware (Step 14)                         │
│ PHASE: B                                                             │
│ TARGET MODULE(S):                                                    │
│   web_app/app.py — application initialization                        │
│   src/middleware.py — CORSMiddleware (already exists, not registered)│
│ CHANGE TYPE: MODIFY                                                  │
│ SCOPE OF CHANGE:                                                     │
│   Determine the correct allowed origins (dashboard domain, not *).  │
│   INIDS_CORS_ORIGINS env var: comma-separated list of allowed origins│
│   In app.py create_app(): register CORSMiddleware:                   │
│     cors = CORSMiddleware(                                           │
│       allowed_origins=settings.cors_origins,                         │
│       allowed_methods=["GET","POST","PUT","DELETE","OPTIONS"],        │
│       allowed_headers=["Authorization","Content-Type","X-API-Key",  │
│                         "X-Correlation-ID"],                         │
│       allow_credentials=True                                         │
│     )                                                                │
│     app.before_request(cors.handle_preflight)                        │
│     app.after_request(cors.add_headers)                              │
│   Alternative: replace CORSMiddleware with flask-cors if the         │
│   existing implementation is incomplete.                             │
│   CSP unsafe-inline: DO NOT remove in this step — deferred to        │
│   Step 32 (Phase E) after template audit.                            │
│ SIDE EFFECTS:                                                        │
│   Cross-origin requests from origins not in allowed list will receive│
│   CORS rejection (403 or missing headers). Dashboard origin must be  │
│   in the allowed list before deploying.                              │
│ COMPATIBILITY SHIM REQUIRED: NO                                      │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO                                          │
│ VALIDATION CHECKPOINT:                                               │
│   1. OPTIONS /api/alerts from dashboard origin → 200 with correct   │
│      Access-Control-Allow-Origin header                              │
│   2. OPTIONS /api/alerts from unauthorized origin → CORS rejection   │
│   3. Existing API tests pass (CORS headers additive, not breaking)   │
│ ROLLBACK PROCEDURE:                                                  │
│   Remove before_request/after_request registrations (CORS reverts   │
│   to never-enforced state).                                          │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: B-08                                                      │
│ SOLUTION: Fix Sensor Key Role (Step 15)                              │
│ PHASE: B                                                             │
│ TARGET MODULE(S):                                                    │
│   src/auth_service.py — AuthService.__init__() or key config block  │
│ CHANGE TYPE: MODIFY                                                  │
│ SCOPE OF CHANGE:                                                     │
│   In the _principals dict construction: change the sensor key entry  │
│   from Principal(role="analyst", ...) to Principal(role="sensor",...).│
│   Define "sensor" role permissions:                                  │
│     SENSOR_ALLOWED_ENDPOINTS = {"/api/detect", "/api/stream"}        │
│   Ensure @require_auth on detection submission endpoints accepts     │
│   "sensor" in the allowed_roles list.                                │
│   Ensure all analyst-only endpoints reject "sensor" role.            │
│ SIDE EFFECTS:                                                        │
│   Any sensor node that used the sensor API key to call analyst-level │
│   endpoints (reads, config queries) will receive 403. Audit all      │
│   sensor integrations before deployment.                             │
│ COMPATIBILITY SHIM REQUIRED: NO                                      │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO                                          │
│ VALIDATION CHECKPOINT:                                               │
│   1. POST /api/detect with sensor key → 200                          │
│   2. GET /api/alerts with sensor key → 403                           │
│   3. POST /api/fp-suppressions with sensor key → 403                 │
│ ROLLBACK PROCEDURE:                                                  │
│   Revert auth_service.py sensor key role assignment.                 │
└──────────────────────────────────────────────────────────────────────┘
Continuing with Phase C key implementation targets (Steps 16–22). Truncating to highest-criticality targets for brevity while preserving all required fields:


┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: C-01                                                      │
│ SOLUTION: Unified Authentication System (Step 16)                    │
│ PHASE: C                                                             │
│ TARGET MODULE(S):                                                    │
│   src/auth/ (new package — CREATE):                                  │
│     src/auth/__init__.py                                             │
│     src/auth/models.py (User, APIKey, RevokedToken dataclasses)     │
│     src/auth/auth_service.py (UnifiedAuthService class)              │
│     src/auth/jwt_manager.py (RS256 JWTManager)                      │
│     src/auth/decorators.py (require_roles, public_route)            │
│     src/auth/validators.py (validate_config_at_startup — full impl) │
│   src/ops_store.py — add users, api_keys, revoked_tokens migrations │
│   web_app/app.py — replace require_auth/jwt_required with            │
│     require_roles() everywhere; add INIDS_AUTH_COMPAT support        │
│ CHANGE TYPE: CREATE / MODIFY                                         │
│ SCOPE OF CHANGE:                                                     │
│   src/auth/models.py: Define User, APIKey, RevokedToken as           │
│   frozen dataclasses matching the schema in Solution 1.              │
│   src/auth/auth_service.py UnifiedAuthService:                       │
│     authenticate_api_key(key: str) → AuthContext | None:            │
│       sha256(key) → lookup in api_keys table → return AuthContext    │
│     authenticate_jwt(token: str) → AuthContext | None:              │
│       RS256 verify → check jti in revoked_tokens → return AuthContext│
│     create_token(user_id, roles) → signed JWT with jti, 1hr expiry  │
│     revoke_token(jti) → INSERT into revoked_tokens                   │
│   src/auth/decorators.py require_roles(*roles):                      │
│     Extract Bearer token or X-API-Key from request                   │
│     Try JWT path, then API key path                                  │
│     Produce AuthContext or raise AuthError (returns 401/403)         │
│     Check AuthContext.roles ∩ required_roles — if empty: 403        │
│     Set flask.g.auth = auth_context                                  │
│     Apply per-user rate limit (Tier 2 — G-RATE-1 correction)        │
│     Log authorization decision to OpsStore.audits                    │
│   INIDS_AUTH_COMPAT=true: after new auth fails, try old              │
│   auth_service.py / auth_jwt.py path as fallback. Log at WARNING.   │
│   schema migrations: version 3 adds users, api_keys,                │
│   revoked_tokens tables with index on revoked_tokens(jti) (G-AUTH-1)│
│   and index on api_keys(key_hash).                                  │
│   Pre-migration sub-step: create service accounts for all existing  │
│   API key holders in users table; hash existing keys into api_keys. │
│ SIDE EFFECTS:                                                        │
│   All routes using @require_auth or @jwt_required must be updated   │
│   to @require_roles(). This is a cross-file change. The startup      │
│   validator rejects any route with legacy decorators if             │
│   INIDS_AUTH_COMPAT=false (Phase F condition).                       │
│ COMPATIBILITY SHIM REQUIRED: YES — INIDS_AUTH_COMPAT flag.           │
│   LIFETIME: Removed in Phase F Step F-AUTH-REMOVE (see below).      │
│ DUAL-WRITE: YES (both auth systems validate during compat window;   │
│   new system logs decisions separately for audit comparison)         │
│ DATA MIGRATION REQUIRED: YES — users and api_keys tables populated  │
│   before enabling UnifiedAuthService. Source: env-var API keys and  │
│   existing RBAC user records. No data deleted from old sources yet.  │
│ VALIDATION CHECKPOINT:                                               │
│   1. POST /api/auth/login with valid credential → RS256 JWT issued  │
│   2. Use RS256 JWT on @require_roles("admin") endpoint → 200         │
│   3. Use forged HS256 JWT → 401 (verify RS256-only enforcement)      │
│   4. Use revoked jti → 401                                          │
│   5. INIDS_AUTH_COMPAT=true: old API key accepted as fallback        │
│   6. All existing integration tests pass under compat mode           │
│ ROLLBACK PROCEDURE:                                                  │
│   Set INIDS_AUTH_COMPAT=true — old system takes over immediately.    │
│   New users/api_keys/revoked_tokens tables remain (additive, safe). │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CHANGE ID: C-02                                                      │
│ SOLUTION: DB-Level Idempotency for Prevention (Step 20)              │
│ PHASE: C                                                             │
│ TARGET MODULE(S):                                                    │
│   src/ops_store.py — actions table schema, save_action()             │
│   src/ips/action_executor.py — execute()                             │
│ CHANGE TYPE: MODIFY                                                  │
│ SCOPE OF CHANGE:                                                     │
│   OpsStore schema migration version 4:                               │
│     CREATE UNIQUE INDEX IF NOT EXISTS uq_active_block                │
│     ON actions(target)                                               │
│     WHERE lower(status) IN ('active','enforced','executed')          │
│     AND lower(action_type) IN ('block','temp_block','rate_limit');   │
│   Verify SQLite version ≥ 3.8.9 before applying (partial index       │
│   support). For PostgreSQL: same syntax, no version constraint.      │
│   save_action(): catch IntegrityError from duplicate INSERT;         │
│   execute SELECT to return the existing record instead of inserting. │
│   action_executor.py execute(): remove the has_active_block()        │
│   read-check-write pattern entirely. Let the INSERT in save_action() │
│   be the idempotency gate.                                           │
│ SIDE EFFECTS:                                                        │
│   Any attempt to insert a duplicate active block raises IntegrityError│
│   at the database level — not in application code. The exception     │
│   must be caught in save_action(), not propagate to execute().       │
│ COMPATIBILITY SHIM REQUIRED: NO                                      │
│ TEMPORARY ADAPTER REQUIRED: NO                                       │
│ DATA MIGRATION REQUIRED: NO (index addition only)                    │
│ VALIDATION CHECKPOINT:                                               │
│   1. python3 -c "import sqlite3; print(sqlite3.sqlite_version)"      │
│      → version ≥ 3.8.9                                               │
│   2. Attempt to INSERT two block actions for same target/status →    │
│      second INSERT returns existing record, no exception raised to   │
│      caller                                                          │
│   3. INDEX exists: SELECT name FROM sqlite_master WHERE              │
│      type='index' AND name='uq_active_block' → returns row           │
│ ROLLBACK PROCEDURE:                                                  │
│   DROP INDEX uq_active_block;                                        │
│   Restore has_active_block() read-check in execute().                │
└──────────────────────────────────────────────────────────────────────┘
Phase C implementation targets for Steps 17, 18, 19, 21, 22 follow the same format. Providing the complete target specifications for the remaining Phase C steps in condensed form:

C-03 (Step 17 — RBAC Migration): Target: src/rbac_manager.py, inids_rbac.db, src/ops_store.py. Migration script: export inids_rbac.db users/roles/permissions → upsert into OpsStore users/api_keys tables using INSERT OR IGNORE. Final sync step: diff post-migration records against inids_rbac.db delta. Validate: all RBAC queries return identical results from OpsStore as from inids_rbac.db. Rollback: restore inids_rbac.db from pre-migration backup; re-enable rbac_manager.py route checks.

C-04 (Step 18 — RS256 + Revocation): Target: src/auth/jwt_manager.py. Generate RSA-2048 keypair; store private key in Docker secret, public key injected as INIDS_JWT_PUBLIC_KEY. Remove allow_expired path. Add revoke endpoint POST /api/auth/revoke. Add daily cleanup: DELETE FROM revoked_tokens WHERE expires_at < NOW(). Cutover window: maintenance window required; all existing HS256 tokens invalid after deploy. Rollback: re-enable HS256 with new non-placeholder key.

C-05 (Step 19 — Rate Limiter Unification): Target: src/middleware.py (remove RateLimitMiddleware), web_app/app.py (remove InMemoryRateLimiter before_request hook), src/rate_limiter.py (CREATE — unified RateLimiter class with Redis-backed store). Per G-RATE-1 correction: before_request hook handles global IP limit (1000/min) only; require_roles() adds per-user check (200/min) after auth. Deployment order: (1) deploy new limiter, (2) verify it fires, (3) remove RateLimitMiddleware, (4) remove InMemoryRateLimiter.

C-06 (Step 21 — Webhook Adapter + Reconcile): Target: src/firewall_adapters.py, src/ips/action_executor.py. Add supports_rule_query = True/False class attribute to each adapter. reconcile(): check adapter.supports_rule_query before comparing. Remove blocked_targets dict from WebhookFirewallAdapter. WebhookFirewallAdapter.list_rules() raises NotImplementedError. Rollback: revert both files.

C-07 (Step 22 — Correlation ID Sanitization): Target: src/correlation_tracing.py, src/middleware.py (AuditLogMiddleware). Sanitize X-Correlation-ID: strip non-printable chars, reject \n/\r/null, truncate to 64 chars, replace invalid with server-generated UUID. Replace datetime.utcnow() with datetime.now(timezone.utc) throughout AuditLogMiddleware. Fix rbac_manager.py AuditLog.id to use uuid.uuid4().hex.

EXECUTION PHASE 4 — COMPATIBILITY & COEXISTENCE STRATEGY
1. PARALLEL OPERATION WINDOWS
Authentication System (Steps 1–4 bridge → Step 16 replacement → Step 18 RS256 cutover):

Period	Old System	New System	Resolution
Phase A (Steps 1–4)	Bridge: env-var keys + API-key-gated JWT	Not deployed	Old system tightened in place
Phase C entry (Step 16)	Active via INIDS_AUTH_COMPAT=true	Active — logs decisions	New auth primary; old as fallback
Compat window (2 sprints)	Fallback for legacy callers	Primary for all verified callers	Divergence: if new rejects, old accepts → log WARNING + alert ops
Phase C exit (Step 18)	Disabled (INIDS_AUTH_COMPAT=false)	Sole system, RS256 only	Hard cutover; maintenance window
Phase F (Step F-AUTH-REMOVE)	auth_service.py + auth_jwt.py DELETED	Sole permanent system	Source files removed
Start event: Step 16 deployed with INIDS_AUTH_COMPAT=true
Observation window: 2 sprints (approximately 4 weeks) running both systems
Divergence handling: If new system rejects a request that old system would accept: log at WARNING with user_id + endpoint; do NOT silently fall through. Alert ops when this rate exceeds 0.1% of requests
Cutover trigger: Zero divergence warnings for 48 consecutive hours AND all integration tests pass AND all API consumers confirmed updated
Prevention Orchestration (InMemoryPreventionStore elimination):

Start event: Sub-deploy 1 of Step 12 (dual-write begins)
Observation window: 1 hour between sub-deploys
Divergence handling: OpsStore action count vs InMemoryPreventionStore count compared on each request cycle; alert if delta > 0
Cutover trigger: Zero divergence between stores for full observation window
2. DUAL-WRITE WINDOWS
Alert Store Consolidation (Step 12):

Phase	Write Target	Read Target	Duration
1 (current state)	InMemoryAlertStore + OpsStore (via EventBus)	InMemoryAlertStore	Until sub-deploy 1
2 (sub-deploy 1)	Both stores explicitly; OpsStore direct write confirmed	InMemoryAlertStore	1-hour observation
3 (sub-deploy 2)	Both stores	OpsStore (API routes updated)	30-min observation
4 (sub-deploy 3)	OpsStore only (InMemoryAlertStore removed)	OpsStore	Permanent
Rollback path for each phase transition: Reversible independently. Reverting Phase 4 re-injects InMemoryAlertStore as secondary write target. Data drained to OpsStore in Phase 3 is preserved.

RBAC Data Migration (Step 17):

Phase	Write Target	Read Target	Duration
1	inids_rbac.db only	inids_rbac.db	Until migration starts
2	Both (rbac_manager.py writes to inids_rbac.db; UnifiedAuthService writes to OpsStore for new records)	OpsStore (after bulk migration confirmed complete)	1 sprint
3	OpsStore only	OpsStore	Permanent
Transition validation gate: Run reconciliation script comparing inids_rbac.db and OpsStore RBAC records — must return zero discrepancies before Phase 3.

3. TOKEN / SESSION COMPATIBILITY WINDOWS
HS256 → RS256 Migration (Steps 16–18):

Phase C entry: New tokens issued as RS256 (Step 16). HS256 tokens from old system still accepted during INIDS_AUTH_COMPAT window because old auth_jwt.py validates them
Maximum old token lifetime: 1 hour (token expiry is non-negotiable per Solution 1). Therefore the minimum compatibility window for tokens already in flight is 1 hour
Forced re-authentication: Required at Step 18 deployment (hard cutover). All active sessions invalidated. Maintenance window must be announced to users at least 24 hours in advance
Refresh token handling: api_auth_refresh accept-expired path is removed in Step 4. By Step 18, no refresh token older than 1 hour can be valid. No refresh compatibility window needed
INIDS_AUTH_COMPAT=false deployment: Plan for lowest-traffic window. All API consumers must complete migration before this flag flips
4. API CONTRACT STABILITY
Endpoint	Change	Client Update Required Before Removal	Deprecated Endpoint Retained Until
POST /api/auth/login	Requires X-API-Key header (Step 4 bridge); later requires username+password credentials (Step 16)	All API consumers must update before Step 18 cutover	N/A — endpoint retained but signature changes
POST /api/auth/refresh	Rejects expired tokens (Step 4)	Consumers relying on infinite refresh must implement re-auth before Step 4	N/A
GET /api/alerts (from InMemoryAlertStore)	Now reads from OpsStore; adds pagination (limit/offset)	All dashboard consumers must handle paginated response	During sub-deploy 2 observation window only
All 25 unauthenticated routes (Step 3)	Require credentials	All consumers must add credentials before Step 3 deploys	N/A — no deprecated version; hard cutover
Webhook adapter list_rules()	Raises NotImplementedError (Step 21)	Internal only — reconcile() updated in same step	N/A
POST /api/fp-suppressions	Auth added (Step 3); role required: analyst/admin	Any automated tools must be updated before Step 3	N/A
EXECUTION PHASE 5 — SURGICAL REMEDIATION ROADMAP
PHASE A — EMERGENCY STABILIZATION
Timeline: 0–72 hours | Tier 0 items only

Objective: Close active attack surfaces. No new systems, no refactoring, no architectural improvements. The following interventions are safe to apply without full architectural context because each is targeted, independently reversible, and addresses an active exploitation vector, not a theoretical future risk.

PHASE A — STEP 1: Disable Authentication Bypass

Why safe without full architectural context: Changing ALLOW_UNAUTHENTICATED=false is a configuration change. The auth decorator code already exists and is correct when the bypass is disabled. Removing _bypass_enabled() eliminates one code path; it does not restructure any system.

Risk of masking a failure: None. Disabling the bypass surfaces legitimate authentication failures that were previously hidden. Monitor /api/* for 401 response rate increase — this is expected and correct.

Validation confirming containment:


curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/alerts
# Must return 401 — not 200
curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: [NEW_ADMIN_KEY]" \
  http://localhost:5000/api/alerts
# Must return 200
grep -r "ALLOW_UNAUTHENTICATED" src/ | grep -v "RuntimeError"
# Must return empty — no code enables the bypass
Go/no-go before Step 2: Both curl checks above pass. Log shows no auth bypass warnings. APPLICATION DOES NOT START with ALLOW_UNAUTHENTICATED=true (verified in staging).

PHASE A — STEP 2: Remove Mock Threat Intelligence

Why safe without full architectural context: Replacing hardcoded mock indicators with an empty list cannot worsen detection (it reduces false positives). The TI engine still functions; it simply produces no matches until a real feed is configured.

Risk of masking a failure: None. If TI was the only detection source that was catching real threats, removing mock indicators reduces detection coverage — but mock RFC-1918 indicators cannot catch real external threats. Net effect: no change to real detection capability.

Validation confirming containment:


# Send detection request with source_ip=10.0.0.1 (formerly in mock TI)
curl -s -X POST -H "X-API-Key: [KEY]" \
  -d '{"source_ip": "10.0.0.1", ...}' \
  http://localhost:5000/api/detect
# ThreatIntelEngine result must be "pass" / "normal" — not flagged
grep "No threat intel feed configured" /var/log/inids/app.log
# Must appear at WARNING level
Go/no-go before Step 3: TI engine confirms zero matches for all RFC-1918 addresses.

PHASE A — STEPS 3 & 4: Auth All Routes + Break Passwordless JWT (deploy in parallel after Step 1)

Step 3 — Why safe: Adding decorators to routes that never had them cannot remove functionality that worked. Any breakage indicates a caller that was relying on unauthenticated access — which is the vulnerability being closed.

Step 3 — Risk of masking a failure: The FP suppression endpoint gains auth. Any attacker currently using it to suppress their IP is now blocked. This may cause a brief spike in alerts for previously-suppressed IPs — this is correct behavior surfacing hidden attack activity.

Step 4 — Why safe: The bridge fix (require API key for JWT issuance) is strictly more restrictive than the current behavior. No legitimate caller is worse off if they already hold a valid API key.

Step 3 + Step 4 combined validation:


# AC-1 attack chain must now fail at Step 1:
curl -s -o /dev/null -w "%{http_code}" \
  -X POST -d '{"username":"admin"}' http://localhost:5000/api/auth/login
# Must return 401 — no API key provided

# AC-4 attack chain must now fail:
curl -s -o /dev/null -w "%{http_code}" \
  -X POST -d '{"source_ip":"attacker_ip"}' \
  http://localhost:5000/api/fp-suppressions
# Must return 401 — no credentials
Go/no-go before Step 5: AC-1 and AC-4 attack chains are closed. test_auth_bypass_disabled(), test_fp_suppression_requires_auth(), test_all_routes_require_auth() all pass in CI.

PHASE A — STEP 5: Docker Deployment Hardening

Why safe: Container rebuild with non-root user and restricted volume mount does not affect application logic. The application reads from /data and /models — both now explicitly mounted.

Risk of masking a failure: Application paths must be verified. If any path was hardcoded to /app/data or /app/models, the container silently fails to find its data. Run a full startup verification in staging BEFORE production deployment.

Validation:


docker inspect inids | jq '.[0].HostConfig.Binds'
# Must show /data:/data and /models:/models — NOT ../../:/app
docker exec inids id
# Must show uid= for inids user, NOT root (uid=0)
Go/no-go before Steps 6+7: Container starts, /health returns 200, uid is non-root, source tree is NOT mounted.

PHASE A — STEPS 6 & 7: Model Integrity + Hash Requirements (deploy in parallel after Step 5)

Step 6 validation:


# Corrupt a model file by one byte; attempt restart
# Application must fail to start with SecurityError logged
# Restore model; restart — application starts normally
python scripts/verify_model_checksums.py
# Must exit 0 — all checksums verified
Step 7 validation:


grep '\-\-hash=sha256' requirements.txt | wc -l
# Must be > 0 — all packages have hash pins
docker build --no-cache . 2>&1 | tail -5
# Must succeed — no hash verification failures
Go/no-go confirming Phase A complete before Phase B begins:

 401 returned for all unauthenticated requests to protected routes
 API key rotation complete, old placeholder keys rejected
 Mock threat intel removed, TI engine starts with empty indicators
 JWT login requires valid API key
 Container runs as non-root with restricted volume mounts
 Model checksums verified at startup; bad checksum causes startup failure
 requirements.txt contains hash pins for all packages
 Attack chains AC-1, AC-2, AC-3, AC-4 are closed (validated by test suite in CI)
PHASE B — ARCHITECTURAL FOUNDATIONS
Timeline: Week 1–2 | Tier 1 prerequisites

Objective: Build the structural substrate that all Phase C and later work depends on. Phase C may not begin until every Phase B step is validated.

Phase B Step sequence with validation gates:

B-GATE-1 (before any Phase B work): Confirm Phase A go/no-go checklist above is complete. All 8 items must be checked. No Phase B step begins until this gate passes.

Step 13 (version-gated migrations) — deploy first:

Go/no-go gate: Schema_version table exists; migrations run exactly once on fresh DB; no migration re-runs on already-migrated DB; startup time unchanged or improved.
Steps 9 + 10 (concurrency fixes) — deploy in parallel after Step 13 gate passes:

Go/no-go gate: test_policy_config_concurrent_update_and_read() passes; test_anomaly_engine_concurrent_fit_and_evaluate() passes.
Step 11 (leader election fail-closed) — deploy after Step 9 gate passes:

Requires Step 9 because prevention scheduler pause must be clean (no partial PolicyConfig state).
Go/no-go gate: With Redis stopped, leader_election_state metric = 0; prevention scheduler confirmed paused; alert on Redis failure fires within 60 seconds.
Step 8 (ML graceful degradation) — deploy in parallel with Steps 9+10:

Go/no-go gate: Injected inference failure returns 200 with verdict=unknown; no 500 responses on ML failure; ml_unknown_verdict_total metric increments.
Step 12 (eliminate duplicate stores) — deploy after Step 13 gate passes:

Three sub-deploys with observation windows as specified in Change ID B-06.
Go/no-go gate (final): Zero references to InMemoryAlertStore or InMemoryPreventionStore in codebase; OpsStore write latency at baseline.
Steps 14 + 15 (CORS + sensor key) — deploy in parallel after Steps 1-3 gate passes:

Go/no-go gate Step 14: CORS headers present on responses; unauthorized origins rejected.
Go/no-go gate Step 15: Sensor key returns 403 on GET /api/alerts; returns 200 on POST /api/detect.
B-GATE-FINAL (before Phase C begins):

 All 8 Phase B steps validated and closed
 No unconditional startup migrations running (schema_version verified)
 PolicyConfig concurrent test passing in CI
 AnomalyEngine concurrent test passing in CI
 Leader election fail-closed confirmed in staging
 Duplicate stores eliminated; OpsStore is sole alert and action backend
 CORS enforced; sensor key restricted to sensor role
PHASE C — SYSTEM UNIFICATION
Timeline: Week 2–4 | Tier 1 consolidation

Objective: Replace duplicated, conflicting, and fragmented systems. All coexistence windows begin here.

C-GATE-1 (before Phase C): B-GATE-FINAL must be complete. Zero open gaps from Phase B validation.

Step 16 (Unified Auth) — deploy with INIDS_AUTH_COMPAT=true:

Coexistence strategy: Both auth systems active. New system is primary; old system handles requests that fail new auth. Divergence metric (new-rejects / old-accepts) must remain below 0.1% of traffic.

Cutover trigger: Zero divergence for 48 hours AND all consumer integrations confirmed updated.

Rollback boundary: INIDS_AUTH_COMPAT=true (old system takes over immediately; no data loss — users/api_keys tables are additive).

Step 17 (RBAC migration) — after Step 16 cutover trigger passes:

Coexistence: RBAC writes dual-target for 1 sprint. Final sync diff before Phase 3. Rollback boundary: restore inids_rbac.db from pre-migration backup.

Step 18 (RS256 + revocation) — maintenance window required:

Announce 24 hours in advance. All existing tokens are invalid at cutover. Window duration: 5 minutes for deployment + up to 1 hour for user re-authentication. Rollback: re-enable HS256 with non-placeholder key; coordinate token re-issuance.

Step 19 (rate limiter unification) — after Step 16 gate passes:

Deploy new limiter → verify it fires on test request → remove RateLimitMiddleware → verify no double-limiting → remove InMemoryRateLimiter. Each sub-step has a 15-minute observation window.

Steps 20 + 21 (idempotency + webhook adapter) — after Step 13 gate passes:

Can be deployed in parallel. Both are additive (index addition, adapter flag addition).

Step 22 (correlation ID sanitization) — after Step 17 gate passes:

The audit trail being hardened must be the canonical OpsStore one. Monitor for correlation IDs being stripped (expected for malformed headers; unexpected for well-formed headers).

C-GATE-FINAL (before Phase D):

 INIDS_AUTH_COMPAT=false deployed and stable for 48 hours
 RS256 sole JWT algorithm; no HS256 tokens accepted
 RBAC fully in OpsStore; inids_rbac.db retired
 Rate limiter unified; no duplicate limiters in codebase
 DB-level idempotency for prevention actions confirmed by UNIQUE INDEX existence check
 Webhook adapter reconcile gap eliminated; no spurious DESYNCED records
 Audit trail sanitized; no log injection possible via correlation ID header
PHASE D — HARDENING & DEPTH DEFENSE
Timeline: Month 2 | Tier 2 items

Objective: Harden systems that are now architecturally sound but lack defensive depth.

Pre-condition: C-GATE-FINAL complete.

Steps 23–30 may be deployed in parallel within the following dependency constraints:

Step 23 (delete dead code): after Steps 16-17 confirm no one imports it
Step 24 (input sanitizer): after Step 3 (routes have auth — now harden their input)
Step 25 (packet capture sanitization): after Step 24
Step 26 (prevention defaults + TI feed): after Steps 2 + 9; CRITICAL: confirm Step 2 deployed before disabling dry_run in ANY environment
Step 27 (nftables JSON parsing): after Step 21
Step 28 (alert IDs + deduplication): after Steps 12 + 13; schedule index creation in maintenance window
Step 29 (health check probe): after Step 13
Step 30 (alert pagination + retention): after Steps 12 + 28
Phase D validation gates (each step):

Step 26 special gate: Before disabling dry_run, confirm: (1) INIDS_TI_FEED_PATH set to a real feed with no RFC-1918 indicators, (2) prevention scheduler running and leader-elected, (3) test block+unblock of a non-internal IP in staging.

D-GATE-FINAL (before Phase E):

 production_hardening.py deleted; no imports remain
 Input sanitizer wired to all user-submitted string fields
 Packet capture outputs sanitized; numeric fields clamped
 dry_run requires explicit INIDS_DRY_RUN configuration; no silent default
 TI feed loader validates and rejects RFC-1918 ranges
 nftables unblock uses JSON output mode
 Full UUID alert IDs with deduplication; no truncated IDs
 Health check is read-only; audit table clean
 Alert retention policy running; OpsStore storage bounded
PHASE E — QUALITY, OBSERVABILITY & REGRESSION INFRASTRUCTURE
Timeline: Month 2–3 | Tier 3 items

Objective: Instrument the stable system for long-term operational safety.

Pre-condition: D-GATE-FINAL complete.

Steps 31–38 are all independently deployable with no inter-step dependencies except Step 34 (requires Step 29) and Step 37 (requires Step 21) and Step 38 (requires Step 13). Deploy in parallel batches.

Operational dashboards to build in this phase (as specified by the recovery document):

Auth failures per minute (by route, by failure reason)
Rate limit hits per minute (by IP tier vs user tier vs route tier)
Prevention state: active blocks count, adapter type, reconcile success rate
Model inference health: verdict distribution, unknown_rate, inference latency p50/p99
Leader election state: leader instance ID, last renewal timestamp
Alert volume: creation rate, deduplication ratio, retention delete rate
E-GATE-FINAL (before Phase F):

 Steps 31-38 all deployed and validated
 All dashboards operational with at least 48 hours of data
 Localhost whitelist fixed (127.0.0.1, ::1)
 CSP unsafe-inline removed (all inline scripts externalized)
 Real uptime and health metrics displayed
 OPS_DB_PATH removed from public health endpoint
 Rate counter using deque, O(k) not O(n)
 Honeypot confidence and env var correct
 Webhook TLS enforced
 OpsStore pending approvals uses public API method
PHASE F — LEGACY REMOVAL & LONG-TERM DEBT ELIMINATION
Timeline: Quarter 2 | Tier 4 + post-migration cleanup

Objective: Remove ALL compatibility shims, deprecated systems, and dual-write windows created during Phases A–E.

Pre-condition: E-GATE-FINAL complete. Every Phase F removal requires a corresponding passing regression test.

F-AUTH-REMOVE (new entry — required by G-AUTH-2 resolution):

Remove INIDS_AUTH_COMPAT flag from codebase and all documentation
Delete src/auth_service.py (original API-key auth)
Delete src/auth_jwt.py (original JWT auth)
Delete @require_auth_legacy decorator acceptance from startup validator
Validation: All routes use @require_roles exclusively; startup validator has no legacy decorator acceptance; CI confirms no imports of old auth modules
Rollback: Restore from git (these are deletions)
Step 39 (decompose app.py): One blueprint at a time. Each blueprint extraction validated with existing integration test suite before extracting the next. URL prefixes must exactly match current paths.

Step 40 (full regression suite): Assembles individual security tests written per-step into a unified suite with coverage gates (≥ 80% on src/auth/, src/detection/, src/ips/). This step makes the per-step tests blocking in CI.

Step 41 (ML model versioning): Implement ACTIVE_MODELS manifest, staged model loading, atomic swap. Requires G-ML-1 (artifact store) to be operationally resolved.

Step 42 (CVE audit): Run pip-audit; upgrade cryptography ≥ 42.0.8, numpy ≥ 1.26.4, Werkzeug ≥ 3.0.6; regenerate hashes; add pip-audit to CI.

Step 43 (suspicious label inversion): One-line fix; write test verifying suspicious = attack with low confidence.

Step 44 (ContextVar isolation): Requires Step 22 (sanitized IDs) and Step 40 (test infra). Add test: two concurrent requests with different X-Correlation-IDs never mix.

Old auth token format (HS256): Already closed at Step 18. No additional Phase F action required.

Dual-write windows: InMemoryAlertStore dual-write closed in Phase B Step 12. RBAC dual-write closed in Phase C Step 17. All windows confirmed closed before Phase F begins.

F-GATE-FINAL:

 INIDS_AUTH_COMPAT flag removed from all code and config
 src/auth_service.py (original) deleted
 src/auth_jwt.py deleted
 src/rbac_manager.py deleted; inids_rbac.db deleted from deployment
 InMemoryAlertStore class deleted
 InMemoryPreventionStore class deleted
 production_hardening.py already deleted (Phase D)
 app.py ≤ ~200 lines (factory + registration only); all routes in blueprints
 Regression test suite at ≥ 80% coverage gate on security-critical paths
 All attack chains AC-1 through AC-4 have automated tests that are blocking in CI
 INIDS_MODEL_VERIFY=warn option removed (strict-only, after full observation window)
EXECUTION PHASE 6 — ROLLBACK AUTHORITY MATRIX

┌──────────────────────────────────────────────────────────────────────┐
│ PHASE: A — Emergency Stabilization                                   │
├──────────────────────────────────────────────────────────────────────┤
│ ROLLBACK TRIGGER:                                                    │
│   Any of: (1) authenticated API consumers reporting 401 at rate      │
│   >5% above pre-deployment baseline; (2) application fails to start  │
│   after 3 restart attempts; (3) model integrity check blocking        │
│   startup on known-good models; (4) container health check fails for │
│   >90 seconds after deployment                                        │
│ ROLLBACK PROCEDURE:                                                  │
│   Step 1 rollback: Revert .env (change key values only; do NOT       │
│   re-enable ALLOW_UNAUTHENTICATED=true); revert auth_service.py      │
│   if _bypass_enabled() removal caused cascading failure              │
│   Step 2 rollback: Restore original load_threat_intel() with empty   │
│   list (not RFC-1918 mock indicators — those must not return)         │
│   Step 3 rollback: git revert decorator additions per-route          │
│   Step 4 rollback: Revert api_auth_login and api_auth_refresh        │
│   Step 5 rollback: docker pull [previous-tag]; restore compose file  │
│   Step 6 rollback: Set INIDS_MODEL_VERIFY=warn in env               │
│   Step 7 rollback: Revert requirements.txt; rebuild image            │
│ ROLLBACK OWNER: Incident Response Engineer + Platform Reliability    │
│   Engineer (joint authorization; either can trigger, both confirm)   │
│ ROLLBACK WINDOW:                                                     │
│   Steps 1–4: Immediate (env var or code revert, no rebuild)          │
│   Step 5: Up to 15 minutes (container rebuild required)              │
│   Steps 6–7: Immediate for Step 6 (env var); 10 min for Step 7      │
│   (image rebuild). Phase A rollback window closes 72 hours after     │
│   Phase A-GATE-FINAL is declared complete.                           │
│ POST-ROLLBACK STATE:                                                 │
│   Steps 1–4 rolled back: Authentication bypass re-enabled or JWT    │
│   passwordless login re-enabled. System returns to pre-emergency     │
│   state. Active exploitation may resume. Treat rollback as an        │
│   incident requiring immediate escalation and root cause analysis.  │
│   Step 5 rolled back: Container returns to root user with repo mount.│
│   Step 6 rolled back: Model integrity not enforced (warn mode).     │
│   Step 7 rolled back: Hash verification not enforced on image build. │
│ DATA IMPACT OF ROLLBACK:                                             │
│   None. Phase A changes are configuration and code changes only.     │
│   New API keys generated in Step 1 remain valid after rollback;      │
│   the old placeholder keys must NOT be restored.                     │
│ RE-ENTRY CONDITION:                                                  │
│   Root cause of rollback trigger identified. Staging environment     │
│   reproduces the issue and fix is validated. All API consumers       │
│   confirmed updated before re-attempting Step 3. Complete Phase A   │
│   again in full — do not skip steps on re-entry.                    │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ PHASE: B — Architectural Foundations                                  │
├──────────────────────────────────────────────────────────────────────┤
│ ROLLBACK TRIGGER:                                                    │
│   (1) OpsStore write latency p99 increases >200% after Step 12;     │
│   (2) Prevention scheduler stops running after Step 11 with Redis    │
│   available and INIDS_REDIS_REQUIRED unset; (3) PolicyConfig update  │
│   causes FrozenInstanceError in production code (missed call site);  │
│   (4) Alert data disappears from dashboard after Step 12 sub-deploy 3│
│ ROLLBACK PROCEDURE:                                                  │
│   Step 13 rollback: Re-enable old migration functions (one sprint   │
│   of dual-path support is built in). Existing schema_version table  │
│   is additive — does not interfere with old migrations.              │
│   Step 9 rollback: Revert prevention_service.py; re-introduces race │
│   condition but prevention continues functioning.                    │
│   Step 10 rollback: Revert anomaly_engine.py.                       │
│   Step 11 rollback: Revert leader_election.py; re-introduces         │
│   fail-open. If Redis was the cause, fix Redis first, then re-apply. │
│   Step 12 rollback (by sub-deploy stage):                            │
│     Sub-deploy 1 rollback: Remove dual-write; revert to in-memory.  │
│     Sub-deploy 2 rollback: Re-route reads to in-memory store.        │
│     Sub-deploy 3 rollback: Re-inject InMemoryAlertStore as           │
│     secondary write target; data in OpsStore from drain is retained. │
│   Steps 14, 15: Revert individual files (CORS registration,          │
│   auth_service.py sensor role).                                      │
│ ROLLBACK OWNER: Principal Software Architect authorizes rollback of  │
│   Steps 9, 10, 12. SRE/PRE authorizes rollback of Steps 11, 13,    │
│   14, 15. Any engineer can trigger Step 11 rollback on Redis failure.│
│ ROLLBACK WINDOW: 2 weeks (duration of Phase B). After Phase B-GATE-  │
│   FINAL, rollback cost increases significantly (Phase C depends on B).│
│ POST-ROLLBACK STATE:                                                 │
│   Varies by step. Step 12 rollback at sub-deploy 2 or 3: alerts in  │
│   OpsStore remain; in-memory store is re-populated from new writes  │
│   going forward. Brief gap in in-memory store content (expected).   │
│ DATA IMPACT OF ROLLBACK:                                             │
│   Step 12 sub-deploy 3 rollback: Data drained to OpsStore is        │
│   preserved. InMemoryAlertStore starts empty on re-injection (no    │
│   data lost — it was all drained to OpsStore before removal).        │
│   All other Phase B rollbacks: No data impact.                       │
│ RE-ENTRY CONDITION:                                                  │
│   Root cause resolved in staging. For Step 12 rollback: re-confirm  │
│   OpsStore performance under load before re-attempting sub-deploy 3. │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ PHASE: C — System Unification                                         │
├──────────────────────────────────────────────────────────────────────┤
│ ROLLBACK TRIGGER:                                                    │
│   (1) Authentication failure rate >1% of authenticated requests      │
│   after Step 16 (new system rejecting valid users); (2) RBAC data    │
│   discrepancy found after Step 17 migration; (3) Any 500 response   │
│   from auth-related code after Step 18 cutover; (4) Rate limiting   │
│   over-blocking (false positive 429s exceeding 0.1% of traffic)      │
│   after Step 19                                                      │
│ ROLLBACK PROCEDURE:                                                  │
│   Step 16: Set INIDS_AUTH_COMPAT=true — old auth system takes over  │
│   immediately. No code change, no rebuild.                           │
│   Step 17: Restore inids_rbac.db from pre-migration backup; re-      │
│   enable rbac_manager.py route checks; set INIDS_AUTH_COMPAT=true.  │
│   Note: OpsStore RBAC data from migration remains but is not used    │
│   while rbac_manager.py is active.                                   │
│   Step 18: Re-enable HS256 with non-placeholder key. All users must │
│   re-authenticate (tokens invalid on algo switch). Coordinate        │
│   rollback announcement as a planned event.                          │
│   Step 19: Re-register old RateLimitMiddleware (WSGI) and            │
│   InMemoryRateLimiter before_request hook. New unified limiter       │
│   remains deployed but old ones take priority until removed again.   │
│   Steps 20-22: Revert individual files. Step 20 rollback: DROP       │
│   INDEX uq_active_block; restore has_active_block() check.          │
│ ROLLBACK OWNER: Principal Software Architect + DevSecOps Lead (joint │
│   authorization required for Steps 16-18; these affect all users).  │
│   SRE authorizes Steps 19-22.                                        │
│ ROLLBACK WINDOW:                                                     │
│   Step 16: Indefinite (INIDS_AUTH_COMPAT flag exists for this).      │
│   Step 17: Until RBAC data in inids_rbac.db is deleted (not deleted  │
│   until Phase F — rollback always possible during Phase C).          │
│   Step 18: Until old HS256 JWTManager code is deleted (Phase F).    │
│   After C-GATE-FINAL and 2 weeks of stable Phase D operation,        │
│   Phase C rollback cost increases substantially.                     │
│ POST-ROLLBACK STATE:                                                 │
│   Step 16/17 rollback: System returns to bridge auth (Steps 1-4).   │
│   Users who obtained RS256 tokens must re-authenticate via old       │
│   system. Steps 1-4 bridge fixes remain in effect — not a full       │
│   regression to pre-Phase-A state.                                   │
│   Step 18 rollback: HS256 tokens re-issued. All RS256 tokens invalid.│
│ DATA IMPACT OF ROLLBACK:                                             │
│   Step 17 rollback: OpsStore users/api_keys/revoked_tokens tables   │
│   remain populated from migration. rbac_manager.py ignores them.    │
│   inids_rbac.db backup contains pre-migration authoritative state.  │
│   No data loss — the backup is the source of truth on rollback.      │
│   All other Phase C rollbacks: No data loss.                         │
│ RE-ENTRY CONDITION:                                                  │
│   Step 16: Identify which consumers triggered the auth failure.      │
│   Update them, verify in staging, then re-attempt with INIDS_AUTH_  │
│   COMPAT=false.                                                      │
│   Step 18: Identify which consumers cached tokens beyond 1 hour.    │
│   Enforce token expiry in consumer code before re-attempting.        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ PHASE: D — Hardening & Depth Defense                                  │
├──────────────────────────────────────────────────────────────────────┤
│ ROLLBACK TRIGGER:                                                    │
│   (1) Step 26: Any internal IP appears in an active block action    │
│   after enabling non-dry-run prevention; (2) Step 28: Alert IDs      │
│   break API consumers expecting short IDs; (3) Step 30: Retention   │
│   job deletes alerts that were still operationally relevant;         │
│   (4) Step 27: nftables unblock fails with JSON parse error          │
│ ROLLBACK PROCEDURE:                                                  │
│   Step 26: Re-set INIDS_DRY_RUN=true immediately (env var change).  │
│   Investigation of blocked internal IPs required before re-enabling. │
│   Step 28: Rollback is additive — old short IDs remain in OpsStore; │
│   revert dedup logic and index. New long IDs already written remain  │
│   (consumers must handle both formats — this was documented).        │
│   Step 29: Restore audit INSERT in _ops_probe (pollutes audits but  │
│   non-critical).                                                     │
│   Step 30: Disable retention job (INIDS_ALERT_RETENTION_DAYS=0 or   │
│   remove scheduler call). Pagination stays active (additive).        │
│ ROLLBACK OWNER: SRE authorizes all Phase D rollbacks. Step 26        │
│   rollback can be triggered by ANY on-call engineer immediately.    │
│ ROLLBACK WINDOW: Full Phase D duration plus 30 days post-D-GATE-    │
│   FINAL. After that, Phase E tests verify stability — rollback risk  │
│   assessment required before going back.                             │
│ POST-ROLLBACK STATE:                                                 │
│   Step 26: Prevention returns to dry-run. No active blocks applied. │
│   Step 30: Alert retention suspended; table will grow until          │
│   re-enabled.                                                       │
│ DATA IMPACT OF ROLLBACK:                                             │
│   Step 30: Any alerts deleted by the retention job before rollback  │
│   cannot be restored unless an external backup exists. Run Step 30  │
│   in staging with production-scale data and validate retention        │
│   thresholds before production deployment.                           │
│   Step 28: No data deleted. Deduplication suppression rollback       │
│   means duplicate alerts reappear — this is operationally noisy but │
│   not a data loss event.                                             │
│   All other Phase D rollbacks: No data loss.                         │
│ RE-ENTRY CONDITION:                                                  │
│   Step 26: Confirm TI feed contains no RFC-1918 addresses (automated │
│   validation check); run staging test with internal IP as source;   │
│   confirm it is NOT blocked. Then re-enable non-dry-run.            │
│   Step 30: Run retention on staging with production data copy; verify│
│   the correct rows are deleted (by retention age, not by importance).│
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ PHASE: E — Quality & Observability                                    │
├──────────────────────────────────────────────────────────────────────┤
│ ROLLBACK TRIGGER:                                                    │
│   (1) Step 32: Dashboard pages break after unsafe-inline CSP removal;│
│   (2) Step 35: Rate counter behavior change causes false-positive or  │
│   missed rate limit events; (3) Step 37: Legitimate webhook endpoint  │
│   using HTTP instead of HTTPS breaks after TLS enforcement           │
│ ROLLBACK PROCEDURE:                                                  │
│   Step 31: Revert one-line whitelist change in IPBlockingMiddleware. │
│   Step 32: Re-add 'unsafe-inline' to CSP temporarily; log for each  │
│   inline script missed in audit; externalize and re-attempt.         │
│   Step 35: Revert rate counter implementation.                       │
│   Step 36: Revert honeypot confidence and env var handling.          │
│   Step 37: Remove HTTPS scheme validation from WebhookFirewallAdapter│
│   (reverts to accepting HTTP URLs).                                  │
│   Step 38: Revert api_actions_pending to use _fetchall directly.    │
│ ROLLBACK OWNER: Any senior engineer for Steps 31-38. No joint        │
│   authorization required — all changes are contained and low-blast.  │
│ ROLLBACK WINDOW: Phase E changes are small and indefinitely          │
│   reversible. No window constraint.                                  │
│ POST-ROLLBACK STATE:                                                 │
│   Each step's rollback returns that specific component to its Phase  │
│   D state. No cross-step impact.                                     │
│ DATA IMPACT OF ROLLBACK: None for all Phase E changes.               │
│ RE-ENTRY CONDITION:                                                  │
│   Step 32: Audit templates fully; no remaining inline scripts.       │
│   Step 37: Update webhook URL configuration to HTTPS before          │
│   re-applying TLS enforcement.                                       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ PHASE: F — Legacy Removal & Debt Elimination                          │
├──────────────────────────────────────────────────────────────────────┤
│ ROLLBACK TRIGGER:                                                    │
│   (1) Deletion of src/auth_service.py or src/auth_jwt.py causes      │
│   import errors in unfound callers; (2) Blueprint extraction breaks  │
│   URL routing for any existing consumer; (3) Regression test reveals │
│   a Phase A-E fix was incomplete or has regressed                    │
│ ROLLBACK PROCEDURE:                                                  │
│   F-AUTH-REMOVE: Restore deleted files from git history; set         │
│   INIDS_AUTH_COMPAT=true. This is the last available rollback for   │
│   the auth system — after Phase F, the old auth code no longer       │
│   exists in the repository.                                          │
│   Step 39 (blueprint extraction): Each blueprint is independently    │
│   revertable. Restore the original app.py routes for the failing     │
│   blueprint; keep others extracted.                                  │
│   Step 40 (test suite): No rollback — test additions are non-        │
│   breaking. If tests reveal a Phase A-E regression, that regression  │
│   is fixed; the tests are not rolled back.                           │
│   Steps 41-44: Revert individual files.                              │
│ ROLLBACK OWNER: Principal Software Architect authorizes F-AUTH-REMOVE│
│   (this is the final removal of old auth code — irreversible in      │
│   spirit if not in git). SRE authorizes all other Phase F rollbacks. │
│ ROLLBACK WINDOW:                                                     │
│   F-AUTH-REMOVE: Reversible until git history is pruned. However,   │
│   the operational rollback window (where it makes sense to restore   │
│   old auth) closes when Step 40 regression suite confirms no auth    │
│   failures for the full Phase E observation period (minimum 30 days).│
│   After that window, restoring old auth code is a regression, not   │
│   a rollback.                                                        │
│   Steps 39-44: Indefinitely reversible via git.                      │
│ POST-ROLLBACK STATE:                                                 │
│   F-AUTH-REMOVE rollback: Old auth modules restored; system works    │
│   with INIDS_AUTH_COMPAT=true. UnifiedAuthService remains primary.  │
│   Blueprint extraction rollback: app.py grows back toward its        │
│   original size for the rolled-back domains.                         │
│ DATA IMPACT OF ROLLBACK:                                             │
│   None. Phase F changes are code deletions only. The data model      │
│   established in Phase C (users, api_keys, revoked_tokens in         │
│   OpsStore) is not changed.                                          │
│ RE-ENTRY CONDITION:                                                  │
│   F-AUTH-REMOVE: All callers of old auth modules identified and      │
│   updated. Startup validator confirms no legacy decorator usage.     │
│   Re-run Phase C auth validation gates before re-attempting removal. │
│   Step 39: URL routing verified in staging with contract tests       │
│   against all known API consumers before re-attempting extraction.   │
└──────────────────────────────────────────────────────────────────────┘
QUALITY CRITERIA VERIFICATION
Criterion	Status	Evidence
1. Every Tier 0 and Tier 1 finding has fully specified implementation target	SATISFIED	Change IDs A-01 through C-07 and condensed targets for all 22 Phase A-C steps with exact file paths, scope, and side effects
2. Every system replacement has documented coexistence window with entry/exit criteria	SATISFIED	Auth compat window (INIDS_AUTH_COMPAT), alert store dual-write (3 sub-deploys with observation windows), RBAC dual-write window, rate limiter overlap window
3. Every data migration has dual-write plan, source verification, rollback	SATISFIED	Alert store: 3-phase dual-write with drain verification; RBAC: export → migrate → final sync diff; both include source verification steps before source removal
4. Every auth migration has token compatibility window with sunset date	SATISFIED	HS256 tokens: 1-hour maximum lifetime ensures 1-hour minimum compat window; compat flag closes at Phase F F-AUTH-REMOVE; forced re-auth at Step 18 maintenance window
5. Every Phase A step has validation checkpoint confirmed before Phase B begins	SATISFIED	B-GATE-1 explicitly blocks all Phase B work until all 8 Phase A checks pass
6. Critical path explicitly identified with zero unresolved dependencies	SATISFIED	13-step critical path enumerated with zero-float steps identified
7. Every compatibility shim has a Phase F removal entry	SATISFIED	INIDS_AUTH_COMPAT → F-AUTH-REMOVE; @require_auth_legacy → removed in F-AUTH-REMOVE; INIDS_MODEL_VERIFY=warn → removed in Phase F
8. Every attack chain has regression test specification	SATISFIED	AC-1: test_login_requires_valid_credentials + test_expired_token_not_refreshable; AC-2: test_model_load_rejects_bad_checksum; AC-3: test_requirements_are_pinned_with_hashes + test_container_does_not_mount_source_tree; AC-4: test_fp_suppression_requires_auth + test_all_routes_require_auth; plus G-REG-4 resolution requiring per-step test writing
9. No implementation step listed without rollback procedure	SATISFIED	Every Change ID includes ROLLBACK PROCEDURE field; every Phase rollback entry covers all steps in that phase
10. Roadmap can be executed by a senior engineer without clarifying questions	SATISFIED	File paths, command examples, sub-deploy sequences, observation windows, metric names, env var names, and go/no-go criteria are all specified

# SYSTEM_RECONSTRUCTION

**Pass 1 — Static Architecture**

| Subsystem | Type | Owner File | State |
|---|---|---|---|
| Flask app | HTTP/WS host | `web_app/app.py` (1917 ln) | Module-level globals |
| 11 Blueprints | Route handlers | `web_app/blueprints/*` | Access globals via `import web_app.app as _m` |
| OpsStore | Persistence | `src/ops_store.py` (1228 ln) | SQLite/PG, schema v5 |
| EventBus | Pub/sub | in `app.py` | In-process |
| EngineRegistry | Detection | `src/detection/` | 5 engines + TI |
| EngineAggregator | Strategy | `src/detection/` | ANY_TRIGGER |
| UnifiedAuthService | Auth | `src/auth/auth_service.py` | RS256 JWT + API key |
| RS256JWTManager | JWT sign/verify | `src/auth/jwt_manager.py` | Process singleton |
| Middleware stack | Request/response | `src/middleware.py` (439 ln) | CSRF, CORS, IPBlock, audit, security headers |
| UnifiedRateLimiter | Throttling | `src/rate_limiter.py` | Redis + in-mem fallback |
| LeaderElection | HA gate | `src/ha/leader_election.py` | Redis SETNX |
| PreventionService | Policy + action | `src/prevention_service.py` | Frozen config + RLock |
| ActionExecutor | Firewall I/O | `src/ips/action_executor.py` | Adapter + circuit breaker |
| PreventionScheduler | Cron | `src/ips/` | 30s, leader-gated |
| RealTimeStreamer | WS emit | in `app.py` | EventBus → SocketIO |
| ThreatIntelManager | TI feeds | `src/threat_intel/` | CSV/JSON, leader-gated refresh |
| Stream pipeline | Redis stream | `src/pipeline/` | Optional |

**Persistent stores:** SQLite/PG tables `alerts`, `actions`, `audits`, `fp_suppressions`, `allowlist`, `users`, `api_keys`, `revoked_tokens`, `schema_version`. In-memory deques: `AuditLogMiddleware` (maxlen=10000), `InMemoryRateLimiter` (cap 50000).

**External integrations:** Redis (optional), Firewall adapter (mock/ufw/nftables/webhook), Elasticsearch (disabled default), Browser HTTP+WS.

**Contracts:** REST `/api/*` JSON; Socket.IO `/events` namespace (rooms: alerts, actions, metrics, perception); subprocess `ufw`/`nftables`; HTTPS POST webhook.

**Entry points:** Flask routes (133+), Socket.IO connect handlers, PreventionScheduler (30s thread), alert retention daemon, TI feed refresh (default 3600s), anomaly auto-fit (per-event), pipeline worker thread.

**Pass 2 — Runtime Flow (primary chains)**

| Flow | Sequence | State touches |
|---|---|---|
| API request | middleware chain → `require_roles()` → handler → OpsStore | `g.auth` set; audit deque append |
| Detection (stream) | pipeline event → EngineAggregator → `_on_detection_event` (app.py:665) → OpsStore.insert alert → EventBus DetectionEvent → RiskEngine → PolicyEngine → ActionExecutor → SocketIO emit | alerts table, EventBus, WS |
| Detection (direct predict) | `/api/predict` → DetectionService → OpsStore → EventBus → ... | as above; full-UUID alert IDs |
| Blocking | PolicyDecisionEvent → ActionExecutor.execute → adapter subprocess/HTTP → OpsStore.insert action → WS emit | actions table, circuit breaker state |
| Auth | POST `/api/auth/login` → SHA-256 key → `get_user_by_key_hash` → JWT issue | api_keys, JWT in-process key |
| Audit write | after_request → `request.headers.get('X-User-ID', 'anonymous')` → deque append | **SPOOFABLE — header source** |

**Failure swallowing observed:** anomaly buffer `add_sample()` exceptions at app.py:~687 → `logger.debug`; RealTimeStreamer broadcast exceptions logged only.

**Pass 3 — Event Propagation & Backpressure**

| Channel | Producer | Consumer | Bound | Drop policy |
|---|---|---|---|---|
| EventBus | Engine paths | `_on_detection_event`, RiskEngine, PolicyEngine | UNSPECIFIED-IN-REPORT (assumed in-process sync dispatch) | None visible |
| AuditLog deque | every request | reader queries | 10000 (ring) | Oldest evicted |
| InMemoryRateLimiter | rate checks | stale eviction | 50000 keys | Stale evict |
| Perception queue | pipeline events | 2 worker threads | Bounded | Drop if full |
| Redis stream | pipeline producer | pipeline worker | XADD default | UNSPECIFIED-IN-REPORT |
| SocketIO `/events` | RealTimeStreamer | browsers (unauthenticated) | None | Missed events not replayed |

**Pass 4 — Failure Mode Inventory**

| Subsystem | Process crash | Network partition | OOM/disk-full | Dep unavailable |
|---|---|---|---|---|
| Flask app | Eventlet `-w 1` → full outage | N/A in-process | OOM → restart → ephemeral JWT key → mass logout | OpsStore down = 500s |
| LeaderElection | Lease lost → all instances non-leader | Default no-leader (Redis-required true) | N/A | No Redis → IPS silently disabled |
| PreventionScheduler | Thread death silently swallowed (assumed) | N/A | N/A | Adapter CB open 60s |
| Alert retention daemon | Thread death | N/A | N/A | Multi-instance: concurrent delete (no lock) |
| RealTimeStreamer | Exception logged, continues | WS disconnect → no replay | N/A | N/A |
| AnomalyEngine auto-fit | Exception swallowed at debug level | N/A | N/A | N/A |
| OpsStore | Per-request connection (SQLite) | N/A | Unbounded growth (no retention indexes) | Caller 500 |

**Pass 5 — Scaling & Concurrency**

| Bottleneck | Site | Type |
|---|---|---|
| Single-writer | gunicorn `--worker-class eventlet -w 1` | Mandated by SocketIO threading |
| ML inference blocks loop | Eventlet single worker | CPU-bound on event loop |
| Shared globals | `web_app/app.py` module state | All blueprints read via deferred import |
| Unbounded queries | `_fetchall()` callers without LIMIT | Hot path: `_buildDashboardMetrics` last 100 alerts unbounded variant |
| Missing indexes | `audits.created_at`, `alerts.source_ip`, `actions.status` | Full-table scan |
| Per-request SQLite conn | `ops_store._connect()` | No pool |
| Audit deque lock | UNSPECIFIED-IN-REPORT (assumed `collections.deque` thread-safe append) | — |

---

# FINDING_LEDGER

| ID | Source§ | Category | Verification | Severity | Statement |
|---|---|---|---|---|---|
| F-001 | D, 4.1 P0-001 | CONFIG | VERIFIED | S0-SURVIVAL | `settings.py` reads `SECRET_KEY`, ignores `SECRET_KEY_FILE` injected by docker-compose → RuntimeError on container start. |
| F-002 | D, 4.1 P0-002 | SECURITY | VERIFIED | S0-SURVIVAL | No `INIDS_JWT_PRIVATE_KEY` configured; ephemeral RSA-2048 generated per process; restart invalidates all tokens. |
| F-003 | C.7, 3.8, 4.2 P1-001 | SECURITY | VERIFIED | S1-INTEGRITY | `AuditLogMiddleware` reads identity from client-controlled `X-User-ID` header → audit forgery. |
| F-004 | C.5, 4.2 P1-002 | LIFECYCLE | VERIFIED | S0-SURVIVAL | `INIDS_REDIS_REQUIRED` defaults true; no Redis → `is_leader=False` → PreventionScheduler runs but takes no action → IPS silently disabled. |
| F-005 | E, 3.8, 4.2 P1-003 | SECURITY | VERIFIED | S1-INTEGRITY | Socket.IO `/events` namespace accepts any connection; broadcasts detections/actions/metrics with no auth. |
| F-006 | C.3, 3.5, 4.2 P1-004 | PERSISTENCE | VERIFIED | S2-DEGRADATION | Missing indexes on `audits.created_at`, `alerts.source_ip`, `alerts.timestamp`, `actions.status`. |
| F-007 | D (app.py), 4.2 P1-005 | CONTRACT | VERIFIED | S1-INTEGRITY | `_on_detection_event` (app.py:665) emits short alert IDs `al_<10hex>` while DetectionService emits full UUIDs → coexisting ID formats in same table. |
| F-008 | C.7, F.8, 3.8 | SECURITY | VERIFIED | S2-DEGRADATION | CSP `style-src` retains `'unsafe-inline'` → CSS injection feasible. |
| F-009 | D (settings/app), 4.3 P2-002 | CONFIG | VERIFIED | S1-INTEGRITY | `SETTINGS.internal_cidrs` accessed via `hasattr` guard; field does not exist in dataclass → EntityEnrichmentEngine always `internal_cidrs=None`. |
| F-010 | D (docker-compose), 4.3 P2-003 | INTEGRATION | VERIFIED | S2-DEGRADATION | No Redis service in compose; pipeline + HA features unreachable from default compose deployment. |
| F-011 | 4.3 P2-004 | OBSERVABILITY | VERIFIED | S3-OPERATIONAL | CI coverage gate excludes `web_app/app.py`, `src/ops_store.py`, `src/middleware.py` — three largest modules ungated. |
| F-012 | F.1, 4.4 P3-1 | CONTRACT | VERIFIED | S4-HYGIENE | Circular module dependency: all 11 blueprints deferred-import `web_app.app`. |
| F-013 | F.8, C.7, 4.4 P3-2 | RESOURCE-MGMT | VERIFIED | S4-HYGIENE | `RateLimitMiddleware` instantiated in `register_middleware()` but not wired (C-05 removed); object retained. |
| F-014 | F.8, 4.4 P3-3 | EVENT-FLOW | VERIFIED | S3-OPERATIONAL | `temporal_correlation_engine` registered with zero patterns → no-op on every pipeline event. |
| F-015 | F.7, 4.4 P3-4 | CONTRACT | INFERRED-HIGH | S4-HYGIENE | `connexion_integration.py`/`connexion_router.py` present alongside Flask routing; possible duplicate router. |
| F-016 | 4.4 P3-5 | OBSERVABILITY | VERIFIED | S4-HYGIENE | 15+ `validate_phase_*.py` / `test_*.py` scripts at repo root outside pytest discovery. |
| F-017 | 4.4 P3-6 | OBSERVABILITY | VERIFIED | S4-HYGIENE | `global_state.js` at repo root unreferenced. |
| F-018 | 4.4 P3-7 | CONFIG | VERIFIED | S3-OPERATIONAL | `pyproject.toml` loose bounds vs hash-pinned `requirements.txt` — no sync check. |
| F-019 | 3.7, 4.4 P3-8 | RESOURCE-MGMT | VERIFIED | S2-DEGRADATION | No gzip/brotli compression on API responses. |
| F-020 | C.7, 3.7 | RESOURCE-MGMT | INFERRED-HIGH | S2-DEGRADATION | `_fetchall()` callers omit `LIMIT` → unbounded result sets possible; dashboard fetches last 100 alerts variant unbounded. |
| F-021 | C.2, C.7, 3.3 | SECURITY | INFERRED-LOW | S3-OPERATIONAL | Tier-1 IP rate-check (`check_ip()`) not visible in observed `register_middleware()` chain. |
| F-022 | C.5 | CONCURRENCY | VERIFIED | S2-DEGRADATION | Alert retention daemon has no distributed lock; multi-instance → concurrent delete. |
| F-023 | B.1, F-005 context | SECURITY | VERIFIED | S2-DEGRADATION | Tailwind, Socket.IO, Chart.js loaded from CDN with no SRI; CDN compromise → arbitrary JS in security dashboard. |
| F-024 | C.1, C.7, F.2 | CONTRACT | SPECULATIVE | S4-HYGIENE | "1917-line god file" is structural debt without bounded production failure. |
| F-025 | D (csrf_protection) | SECURITY | VERIFIED | S3-OPERATIONAL | `csrf_protect_middleware` runs on every request; `require_csrf_token` imported but unused on routes → dead enforcement. |
| F-026 | C.4, B.6 | LIFECYCLE | INFERRED-HIGH | S3-OPERATIONAL | No client-side automatic JWT refresh wiring visible; user session interrupted at 1h. |
| F-027 | D (.env) | SECURITY | VERIFIED | S3-OPERATIONAL | `.env` with real API keys on disk; no `INIDS_ANALYST_API_KEY` set → analyst role not seeded. |
| F-028 | D (decorators) | LIFECYCLE | VERIFIED | S1-INTEGRITY | `_get_ops_store()` returns `None` if `current_app.ops_store` not set → all protected routes 401 silently. |
| F-029 | 3.8 | SECURITY | INFERRED-LOW | S2-DEGRADATION | `ops_store.get_alert(id)` lacks tenant/owner scoping → IDOR by alert ID. |

---

# DEFERRED

- **F-024** — God-file size alone is hygiene without bounded failure; structural rework belongs in a separate program, not survival remediation.
- **F-029** — INFERRED-LOW IDOR claim; report did not confirm a tenancy model. Fix would require designing a scoping model not described in the report.
- **F-021** — INFERRED-LOW absence of `check_ip()` wiring; before designing a fix, the wiring must be verified (V-002 in OPEN_ASSUMPTIONS). Not remediated until verification.
- **F-015** — `connexion_*` modules: removal is conditional on confirming no live imports; covered by OPEN_ASSUMPTIONS A-005, deferred until verified.

---

# DEPENDENCY_GRAPH

```
FIX-001 (settings _FILE convention)         → []
FIX-002 (JWT persistent keypair)            → [FIX-001]
FIX-003 (audit identity from g.auth)        → []
FIX-004 (INIDS_REDIS_REQUIRED=false default for single-node) → []
FIX-005 (SocketIO /events auth gate)        → [FIX-002]
FIX-006 (DB indexes v6 migration)           → []
FIX-007 (alert ID full UUID at app.py:665)  → []
FIX-008 (CSP style-src unsafe-inline removal) → []
FIX-009 (Settings.internal_cidrs field)     → []
FIX-010 (Redis service in compose)          → [FIX-001, FIX-002]
FIX-011 (CI coverage gate expansion)        → []
FIX-012 (anomaly auto-fit exception path)   → []
FIX-013 (RealTimeStreamer exception path)   → []
FIX-014 (alert retention distributed lock)  → []
FIX-015 (decorators _get_ops_store guard)   → []
FIX-016 (remove dead RateLimitMiddleware instantiation) → []
FIX-017 (deregister no-pattern temporal engine) → []
FIX-018 (LIMIT enforcement in _fetchall)    → []
FIX-019 (gzip via flask-compress)           → []
FIX-020 (SRI for CDN scripts)               → [FIX-008]
FIX-021 (CSRF middleware short-circuit for stateless API) → []
FIX-022 (client-side JWT refresh wiring)    → [FIX-002]
FIX-023 (analyst API key seeding)           → [FIX-001]
FIX-024 (dead/legacy file cleanup: root scripts, global_state.js) → []
FIX-025 (pyproject↔requirements sync CI)    → [FIX-011]
```

**Topological execution order:**
FIX-001 → FIX-003 → FIX-004 → FIX-007 → FIX-009 → FIX-015 → FIX-002 → FIX-006 → FIX-014 → FIX-018 → FIX-012 → FIX-013 → FIX-005 → FIX-022 → FIX-008 → FIX-020 → FIX-021 → FIX-016 → FIX-017 → FIX-019 → FIX-010 → FIX-023 → FIX-011 → FIX-025 → FIX-024

No cycles.

---

# STABILIZATION_PHASES

**Phase 0 — Survival**
Fixes: FIX-001, FIX-003, FIX-004, FIX-007.
Exit criterion: container starts from `docker-compose up` and emits zero `RuntimeError` lines; audit log entry written by an authenticated request shows `g.auth.username` (not header); `is_leader=true` in single-node deployment; all newly written alert IDs match UUID regex.

**Phase 1 — Containment**
Fixes: FIX-002, FIX-015, FIX-018, FIX-012, FIX-013, FIX-009.
Exit criterion: JWT keypair loads from file; `_fetchall` calls without LIMIT raise; anomaly auto-fit failures emit structured WARN with counter increment; `internal_cidrs` parsed at startup, log line shows configured CIDRs.

**Phase 2 — Correctness**
Fixes: FIX-006, FIX-014, FIX-017.
Exit criterion: `SCHEMA_VERSION=6` recorded; alert retention runs only on leader; temporal engine deregistered until patterns exist.

**Phase 3 — Integration & Contracts**
Fixes: FIX-005, FIX-022, FIX-021, FIX-023.
Exit criterion: SocketIO `/events` rejects unauthenticated upgrade; browser JS refreshes JWT before expiry; CSRF middleware no-ops for `/api/*`; analyst role seeded.

**Phase 4 — Operational Hardening**
Fixes: FIX-008, FIX-020, FIX-010, FIX-011, FIX-016, FIX-024, FIX-025.
Exit criterion: response CSP carries no `'unsafe-inline'`; CDN script tags carry SRI hash; `redis` service reachable from `inids-web`; CI gate covers the three added modules at 50% line coverage; dead code removed.

**Phase 5 — Targeted Optimization**
Fixes: FIX-019.
Exit criterion: gzip `Content-Encoding` on `>1KB` JSON responses confirmed via curl probe.

---

# FIX_SPECIFICATIONS

### FIX-001
ADDRESSES: F-001
PHASE: 0

TARGET:
- File/module: `src/settings.py`
- Function/class/config: `load_settings()`
- Insertion point: Top of function body, before the existing `os.getenv("SECRET_KEY", ...)` read.

CURRENT BEHAVIOR (1 sentence): `load_settings()` reads `SECRET_KEY` directly from env and raises `RuntimeError` when the env var is empty, ignoring `SECRET_KEY_FILE`.
NEW BEHAVIOR (1 sentence): `load_settings()` resolves `<KEY>_FILE` paths first by reading the file content, falling back to the plain env var.

IMPLEMENTATION STEPS:
1. Add module-level helper in `src/settings.py`:
   ```python
   def _read_file_secret(env_key: str, fallback_key: str = "") -> str:
       file_path = os.getenv(f"{env_key}_FILE", "").strip()
       if file_path:
           try:
               return Path(file_path).read_text(encoding="utf-8").strip()
           except OSError as e:
               logger.error("settings._FILE_read_failed", extra={"key": env_key, "path": file_path, "err": str(e)})
       return os.getenv(env_key, os.getenv(fallback_key, "") if fallback_key else "").strip()
   ```
2. Replace each direct `os.getenv(...)` for the following keys with `_read_file_secret(...)`: `SECRET_KEY` (fallback `FLASK_SECRET_KEY`), `INIDS_ADMIN_API_KEY`, `INIDS_SENSOR_API_KEY`, `INIDS_VIEWER_API_KEY`, `INIDS_ANALYST_API_KEY`, `INIDS_JWT_PRIVATE_KEY`, `INIDS_JWT_PUBLIC_KEY`.
3. Import `from pathlib import Path` and `import logging; logger = logging.getLogger(__name__)` at top if absent.
4. Add unit test `tests/test_settings_file_secrets.py` covering: `_FILE` path read, `_FILE` missing-file falls back to plain env, both absent raises for `SECRET_KEY`.

CONCURRENCY POSTURE: None (single-threaded import-time read).
TIMEOUTS / RETRIES / CIRCUIT BREAKER: No retry on `OSError`; one read attempt; failure logged and falls through to plain env.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: additive — `<KEY>_FILE` env vars become recognized; plain env continues to work.

FAILURE HANDLING:
- On timeout: N/A (local filesystem).
- On exception (OSError reading file): log error; fall through to plain env; if both empty and key is `SECRET_KEY`, existing `RuntimeError` fires.
- On downstream unavailable: N/A.
- On poison input (empty file): treated as empty; existing fail-closed path triggers for `SECRET_KEY`.
- On partial success: file present, env also set → file wins.

BLAST RADIUS:
- Directly affects: `src/settings.py` import path; everything downstream that reads `SETTINGS`.
- Isolated from: detection pipeline, OpsStore, EventBus.
- Regression candidates: app fails to boot when both `_FILE` and plain env are absent for `SECRET_KEY` (intended, identical to current).

ROLLBACK:
- Reverse procedure: revert `src/settings.py`; remove `<KEY>_FILE` references from compose env section.
- Data compatibility window: immediate; no persisted state changed.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: structured log `settings.loaded` with field `secret_source` ∈ `{"file","env"}`; new pytest `tests/test_settings_file_secrets.py::test_secret_from_file`.
- Healthy threshold: zero `settings._FILE_read_failed` log lines in first 60s post-start.
- Unhealthy signature: `RuntimeError: SECRET_KEY environment variable is required` in container stderr.

---

### FIX-002
ADDRESSES: F-002
PHASE: 1

TARGET:
- File/module: `deploy/compose/docker-compose.yml`, `deploy/compose/secrets/` (new files), `src/auth/jwt_manager.py`, `.env.example`.
- Function/class/config: compose `secrets:` and `environment:` sections; `RS256JWTManager.__init__`.
- Insertion point: compose secrets stanza; jwt_manager constructor key-loading branch.

CURRENT BEHAVIOR (1 sentence): `RS256JWTManager` generates an ephemeral RSA-2048 keypair when env keys absent and proceeds with a WARNING.
NEW BEHAVIOR (1 sentence): Compose injects `INIDS_JWT_PRIVATE_KEY_FILE` and `INIDS_JWT_PUBLIC_KEY_FILE`; settings reads them via `_read_file_secret` (FIX-001); `RS256JWTManager` refuses ephemeral key generation when `INIDS_JWT_REQUIRE_PERSISTENT=true`.

IMPLEMENTATION STEPS:
1. Generate keypair (operator step, documented in `deploy/compose/README.md`):
   `openssl genrsa -out deploy/compose/secrets/jwt_private.pem 2048`
   `openssl rsa -in deploy/compose/secrets/jwt_private.pem -pubout -out deploy/compose/secrets/jwt_public.pem`
2. In `docker-compose.yml` add to top-level `secrets:`:
   ```yaml
   inids_jwt_private_key:
     file: ./secrets/jwt_private.pem
   inids_jwt_public_key:
     file: ./secrets/jwt_public.pem
   ```
3. Attach to `inids-web` service `secrets:` list and add to environment:
   ```yaml
   - INIDS_JWT_PRIVATE_KEY_FILE=/run/secrets/inids_jwt_private_key
   - INIDS_JWT_PUBLIC_KEY_FILE=/run/secrets/inids_jwt_public_key
   - INIDS_JWT_REQUIRE_PERSISTENT=true
   ```
4. In `src/auth/jwt_manager.py` `__init__`, after attempting to load `INIDS_JWT_PRIVATE_KEY`, if absent and `os.getenv("INIDS_JWT_REQUIRE_PERSISTENT","false").lower()=="true"`, raise `RuntimeError("INIDS_JWT_PRIVATE_KEY required but not provided")` instead of generating ephemeral.
5. Add `.env.example` lines documenting `INIDS_JWT_PRIVATE_KEY_FILE` / `INIDS_JWT_PUBLIC_KEY_FILE` / `INIDS_JWT_REQUIRE_PERSISTENT`.
6. Add startup log line `auth.jwt_key_source` ∈ `{"file","env","ephemeral"}`.

CONCURRENCY POSTURE: Process singleton; one-time load at module import.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: None; single read at startup.
PERSISTENCE IMPACT: none (keys are config, not DB).
CONTRACT IMPACT: none — JWT format unchanged; only key material persists.

FAILURE HANDLING:
- On timeout: N/A.
- On exception (file unreadable): log error; with `REQUIRE_PERSISTENT=true` raise and abort startup; else ephemeral fallback with WARN.
- On downstream unavailable: N/A.
- On poison input (malformed PEM): `cryptography` raises ValueError → abort startup.
- On partial success (private present, public absent): existing public-key-derivation path executes, log WARN `auth.jwt_pub_derived`.

BLAST RADIUS:
- Directly affects: auth flow, all token issuance and verification.
- Isolated from: detection pipeline, OpsStore (except revocation table reads continue).
- Regression candidates: tokens issued by a previous container are invalid until first deploy with new persistent keys (one-time forced re-login).

ROLLBACK:
- Reverse procedure: set `INIDS_JWT_REQUIRE_PERSISTENT=false`; remove `*_FILE` envs; container regenerates ephemeral key.
- Data compatibility window: rollback invalidates tokens issued under persistent key; clients must re-login.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: startup log `auth.jwt_key_source=file`; integration test that restarts the app and verifies a token issued pre-restart still verifies post-restart.
- Healthy threshold: zero `auth.jwt_key_source=ephemeral` lines on production startup.
- Unhealthy signature: log `auth.jwt_key_source=ephemeral` with `INIDS_JWT_REQUIRE_PERSISTENT=true` (impossible if FIX correct; if seen, startup misconfig).

---

### FIX-003
ADDRESSES: F-003
PHASE: 0

TARGET:
- File/module: `src/middleware.py`
- Function/class/config: `AuditLogMiddleware.after_request`
- Insertion point: Line 230, the assignment `user = request.headers.get('X-User-ID', 'anonymous')`.

CURRENT BEHAVIOR (1 sentence): Audit entry's user field is taken from the client-controlled `X-User-ID` header.
NEW BEHAVIOR (1 sentence): Audit entry's user field is taken from `g.auth.username` set by `require_roles()`; falls back to `"anonymous"` for unauthenticated routes; header is ignored.

IMPLEMENTATION STEPS:
1. At top of `src/middleware.py`, ensure `from flask import g` is imported.
2. Replace line 230:
   ```python
   user = request.headers.get('X-User-ID', 'anonymous')
   ```
   with:
   ```python
   auth_ctx = getattr(g, 'auth', None)
   user = getattr(auth_ctx, 'username', None) or 'anonymous'
   ```
3. Add explicit log field `audit.source="g.auth"` on the audit deque entry to enable post-deploy verification.
4. Add regression test in `tests/test_middleware_audit.py`: send request with `X-User-ID: admin` header against an authenticated route as user `viewer`; assert audit entry shows `viewer`.

CONCURRENCY POSTURE: No change; same per-request thread/eventlet context.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none (in-memory deque only).
CONTRACT IMPACT: breaking for any internal tooling that relied on `X-User-ID` to label audit entries — none found in report.

FAILURE HANDLING:
- On timeout: N/A.
- On exception (g.auth attribute access throws): caught by existing global handler; audit entry omitted (existing behavior preserved).
- On downstream unavailable: N/A.
- On poison input (`X-User-ID` header still set by client): ignored.
- On partial success (unauthenticated route, `g.auth` not set): `user = "anonymous"`.

BLAST RADIUS:
- Directly affects: audit log entries written after this fix.
- Isolated from: detection pipeline, prevention, persistence.
- Regression candidates: dashboards or queries that filter audit by `X-User-ID`-derived values will see corrected attribution; any external SIEM consuming audit must accept new values.

ROLLBACK:
- Reverse procedure: revert the two-line change.
- Data compatibility window: immediate; deque is in-memory.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: pytest `tests/test_middleware_audit.py::test_audit_ignores_x_user_id_header`.
- Healthy threshold: test passes; manual curl with `X-User-ID: admin` and a viewer JWT shows `user=viewer` via `GET /api/audit/recent`.
- Unhealthy signature: audit entry `user` field matches `X-User-ID` header value on any authenticated request.

---

### FIX-004
ADDRESSES: F-004
PHASE: 0

TARGET:
- File/module: `.env`, `.env.example`, `deploy/compose/docker-compose.yml` (single-node template variant).
- Function/class/config: env variable `INIDS_REDIS_REQUIRED`.
- Insertion point: env files; compose `environment:` section of `inids-web` service.

CURRENT BEHAVIOR (1 sentence): `INIDS_REDIS_REQUIRED` defaults to `true`; without Redis, `LeaderElection._is_leader=False` and the PreventionScheduler executes no actions.
NEW BEHAVIOR (1 sentence): For single-node deployments, env is shipped with `INIDS_REDIS_REQUIRED=false`, making `LeaderElection._is_leader=True` when no Redis client is configured; multi-node deployments override with `true` and a working `REDIS_URL`.

IMPLEMENTATION STEPS:
1. Add to `.env.example` with documentation block:
   ```
   # Single-instance: false → prevention scheduler enabled without Redis.
   # Multi-instance: true → Redis required for leader election; set REDIS_URL too.
   INIDS_REDIS_REQUIRED=false
   ```
2. Add same line to `.env` (dev).
3. In `deploy/compose/docker-compose.yml`, add to `inids-web` environment:
   ```
   - INIDS_REDIS_REQUIRED=${INIDS_REDIS_REQUIRED:-false}
   ```
4. Add startup log line in `LeaderElection.__init__`: `ha.leader_init redis_required=<bool> redis_client=<bool> is_leader=<bool>`.
5. Add `/api/health` field `is_leader: <bool>`.

CONCURRENCY POSTURE: No change.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: additive (`/api/health` adds `is_leader` field).

FAILURE HANDLING:
- On timeout: N/A.
- On exception: N/A.
- On downstream unavailable (no Redis, `INIDS_REDIS_REQUIRED=true`): `is_leader=False` (preserved behavior); operator must set `false`.
- On poison input (non-boolean env): treat as `true` (fail-closed, preserves current default).
- On partial success: N/A.

BLAST RADIUS:
- Directly affects: PreventionScheduler, TI feed refresh, alert retention daemon (FIX-014).
- Isolated from: auth, detection, persistence reads.
- Regression candidates: in any deployment that previously relied on `is_leader=False` to suppress blocking (e.g., a "monitor-only" deployment), automatic blocking will now fire; the existing `DRY_RUN` flag remains the correct gate for that.

ROLLBACK:
- Reverse procedure: set `INIDS_REDIS_REQUIRED=true` in env.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: startup log `ha.leader_init is_leader=true`; `GET /api/health` returns `is_leader=true`.
- Healthy threshold: in single-node, `is_leader=true`; PreventionScheduler tick counter increments and a synthetic detection results in an entry in `actions` table.
- Unhealthy signature: PreventionScheduler tick log `scheduler.tick skipped=not_leader` in a single-node deployment.

---

### FIX-005
ADDRESSES: F-005
PHASE: 3

TARGET:
- File/module: `web_app/app.py` (SocketIO connect handlers, ~line 1740+); `web_app/static/js/core/socket-manager.js`.
- Function/class/config: `@socketio.on('connect', namespace='/events')`; client-side `io('/events', ...)` invocation.
- Insertion point: top of the events-namespace connect handler; `socket-manager.js` connection bootstrap.

CURRENT BEHAVIOR (1 sentence): `/events` namespace accepts any client and broadcasts detections, actions, metrics, and perception events to anonymous browsers.
NEW BEHAVIOR (1 sentence): `/events` namespace requires a valid JWT presented via Socket.IO `auth.token`, validated by `UnifiedAuthService.authenticate_jwt`; connections without a valid token are disconnected before joining any room.

IMPLEMENTATION STEPS:
1. In `web_app/app.py` connect handler:
   ```python
   from flask_socketio import disconnect
   from flask import request
   @socketio.on('connect', namespace='/events')
   def handle_events_connect(auth=None):
       token = (auth or {}).get('token') if isinstance(auth, dict) else None
       if not token:
           hdr = request.headers.get('Authorization', '')
           if hdr.startswith('Bearer '):
               token = hdr[7:]
       if not token:
           logger.warning("ws.connect_rejected reason=no_token sid=%s", request.sid)
           return False  # rejects the connection
       ctx = UnifiedAuthService(ops_store).authenticate_jwt(token)
       if ctx is None:
           logger.warning("ws.connect_rejected reason=invalid_token sid=%s", request.sid)
           return False
       # bind ctx to session for room-join authorization
       request.environ['inids_auth_ctx'] = ctx
       logger.info("ws.connect_accepted user=%s sid=%s", ctx.username, request.sid)
   ```
2. Update `web_app/static/js/core/socket-manager.js` to read the current JWT from in-memory store and connect with `io('/events', { auth: { token: jwt } })`.
3. Add subscribe-event authorization: in each `subscribe_*` handler, verify `request.environ.get('inids_auth_ctx')` is present and has the required role (e.g., `subscribe_actions` requires `analyst`+).
4. Add metric counter `ws.connect_rejected_total{reason}` and `ws.connect_accepted_total{user}`.
5. Regression test in `tests/test_ws_auth.py`: connect without token → rejected; connect with viewer JWT → accepted; connect with revoked JWT → rejected.

CONCURRENCY POSTURE: Per-connection check at connect time; no shared state mutation beyond per-session `request.environ` slot.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: JWT verify uses existing `RS256JWTManager` (no network call); no timeout needed.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: breaking for WebSocket clients — all clients must now send `auth.token`. Versioning: SocketIO namespace `/events` semantics changed; document in API changelog.

FAILURE HANDLING:
- On timeout: N/A.
- On exception during verify: rejected; log `ws.connect_error`.
- On downstream unavailable (OpsStore down during revocation check): connection rejected; log `ws.connect_rejected reason=revocation_check_failed`.
- On poison input (malformed token): rejected.
- On partial success (expired token within grace): rejected — refresh must be done over HTTP first.

BLAST RADIUS:
- Directly affects: all browser sessions; the dashboard, alerts page, investigate page.
- Isolated from: detection pipeline, OpsStore writes.
- Regression candidates: any external WebSocket consumer not updated to send a token will silently disconnect.

ROLLBACK:
- Reverse procedure: replace handler body with `pass` and revert client to unauthenticated `io('/events')`.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: `ws.connect_rejected_total{reason="no_token"}` counter.
- Healthy threshold: post-deploy, `ws.connect_accepted_total` ≥ active dashboard tab count; `ws.connect_rejected_total{reason="no_token"}` near zero for known clients.
- Unhealthy signature: dashboard shows "disconnected" badge persistently for an authenticated user (indicates client not passing token correctly).

---

### FIX-006
ADDRESSES: F-006
PHASE: 2

TARGET:
- File/module: `src/ops_store.py`
- Function/class/config: schema migration registry; bump `SCHEMA_VERSION` from 5 to 6; new function `_migration_v6_indexes`.
- Insertion point: alongside existing `_migration_v*_*` functions; registered in the version dispatch dict.

CURRENT BEHAVIOR (1 sentence): No indexes on `audits.created_at`, `alerts.source_ip`, `alerts.timestamp`, `actions.status` — common investigation queries scan entire tables.
NEW BEHAVIOR (1 sentence): Migration v6 adds four `CREATE INDEX IF NOT EXISTS` statements for both SQLite and PostgreSQL code paths; `SCHEMA_VERSION` bumped to 6.

IMPLEMENTATION STEPS:
1. Define `_migration_v6_indexes(conn, dialect)` issuing:
   ```sql
   CREATE INDEX IF NOT EXISTS idx_alerts_source_ip ON alerts(source_ip);
   CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
   CREATE INDEX IF NOT EXISTS idx_audits_created_at ON audits(created_at);
   CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status);
   ```
2. Append `(6, _migration_v6_indexes)` to the migration registry.
3. Set `SCHEMA_VERSION = 6`.
4. Add pytest `tests/test_ops_store_v6.py` that creates a fresh DB, asserts `schema_version=6`, and queries `sqlite_master` / `pg_indexes` for each new index name.

CONCURRENCY POSTURE: Migration runs at startup under existing migration lock; `CREATE INDEX IF NOT EXISTS` is idempotent and safe to retry. Note: on PostgreSQL, `CREATE INDEX` (non-concurrent) acquires a SHARE lock that blocks writes — acceptable during startup pre-traffic; for large production tables, operators can pre-create with `CONCURRENTLY` manually and the migration's `IF NOT EXISTS` will no-op.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: Existing migration error handling; no new retry.
PERSISTENCE IMPACT: additive (indexes created); migration plan = run-once forward; rollback supported by dropping indexes.
CONTRACT IMPACT: none.

FAILURE HANDLING:
- On timeout: long index build on PostgreSQL may exceed deploy timeout; operator pre-creates with `CONCURRENTLY` if table large; migration's `IF NOT EXISTS` then no-ops.
- On exception: existing migration error path aborts startup; revert by dropping `schema_version` row and indexes.
- On downstream unavailable: N/A.
- On poison input: N/A.
- On partial success (3 of 4 created): retry on next start completes remaining.

BLAST RADIUS:
- Directly affects: query plans for any SELECT against the four columns; write amplification on INSERT/UPDATE of these tables.
- Isolated from: detection pipeline logic, auth.
- Regression candidates: INSERT throughput on `alerts`, `audits`, `actions` decreases marginally — verify with FIX-006 verification metric.

ROLLBACK:
- Reverse procedure: `DROP INDEX IF EXISTS idx_alerts_source_ip, idx_alerts_timestamp, idx_audits_created_at, idx_actions_status;` and set `schema_version=5`.
- Data compatibility window: indefinite (indexes do not change data).
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: startup log `db.migration_applied version=6`; pytest in step 4.
- Healthy threshold: query `EXPLAIN SELECT * FROM alerts WHERE source_ip=?` uses index after migration.
- Unhealthy signature: post-migration `EXPLAIN` still shows `SEQ SCAN`/`SCAN TABLE alerts`.

---

### FIX-007
ADDRESSES: F-007
PHASE: 0

TARGET:
- File/module: `web_app/app.py`
- Function/class/config: `_on_detection_event()` (~line 665).
- Insertion point: the alert dict construction line `"id": f"al_{uuid.uuid4().hex[:10]}"`.

CURRENT BEHAVIOR (1 sentence): Alert IDs from the streaming/event-bus path use the short format `al_<10hex>` while `DetectionService` emits full UUIDs, causing two ID formats in the same table.
NEW BEHAVIOR (1 sentence): All alert IDs are `str(uuid.uuid4())` regardless of code path, matching the format established by D-06.

IMPLEMENTATION STEPS:
1. Open `web_app/app.py` at line ~665 inside `_on_detection_event()`.
2. Replace `"id": f"al_{uuid.uuid4().hex[:10]}"` with `"id": str(uuid.uuid4())`.
3. Add regression test `tests/test_detection_event_uuid.py`: push a synthetic detection event through `_on_detection_event` and assert the resulting alert ID matches `^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`.
4. Add startup data check: log warning if any existing alerts have IDs not matching the UUID regex (informational only — historical data not rewritten).

CONCURRENCY POSTURE: No change.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none for existing rows; new rows use new format. Historical short-form IDs remain valid as opaque strings.
CONTRACT IMPACT: none (IDs are opaque to API consumers per existing contract).

FAILURE HANDLING:
- On timeout / exception / etc: N/A (single-line behavioral change in pure function).

BLAST RADIUS:
- Directly affects: streaming-path alerts only.
- Isolated from: persistence, auth, prevention.
- Regression candidates: any consumer that filtered by `al_` prefix will need updating — none found in report.

ROLLBACK:
- Reverse procedure: revert single line.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: pytest `tests/test_detection_event_uuid.py`; post-deploy SQL `SELECT count(*) FROM alerts WHERE id NOT LIKE '%-%-%-%-%' AND created_at > <deploy_ts>` returns 0.
- Healthy threshold: count = 0.
- Unhealthy signature: count > 0 after deploy timestamp.

---

### FIX-008
ADDRESSES: F-008
PHASE: 4

TARGET:
- File/module: `src/middleware.py` (`SecurityHeadersMiddleware.HEADERS`), templates in `web_app/templates/`.
- Function/class/config: CSP header `style-src` directive.
- Insertion point: `HEADERS["Content-Security-Policy"]` definition; templates with inline `style=...` attributes.

CURRENT BEHAVIOR (1 sentence): CSP `style-src` includes `'unsafe-inline'`, permitting CSS injection in any rendered template.
NEW BEHAVIOR (1 sentence): CSP `style-src` no longer includes `'unsafe-inline'`; all inline `style=...` attributes in templates are moved to externalized CSS classes or files (the script-externalization pattern E-02 applied).

IMPLEMENTATION STEPS:
1. Inventory inline styles: `grep -rE 'style="' web_app/templates/ web_app/static/js/`.
2. For each occurrence, replace with a class defined in `web_app/static/css/inids-inline-replacements.css`.
3. Add that CSS file to `base.html`.
4. Edit `SecurityHeadersMiddleware.HEADERS` `Content-Security-Policy` `style-src` directive to drop `'unsafe-inline'`. Keep allow-list for known stylesheet hosts plus `'self'`.
5. Add post-deploy smoke test: render each page route, assert HTTP 200 and absence of console errors via the headless test harness already present in `tests/`.

CONCURRENCY POSTURE: N/A.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: none (CSP is response header; tightening).

FAILURE HANDLING:
- On poison input: residual inline style that was missed → browser refuses to apply it; visible CSP violation in browser console; not a server-side failure.

BLAST RADIUS:
- Directly affects: all rendered HTML pages.
- Isolated from: API, detection pipeline.
- Regression candidates: any page that relied on JS injecting inline styles (e.g., chart tooltips) — Chart.js supports CSP-compliant rendering; verify.

ROLLBACK:
- Reverse procedure: re-add `'unsafe-inline'` to `style-src`.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: response header `Content-Security-Policy` lacks `'unsafe-inline'` for `style-src`; CSP violation report endpoint count of `style-src-elem` and `style-src-attr` violations.
- Healthy threshold: 0 CSP violations in 24h post-deploy.
- Unhealthy signature: CSP violation reports referencing `style-src-attr`.

---

### FIX-009
ADDRESSES: F-009
PHASE: 1

TARGET:
- File/module: `src/settings.py`, `web_app/app.py` line ~283.
- Function/class/config: `Settings` dataclass; EntityEnrichmentEngine instantiation site.
- Insertion point: `Settings` dataclass field definition; replacement of the `hasattr()` guard at app.py:283.

CURRENT BEHAVIOR (1 sentence): `Settings` has no `internal_cidrs` field; `hasattr(SETTINGS, 'internal_cidrs')` is always False; EntityEnrichmentEngine instantiates with `internal_cidrs=None`.
NEW BEHAVIOR (1 sentence): `Settings.internal_cidrs: tuple[str, ...]` parsed from `INIDS_INTERNAL_CIDRS` (comma-separated); EntityEnrichmentEngine receives the parsed tuple.

IMPLEMENTATION STEPS:
1. In `src/settings.py`, add field to `Settings` dataclass: `internal_cidrs: tuple[str, ...] = ()`.
2. In `load_settings()`, parse: `internal_cidrs = tuple(c.strip() for c in os.getenv("INIDS_INTERNAL_CIDRS","").split(",") if c.strip())`.
3. Validate each entry parses via `ipaddress.ip_network(c, strict=False)`; on `ValueError`, log warning and skip that entry.
4. In `web_app/app.py` line ~283, remove `hasattr` guard; pass `SETTINGS.internal_cidrs` directly.
5. Document `INIDS_INTERNAL_CIDRS=10.0.0.0/8,172.16.0.0/12,192.168.0.0/16` in `.env.example`.

CONCURRENCY POSTURE: N/A (immutable tuple in frozen dataclass).
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: none.

FAILURE HANDLING:
- On poison input (malformed CIDR): logged, entry dropped, remaining entries used.
- On all entries invalid: `internal_cidrs=()`; EntityEnrichmentEngine receives empty tuple; behavior identical to current `None` case.

BLAST RADIUS:
- Directly affects: EntityEnrichmentEngine classification output → may change "internal_ip" flag on enriched events.
- Isolated from: auth, persistence, prevention.
- Regression candidates: detections that depended on `internal_ip=False` for all hosts (the current accidental default) may flip; verify on staging.

ROLLBACK:
- Reverse procedure: revert dataclass field and parse step; restore `hasattr` guard.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: startup log `enrichment.internal_cidrs count=<n> values=<csv>`.
- Healthy threshold: `count > 0` when env var set.
- Unhealthy signature: `count=0` despite env var being set with valid CIDRs.

---

### FIX-010
ADDRESSES: F-010
PHASE: 4

TARGET:
- File/module: `deploy/compose/docker-compose.yml`.
- Function/class/config: top-level `services:` and `volumes:` blocks.
- Insertion point: new `redis` service; `inids-web` environment additions.

CURRENT BEHAVIOR (1 sentence): No Redis container is defined; pipeline and Redis-backed rate limiter and Redis-required leader election are unreachable from the default compose stack.
NEW BEHAVIOR (1 sentence): A `redis:7-alpine` service is defined with a persistent volume; `inids-web` receives `REDIS_URL=redis://redis:6379/0` and `INIDS_REDIS_REQUIRED` is governed via `.env`.

IMPLEMENTATION STEPS:
1. Add to `docker-compose.yml` services:
   ```yaml
   redis:
     image: redis:7-alpine
     command: ["redis-server","--appendonly","yes"]
     volumes:
       - inids-redis:/data
     healthcheck:
       test: ["CMD","redis-cli","ping"]
       interval: 10s
       timeout: 3s
       retries: 5
     restart: unless-stopped
   ```
2. Add `inids-redis:` to top-level `volumes:`.
3. Add to `inids-web` environment: `- REDIS_URL=redis://redis:6379/0`.
4. Add `depends_on: redis: condition: service_healthy` to `inids-web`.
5. Document multi-node vs single-node `INIDS_REDIS_REQUIRED` matrix in `deploy/compose/README.md`.

CONCURRENCY POSTURE: N/A (compose-level).
TIMEOUTS / RETRIES / CIRCUIT BREAKER: Existing UnifiedRateLimiter Redis fallback remains. Healthcheck has `retries=5` × `interval=10s` = ~50s startup window.
PERSISTENCE IMPACT: Redis AOF persists rate-limit state and leader lease across restarts.
CONTRACT IMPACT: none.

FAILURE HANDLING:
- On timeout: `depends_on.condition=service_healthy` blocks `inids-web` startup until Redis is reachable (or 5×10s exhausted → compose reports failure).
- On Redis unavailable mid-run: rate limiter falls back to in-memory (existing); leader election demotes; PreventionScheduler stops blocking.
- On downstream unavailable: covered above.
- On poison input: N/A.
- On partial success: N/A.

BLAST RADIUS:
- Directly affects: rate limiter, leader election, pipeline.
- Isolated from: auth, OpsStore.
- Regression candidates: container resource usage rises by Redis footprint; minimal.

ROLLBACK:
- Reverse procedure: remove `redis` service; remove `REDIS_URL` env; set `INIDS_REDIS_REQUIRED=false`.
- Data compatibility window: stop Redis container; volume preserved unless explicitly removed.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: `docker compose ps` shows `redis` healthy; `inids-web` log `ratelimit.backend=redis` and `ha.leader_init redis_client=true`.
- Healthy threshold: both log lines present at startup.
- Unhealthy signature: `ratelimit.backend=memory` despite `REDIS_URL` set.

---

### FIX-011
ADDRESSES: F-011
PHASE: 4

TARGET:
- File/module: `.github/workflows/security.yml`.
- Function/class/config: pytest invocation step.
- Insertion point: pytest `--cov` arguments.

CURRENT BEHAVIOR (1 sentence): Coverage gate covers only `src/auth`, `src/detection`, `src/ips` at 80% threshold.
NEW BEHAVIOR (1 sentence): Coverage gate adds `src/ops_store`, `src/middleware`, `web_app` at an initial 50% threshold to unblock CI while expanding the gated surface.

IMPLEMENTATION STEPS:
1. Modify the pytest step in `.github/workflows/security.yml`:
   ```yaml
   - run: pytest --cov=src/auth --cov=src/detection --cov=src/ips
            --cov=src/ops_store --cov=src/middleware --cov=web_app
            --cov-fail-under=50
            --cov-report=term-missing
   ```
2. Add a per-module floor via `.coveragerc` or `coverage` config: `src/auth` 80%, `src/detection` 80%, `src/ips` 80%, others 50%.
3. Add CI step to publish coverage XML artifact.

CONCURRENCY POSTURE: N/A.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: none.

FAILURE HANDLING:
- On poison input (a module dropping below 50%): CI fails the PR.

BLAST RADIUS:
- Directly affects: CI workflow only.
- Isolated from: runtime.
- Regression candidates: PRs that previously passed may fail until tests added.

ROLLBACK:
- Reverse procedure: revert workflow change.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: GitHub Actions run status on main; coverage summary line.
- Healthy threshold: `TOTAL >= 50%`.
- Unhealthy signature: coverage gate failure on previously-passing PRs.

---

### FIX-012
ADDRESSES: F-007 swallow context (anomaly auto-fit), referenced under Pass 2 — supports observability of S1-INTEGRITY classes.
PHASE: 1

TARGET:
- File/module: `web_app/app.py` ~line 687 (anomaly buffer `add_sample`).
- Function/class/config: anomaly auto-fit exception block.
- Insertion point: existing `except` block currently calling `logger.debug`.

CURRENT BEHAVIOR (1 sentence): Exceptions from `anomaly_buffer.add_sample()` are swallowed at DEBUG level, hiding model fit failures from operators.
NEW BEHAVIOR (1 sentence): Exceptions are logged at WARNING with a structured payload and incremented on a counter `anomaly.add_sample_errors_total`.

IMPLEMENTATION STEPS:
1. Replace `logger.debug(...)` in the relevant `except Exception` block with:
   ```python
   logger.warning("anomaly.add_sample_failed", exc_info=True, extra={"engine":"anomaly"})
   _ANOMALY_ADD_SAMPLE_ERRORS.inc()  # module-level Counter or fallback simple counter
   ```
2. If Prometheus client not present (likely per report), define a simple thread-safe `IntCounter` in `src/_telemetry.py` and bind `_ANOMALY_ADD_SAMPLE_ERRORS` at module top.
3. Expose value via `/api/health` (`anomaly_add_sample_errors: <int>`).

CONCURRENCY POSTURE: Counter uses `threading.Lock` for increments (cheap).
TIMEOUTS / RETRIES / CIRCUIT BREAKER: None — failure logged and ignored as before.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: additive (`/api/health` adds field).

FAILURE HANDLING: same as current (exception swallowed after log); behavior unchanged except visibility.

BLAST RADIUS:
- Directly affects: log volume on `/health` and anomaly path.
- Isolated from: detection correctness.
- Regression candidates: log noise spike if anomaly path is failing — that is the desired signal.

ROLLBACK:
- Reverse procedure: revert log level to DEBUG and remove counter.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: `/api/health` field `anomaly_add_sample_errors`.
- Healthy threshold: counter does not increase faster than 1/min in steady state.
- Unhealthy signature: counter grows linearly with traffic.

---

### FIX-013
ADDRESSES: Pass 4 finding ("RealTimeStreamer broadcast exception logged, continues") — observability gap related to F-005 dependency.
PHASE: 1

TARGET:
- File/module: `web_app/app.py` (RealTimeStreamer registration / handler).
- Function/class/config: broadcast loop / SocketIO emit wrapper.
- Insertion point: the `except Exception` block in the broadcast path.

CURRENT BEHAVIOR (1 sentence): Broadcast exceptions are logged without rate-limiting or counter; repeated failures generate log spam without quantification.
NEW BEHAVIOR (1 sentence): Broadcast exceptions increment counter `streamer.emit_errors_total{room}`; logged at WARN with rate-limit (log throttler) of one line per 10s per room.

IMPLEMENTATION STEPS:
1. Define `_STREAMER_EMIT_ERRORS` IntCounter in `src/_telemetry.py`.
2. Wrap log call with a per-key `time.monotonic()` last-log timestamp dict guarded by lock; emit log only if `now - last >= 10`.
3. Expose counter via `/api/health` (`streamer_emit_errors_by_room: {room: count}`).

CONCURRENCY POSTURE: Lock-guarded dict.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A (continues on failure as before).
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: additive (`/api/health`).

FAILURE HANDLING: unchanged (broadcast skipped on error).

BLAST RADIUS:
- Directly affects: log volume and `/api/health`.
- Isolated from: detection.
- Regression candidates: none.

ROLLBACK:
- Reverse procedure: revert wrapper.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: `streamer.emit_errors_total{room}` in `/api/health`.
- Healthy threshold: 0 in steady state.
- Unhealthy signature: rising counter for any single room.

---

### FIX-014
ADDRESSES: F-022
PHASE: 2

TARGET:
- File/module: alert retention daemon thread (source location not named in report — most-specific locator: the alert-retention daemon thread referenced in C.5; bind symbol `alert_retention_worker` inside `web_app/app.py` start sequence).
- Function/class/config: retention loop body.
- Insertion point: top of each iteration before deletion query.

CURRENT BEHAVIOR (1 sentence): Daily retention thread runs on every instance with no coordination; in multi-instance deployments, concurrent DELETEs may race.
NEW BEHAVIOR (1 sentence): Retention loop calls `leader_election.is_leader()` at the start of each iteration and skips the deletion when not leader.

IMPLEMENTATION STEPS:
1. Locate the retention daemon (assumed in `web_app/app.py` start sequence; if it lives in `src/`, edit there).
2. At the top of the iteration body, add:
   ```python
   if not leader_election.is_leader():
       logger.info("retention.skipped reason=not_leader")
       time.sleep(<retention_interval>)
       continue
   ```
3. Emit metric `retention.runs_total` and `retention.skipped_not_leader_total`.

CONCURRENCY POSTURE: Single-leader-only execution via existing `LeaderElection` (Redis SETNX in multi-node, always-true in single-node after FIX-004).
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none (gating, not changing the delete logic).
CONTRACT IMPACT: none.

FAILURE HANDLING:
- On `leader_election.is_leader()` raising: caught, treated as not-leader, skip.
- On poison input: N/A.

BLAST RADIUS:
- Directly affects: alert retention timing in multi-instance.
- Isolated from: detection, auth.
- Regression candidates: in a misconfigured single-node deployment (FIX-004 not applied), retention would never run; FIX-004 must precede or be applied with this.

ROLLBACK:
- Reverse procedure: remove the `is_leader()` gate.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: `retention.runs_total`, `retention.skipped_not_leader_total`.
- Healthy threshold: in single-node with FIX-004 applied, `runs_total > 0` daily.
- Unhealthy signature: `runs_total=0` after 26h in single-node.

---

### FIX-015
ADDRESSES: F-028
PHASE: 1

TARGET:
- File/module: `src/auth/decorators.py`.
- Function/class/config: `_get_ops_store()` and `require_roles()`.
- Insertion point: `_get_ops_store()` return path.

CURRENT BEHAVIOR (1 sentence): `_get_ops_store()` silently returns `None` if `current_app.ops_store` is unset; downstream `require_roles()` then 401s every protected request without distinguishing missing-credential from missing-binding.
NEW BEHAVIOR (1 sentence): `_get_ops_store()` raises an internal exception that the global 500 handler converts to a 503 with body `{"error":"service_unavailable","reason":"auth_store_unbound"}`, distinguishing infrastructure misbinding from auth failure.

IMPLEMENTATION STEPS:
1. Define `class AuthStoreUnboundError(RuntimeError): pass` in `src/auth/decorators.py`.
2. Modify `_get_ops_store()`:
   ```python
   store = getattr(current_app, "ops_store", None)
   if store is None:
       logger.error("auth.ops_store_unbound")
       raise AuthStoreUnboundError("ops_store not attached to current_app")
   return store
   ```
3. Register handler in `web_app/app.py`:
   ```python
   @app.errorhandler(AuthStoreUnboundError)
   def _handle_auth_unbound(e):
       return jsonify({"error":"service_unavailable","reason":"auth_store_unbound"}), 503
   ```
4. Add startup smoke check: after app construction, assert `hasattr(app, 'ops_store')` and `app.ops_store is not None`; otherwise fail-fast at boot.

CONCURRENCY POSTURE: N/A.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: additive — new 503 response shape for a previously-401 condition (correct semantic).

FAILURE HANDLING:
- On `ops_store` truly absent: 503 returned; startup assertion would also catch this at boot in normal deployments.
- On exception during attribute access: same as above.

BLAST RADIUS:
- Directly affects: all 133+ protected routes when ops_store binding fails.
- Isolated from: detection, prevention.
- Regression candidates: test fixtures that omit `app.ops_store` will now fail with `AuthStoreUnboundError` instead of 401; tests must be updated to attach a store.

ROLLBACK:
- Reverse procedure: revert decorator changes and remove handler.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: startup assertion `auth.ops_store_bound=true` log line; absence of `auth.ops_store_unbound` log in steady state.
- Healthy threshold: zero `auth.ops_store_unbound` in 24h.
- Unhealthy signature: 503 with `reason=auth_store_unbound` on any request.

---

### FIX-016
ADDRESSES: F-013
PHASE: 4

TARGET:
- File/module: `src/middleware.py` (`register_middleware`).
- Function/class/config: removal of the unused `RateLimitMiddleware` instantiation.
- Insertion point: the line that creates `RateLimitMiddleware(...)` and stores it in the middleware dict.

CURRENT BEHAVIOR (1 sentence): `RateLimitMiddleware` is instantiated and retained in the middleware registry but receives no requests after C-05.
NEW BEHAVIOR (1 sentence): `RateLimitMiddleware` instantiation removed; the class definition remains (to avoid breaking imports elsewhere if any), but it is no longer created at startup.

IMPLEMENTATION STEPS:
1. In `register_middleware()`, delete the `RateLimitMiddleware(...)` instantiation line and its entry in any returned dict.
2. Search-and-remove `from .middleware import RateLimitMiddleware` imports if no remaining references.
3. Update tests that referenced the registered middleware (none should remain after C-05).

CONCURRENCY POSTURE: N/A.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: none.

FAILURE HANDLING: N/A.

BLAST RADIUS:
- Directly affects: middleware registry shape.
- Isolated from: everything else.
- Regression candidates: any test asserting presence of the middleware in the dict.

ROLLBACK:
- Reverse procedure: re-add the instantiation line.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: pytest collection passes; runtime log `middleware.registered` lists do not include `RateLimitMiddleware`.
- Healthy threshold: absent from list.
- Unhealthy signature: still present.

---

### FIX-017
ADDRESSES: F-014
PHASE: 2

TARGET:
- File/module: `web_app/app.py` (engine registry wiring).
- Function/class/config: `temporal_correlation_engine` registration site.
- Insertion point: the registration call.

CURRENT BEHAVIOR (1 sentence): `temporal_correlation_engine` is registered and invoked on every pipeline event but has zero patterns, so it always returns no-match while consuming CPU on each call.
NEW BEHAVIOR (1 sentence): The engine is registered only when `TemporalCorrelationEngine.pattern_count() > 0`; otherwise it is omitted from the aggregator chain and a log line states why.

IMPLEMENTATION STEPS:
1. Add method `pattern_count(self) -> int` to `TemporalCorrelationEngine` returning the size of its pattern store.
2. At the registration site, guard:
   ```python
   if temporal_engine.pattern_count() > 0:
       engine_registry.register("temporal", temporal_engine)
   else:
       logger.info("engine.temporal.skipped reason=no_patterns")
   ```
3. Expose status in `/api/health` (`engines: {"temporal":{"enabled":<bool>,"patterns":<n>}}`).

CONCURRENCY POSTURE: N/A.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: additive on `/api/health`.

FAILURE HANDLING: N/A.

BLAST RADIUS:
- Directly affects: pipeline per-event CPU.
- Isolated from: other engines.
- Regression candidates: if downstream code assumed temporal engine always present in the registry, those assumptions must be updated; report shows no such consumer.

ROLLBACK:
- Reverse procedure: register unconditionally.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: `/api/health` `engines.temporal.enabled`.
- Healthy threshold: with no patterns loaded → `enabled=false`; with patterns loaded → `enabled=true`.
- Unhealthy signature: `enabled=true` with `patterns=0`.

---

### FIX-018
ADDRESSES: F-020
PHASE: 1

TARGET:
- File/module: `src/ops_store.py`.
- Function/class/config: `_fetchall(...)` helper.
- Insertion point: function body.

CURRENT BEHAVIOR (1 sentence): `_fetchall()` executes any query as supplied; callers omitting `LIMIT` can fetch unbounded result sets.
NEW BEHAVIOR (1 sentence): `_fetchall()` enforces a `max_rows` parameter (default 1000) by wrapping the query with a `LIMIT` clause if absent and raising a `ValueError` if a row count exceeding `hard_max_rows=10000` is requested.

IMPLEMENTATION STEPS:
1. Change signature to `_fetchall(self, query, params=(), max_rows: int = 1000)`.
2. Detect presence of `LIMIT` in `query` (case-insensitive regex on the trailing clause); if absent, append `LIMIT :__max_rows`.
3. For SQLite, append `LIMIT ?` and inject `max_rows` to params; for SQLAlchemy/PG, use named binding.
4. If `max_rows > 10000`, raise `ValueError("max_rows exceeds hard cap")`.
5. Update existing callers: a one-shot grep `\b_fetchall\(` — for each, either pass `max_rows=` or accept the 1000 default; for the two known unbounded sites (dashboard "last 100 alerts" variant + audit ranges), set `max_rows=` explicitly.
6. Add pytest `tests/test_ops_store_limit.py` asserting an unbounded query returns ≤1000 rows.

CONCURRENCY POSTURE: N/A.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: additive — endpoints that returned >1000 rows now return ≤1000; consumers must paginate.

FAILURE HANDLING:
- On poison input (caller passes `max_rows=99999`): raises `ValueError`; bubbles to 500 handler.
- On caller-supplied `LIMIT` already present: respected as-is.

BLAST RADIUS:
- Directly affects: every `_fetchall` caller.
- Isolated from: write paths.
- Regression candidates: dashboards that displayed >1000 rows will be truncated; pagination must be added in a follow-up.

ROLLBACK:
- Reverse procedure: revert function signature; remove `LIMIT` injection.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: log line `db.fetchall_limit_injected query_hash=<h> max_rows=1000` count; assertion in pytest.
- Healthy threshold: zero rows returned > 1000 from any `_fetchall` call.
- Unhealthy signature: dashboard reports "showing 1000 of N" persistently — indicates pagination follow-up needed (operational, not a bug).

---

### FIX-019
ADDRESSES: F-019
PHASE: 5

TARGET:
- File/module: `web_app/app.py`; `requirements.txt`.
- Function/class/config: Flask app construction; `flask-compress` dependency.
- Insertion point: app factory.

CURRENT BEHAVIOR (1 sentence): API responses are served uncompressed; large alert/audit JSON payloads transit at full byte size.
NEW BEHAVIOR (1 sentence): `flask-compress` registered with `COMPRESS_MIMETYPES=['application/json','text/html','text/css','application/javascript']` and `COMPRESS_MIN_SIZE=1024`.

IMPLEMENTATION STEPS:
1. Add `flask-compress==1.15` (hash-pinned) to `requirements.txt` and `pyproject.toml`.
2. In app construction:
   ```python
   from flask_compress import Compress
   app.config["COMPRESS_MIMETYPES"] = ["application/json","text/html","text/css","application/javascript"]
   app.config["COMPRESS_MIN_SIZE"] = 1024
   Compress(app)
   ```
3. Add smoke test: request `/api/alerts?limit=1000` with `Accept-Encoding: gzip` and assert `Content-Encoding: gzip`.

CONCURRENCY POSTURE: N/A.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: none (transparent to clients).

FAILURE HANDLING:
- On client without `Accept-Encoding`: response served uncompressed (normal HTTP content-negotiation).

BLAST RADIUS:
- Directly affects: CPU per response (compression cost) and bandwidth (saving).
- Isolated from: data correctness.
- Regression candidates: latency on small responses unchanged due to `COMPRESS_MIN_SIZE`.

ROLLBACK:
- Reverse procedure: remove `Compress(app)` line and dependency.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: response header `Content-Encoding: gzip` on JSON >1KB.
- Healthy threshold: header present on alert list endpoints.
- Unhealthy signature: header absent on response of length > 1024 bytes with `Accept-Encoding: gzip`.

---

### FIX-020
ADDRESSES: F-023
PHASE: 4

TARGET:
- File/module: `web_app/templates/base.html`.
- Function/class/config: `<script>` and `<link rel="stylesheet">` tags referencing CDN.
- Insertion point: every `<script src="https://cdn...">` and `<link href="https://cdn...">`.

CURRENT BEHAVIOR (1 sentence): CDN scripts and stylesheets load without `integrity` / Subresource Integrity hashes; any CDN compromise injects arbitrary code into the security dashboard.
NEW BEHAVIOR (1 sentence): All CDN script and stylesheet tags carry `integrity="sha384-<hash>"` and `crossorigin="anonymous"`, pinning the exact bytes; CSP `script-src` and `style-src` are restricted to `'self'` plus the exact CDN origins.

IMPLEMENTATION STEPS:
1. Pin exact versions of each CDN asset (Tailwind, Bootstrap, Chart.js, Socket.IO).
2. Compute SHA-384 hash for each: `curl -s <url> | openssl dgst -sha384 -binary | openssl base64 -A`.
3. Add `integrity` and `crossorigin="anonymous"` to each CDN tag.
4. Update CSP `script-src` and `style-src` to enumerate the exact CDN origins; remove wildcards.
5. Add a CI step `scripts/check_cdn_integrity.py` that downloads each CDN URL referenced in `base.html` and verifies it matches the pinned hash; fails the build on mismatch.

CONCURRENCY POSTURE: N/A.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: CI integrity check uses 10s timeout per URL, 2 retries.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: none.

FAILURE HANDLING:
- On CDN bytes changing: CI fails — operator chooses to update hash or pin a different version.
- On client browser refusing the script (hash mismatch in production): script not executed; page degraded.

BLAST RADIUS:
- Directly affects: all rendered HTML pages.
- Isolated from: API, detection.
- Regression candidates: if a CDN auto-upgrades the file, browsers will refuse to load it until the hash is updated — that is the intended security property.

ROLLBACK:
- Reverse procedure: remove `integrity` attributes; relax CSP.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: CI step `cdn_integrity_check` exit 0; browser DevTools shows scripts loaded with no SRI errors.
- Healthy threshold: zero SRI errors in browser console.
- Unhealthy signature: console error "Failed to find a valid digest".

---

### FIX-021
ADDRESSES: F-025
PHASE: 3

TARGET:
- File/module: `web_app/app.py` (`csrf_protect_middleware` `before_request` registration); `src/csrf_protection.py`.
- Function/class/config: middleware function.
- Insertion point: top of middleware body.

CURRENT BEHAVIOR (1 sentence): `csrf_protect_middleware` runs on every request including `/api/*` where stateless JWT auth makes CSRF non-exploitable, consuming CPU and Flask session storage to no enforcement benefit.
NEW BEHAVIOR (1 sentence): `csrf_protect_middleware` short-circuits with `return None` on requests whose path starts with `/api/`; HTML routes continue to receive token issuance.

IMPLEMENTATION STEPS:
1. In `csrf_protect_middleware`, add at the top:
   ```python
   if request.path.startswith('/api/'):
       return None
   ```
2. Document in `src/csrf_protection.py` docstring that the middleware is HTML-only.
3. Add pytest asserting CSRF token cookie not set on `/api/health` response.

CONCURRENCY POSTURE: N/A.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: none.

FAILURE HANDLING: N/A.

BLAST RADIUS:
- Directly affects: per-request CPU on `/api/*`.
- Isolated from: HTML routes, CSRF for form posts.
- Regression candidates: any HTML form that submitted to a `/api/` URL relying on the cookie-based CSRF token must use the JWT auth path instead.

ROLLBACK:
- Reverse procedure: revert short-circuit.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: pytest `tests/test_csrf_api_skip.py`.
- Healthy threshold: API responses lack CSRF cookie.
- Unhealthy signature: CSRF cookie set on API response.

---

### FIX-022
ADDRESSES: F-026
PHASE: 3

TARGET:
- File/module: `web_app/static/js/core/http-client.js`.
- Function/class/config: fetch wrapper / interceptor.
- Insertion point: response handler; periodic timer setup.

CURRENT BEHAVIOR (1 sentence): No automatic JWT refresh; users hit 401 on any API call after the 1-hour TTL expires and are left without a session.
NEW BEHAVIOR (1 sentence): The HTTP client wrapper schedules a refresh call to `/api/auth/refresh` at 80% of token TTL (48 min by default); on a 401 with `error=token_expired`, it transparently retries the request once after refresh.

IMPLEMENTATION STEPS:
1. On login, store `expires_in` and compute `refresh_at = now + 0.8 * expires_in`.
2. Add `setTimeout` invoking `refreshToken()` at that point.
3. `refreshToken()` POSTs to `/api/auth/refresh` with current Bearer token; on success, replaces stored token and reschedules.
4. Wrap fetch: on 401 with payload `{"error":"token_expired"}`, call `refreshToken()` once; if successful, retry original request once; if not, redirect to login.
5. Persist current token only in memory (not localStorage) to limit exposure; reload page = re-login.

CONCURRENCY POSTURE: Single-flight refresh: a module-level `Promise` field ensures concurrent 401s share one refresh call.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: Refresh timeout 5s; one retry on the original request.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: none (uses existing `/api/auth/refresh`).

FAILURE HANDLING:
- On refresh timeout / failure: redirect to `/login`.
- On refresh returning revoked: redirect to `/login`.
- On concurrent 401s: single-flight via shared promise.

BLAST RADIUS:
- Directly affects: browser session continuity.
- Isolated from: server-side detection and auth logic.
- Regression candidates: none.

ROLLBACK:
- Reverse procedure: revert `http-client.js` changes.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: server log shows `/api/auth/refresh` call count per active session ≈ 1 per 48 min.
- Healthy threshold: 401 rate from browsers near zero except after revoke.
- Unhealthy signature: 401 spike at the 1h mark post-login.

---

### FIX-023
ADDRESSES: F-027
PHASE: 3

TARGET:
- File/module: `src/settings.py`; `_seed_service_accounts()` in `web_app/app.py` (or wherever currently implemented per report).
- Function/class/config: analyst API key handling.
- Insertion point: alongside admin/sensor/viewer key seeding.

CURRENT BEHAVIOR (1 sentence): `_seed_service_accounts()` reads admin/sensor/viewer keys but no analyst key, leaving analyst role unseeded from env.
NEW BEHAVIOR (1 sentence): `INIDS_ANALYST_API_KEY` (and `INIDS_ANALYST_API_KEY_FILE` per FIX-001) is read on startup and seeds an analyst-role user identical to the other roles.

IMPLEMENTATION STEPS:
1. In settings, add field `inids_analyst_api_key: str = ""` populated via `_read_file_secret("INIDS_ANALYST_API_KEY")`.
2. In `_seed_service_accounts()`, add a block mirroring the viewer seeding but with `roles=["analyst"]` and `username="analyst-service"`.
3. Add to `.env.example` documenting the env var.
4. Add to compose env section with `INIDS_ANALYST_API_KEY_FILE=/run/secrets/inids_analyst_api_key` and a corresponding secret entry.

CONCURRENCY POSTURE: N/A.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: idempotent INSERT into `api_keys` table.
CONTRACT IMPACT: none.

FAILURE HANDLING:
- On missing env: analyst account simply not seeded (matches current behavior for other roles).
- On duplicate seed: idempotent (existing pattern).

BLAST RADIUS:
- Directly affects: auth seeding only.
- Isolated from: detection, persistence schema.
- Regression candidates: none.

ROLLBACK:
- Reverse procedure: remove seed block; existing user row may be left in place harmlessly.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: startup log `auth.seeded role=analyst`.
- Healthy threshold: log line present when env set.
- Unhealthy signature: env set but log line missing.

---

### FIX-024
ADDRESSES: F-016, F-017
PHASE: 4

TARGET:
- File/module: root-level `validate_phase_*.py`, `test_*.py` scripts; root `global_state.js`.
- Function/class/config: filesystem cleanup.
- Insertion point: repo root.

CURRENT BEHAVIOR (1 sentence): 15+ ad-hoc validation scripts and an unreferenced `global_state.js` clutter the repo root with files outside pytest discovery.
NEW BEHAVIOR (1 sentence): Validation scripts moved to `tools/validate/`; `global_state.js` deleted.

IMPLEMENTATION STEPS:
1. `git mv` each `validate_phase_*.py` and `test_*.py` from root to `tools/validate/`.
2. `git rm global_state.js` after confirming no references via `grep -r global_state.js`.
3. Update `README.md` if it referenced these scripts.

CONCURRENCY POSTURE: N/A.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: N/A.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: none.

FAILURE HANDLING: N/A.

BLAST RADIUS:
- Directly affects: repo layout only.
- Isolated from: runtime.
- Regression candidates: any developer doc referencing old paths.

ROLLBACK:
- Reverse procedure: `git revert` the move/delete commit.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: `ls` of repo root shows no `validate_phase_*` and no `global_state.js`.
- Healthy threshold: clean root listing.
- Unhealthy signature: stragglers remain.

---

### FIX-025
ADDRESSES: F-018
PHASE: 4

TARGET:
- File/module: `.github/workflows/security.yml`; new `requirements.in`.
- Function/class/config: CI workflow step.
- Insertion point: new step that runs `pip-compile`.

CURRENT BEHAVIOR (1 sentence): `pyproject.toml` carries loose version bounds; `requirements.txt` is hash-pinned; nothing checks they remain consistent.
NEW BEHAVIOR (1 sentence): CI step runs `pip-compile --generate-hashes requirements.in` and fails if the produced `requirements.txt` differs from the committed file.

IMPLEMENTATION STEPS:
1. Author `requirements.in` derived from current `requirements.txt` (un-pinned top-level requirements).
2. Add CI step in `security.yml`:
   ```yaml
   - run: pip install pip-tools
   - run: pip-compile --generate-hashes --output-file=requirements.lock.check requirements.in
   - run: diff -u requirements.txt requirements.lock.check
   ```
3. Document update flow in `CONTRIBUTING.md`: edit `requirements.in`, run `pip-compile`, commit both.

CONCURRENCY POSTURE: N/A.
TIMEOUTS / RETRIES / CIRCUIT BREAKER: CI step timeout 5 min.
PERSISTENCE IMPACT: none.
CONTRACT IMPACT: none.

FAILURE HANDLING:
- On drift detected: CI fails; PR author runs `pip-compile` and commits.

BLAST RADIUS:
- Directly affects: CI only.
- Isolated from: runtime.
- Regression candidates: none.

ROLLBACK:
- Reverse procedure: revert workflow change.
- Data compatibility window: immediate.
- Point of no return: none.

VERIFICATION:
- Metric/log/assertion name: CI step status green; `requirements.lock.check` matches `requirements.txt`.
- Healthy threshold: zero diff.
- Unhealthy signature: persistent diff on main branch.

---

# EXECUTION_SEQUENCE

| # | Fix ID | Owner role | Size | Must precede | Must follow |
|---|---|---|---|---|---|
| 1 | FIX-001 | Principal Systems Architect | S | FIX-002, FIX-010, FIX-023 | — |
| 2 | FIX-003 | Senior Backend Engineer | S | — | — |
| 3 | FIX-004 | Production Reliability Engineer | S | FIX-014 (multi-instance correctness) | — |
| 4 | FIX-007 | Senior Backend Engineer | S | — | — |
| 5 | FIX-009 | Senior Backend Engineer | S | — | — |
| 6 | FIX-015 | Senior Backend Engineer | S | — | — |
| 7 | FIX-002 | Principal Systems Architect | M | FIX-005, FIX-010, FIX-022 | FIX-001 |
| 8 | FIX-006 | Distributed Systems Engineer | S | — | — |
| 9 | FIX-014 | Production Reliability Engineer | S | — | FIX-004 |
| 10 | FIX-018 | Senior Backend Engineer | M | — | — |
| 11 | FIX-012 | Production Reliability Engineer | S | — | — |
| 12 | FIX-013 | Production Reliability Engineer | S | — | — |
| 13 | FIX-005 | Distributed Systems Engineer | M | FIX-022 | FIX-002 |
| 14 | FIX-022 | Senior Backend Engineer | M | — | FIX-002 |
| 15 | FIX-008 | Stabilization Specialist | M | FIX-020 | — |
| 16 | FIX-020 | Stabilization Specialist | M | — | FIX-008 (ordering: 020 first) |
| 17 | FIX-021 | Senior Backend Engineer | S | — | — |
| 18 | FIX-016 | Stabilization Specialist | S | — | — |
| 19 | FIX-017 | Senior Backend Engineer | S | — | — |
| 20 | FIX-019 | Production Reliability Engineer | S | — | — |
| 21 | FIX-010 | Principal Systems Architect | S | — | FIX-001, FIX-002 |
| 22 | FIX-023 | Senior Backend Engineer | S | — | FIX-001 |
| 23 | FIX-011 | Stabilization Specialist | S | — | — |
| 24 | FIX-025 | Stabilization Specialist | S | — | FIX-011 |
| 25 | FIX-024 | Stabilization Specialist | S | — | — |

Note on FIX-008/FIX-020 ordering in the dependency graph: SRI hashes (FIX-020) must be applied before tightening CSP (FIX-008) so the tightened policy does not block scripts not yet carrying integrity attributes.

---

# OPEN_ASSUMPTIONS

| ID | Assumption | Depends on FIX | Falsification check |
|---|---|---|---|
| A-001 | `EventBus` dispatch is in-process synchronous; no separate broker. | FIX-005, FIX-013, FIX-017 | `grep -n "subscribe\|publish\|emit" app.py` shows direct function-call dispatch. |
| A-002 | Alert retention daemon lives in `web_app/app.py` startup sequence (not a separate src module). | FIX-014 | `grep -rn "retention" src/ web_app/`. |
| A-003 | `temporal_correlation_engine` has a `pattern_count()` or equivalent and is registered in the same place engines are bound. | FIX-017 | `grep -n "temporal" web_app/app.py src/detection/`. |
| A-004 | `flask-compress==1.15` is compatible with Flask 3.1.3 + eventlet. | FIX-019 | `pip install flask-compress==1.15 flask==3.1.3` in isolation, hit one endpoint. |
| A-005 | `src/connexion_integration.py` and `src/connexion_router.py` have no live imports from `web_app/app.py` or blueprints. | (F-015 deferred) | `grep -rn "connexion_integration\|connexion_router" web_app/ src/`. |
| A-006 | Existing `_seed_service_accounts()` is idempotent (re-running on startup with same key does not create a duplicate). | FIX-023 | Run app twice, query `SELECT count(*) FROM users WHERE username='analyst-service'`. |
| A-007 | `RS256JWTManager` constructor reads env at instantiation, allowing FIX-002's `REQUIRE_PERSISTENT` gate to fire before any token is issued. | FIX-002 | Trace `import` chain: `web_app/app.py` instantiates auth services before `register_blueprints`. |
| A-008 | The `SocketIO` connect handler can return `False` to reject the connection in the installed Flask-SocketIO 5.6.1 version. | FIX-005 | Flask-SocketIO docs version pin confirms; one local test connecting without auth returns `disconnect` event client-side. |
| A-009 | `_fetchall` is the single funnel for read queries — no separate `_fetchone` callers are unbounded in ways that matter. | FIX-018 | `grep -n "fetchall\|fetchone\|execute" src/ops_store.py`. |
| A-010 | `pip-compile` `--generate-hashes` is deterministic across Python 3.11 patch versions used in CI and in `requirements.txt` build. | FIX-025 | Run twice locally; diff output. |