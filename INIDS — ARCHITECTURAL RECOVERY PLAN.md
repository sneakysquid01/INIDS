INIDS — ARCHITECTURAL RECOVERY PLAN
Classification: Engineering — Recovery Sprint Authority

Derived from: Forensic Audit Report (72 findings, 13 categories)

Audience: Principal Architect, Senior Security Engineer, SRE, Backend Systems, DevSecOps, Incident Response

PHASE 1 — SYSTEM ARCHITECTURE RECONSTRUCTION
1. Authentication Architecture [CRITICALLY UNCLEAR]
Current State (as-built, not as-intended):

Three fully independent authentication systems operate in the same process with no bridging identity layer:

System 1 — API Key Authentication (src/auth_service.py)

Keys loaded from environment at startup: admin key, sensor key, viewer key
Keys stored in AuthService._principals dict mapping key → Principal(role, token)
require_auth decorator wraps API endpoints
Critical bypass: _bypass_enabled() returns True when ALLOW_UNAUTHENTICATED=true (committed value). When bypassed, all API key checks are skipped and a synthetic AuthContext is returned regardless of request content
No token expiry, no rotation mechanism, no revocation
Sensor key incorrectly mapped to "analyst" role rather than a restricted "sensor" role
System 2 — JWT Authentication (src/auth_jwt.py)

JWTManager initialized with secret_key from settings
When a secret_key string is provided, algorithm downgrades from ES256 to HS256 (symmetric signing)
api_auth_login endpoint issues tokens for any submitted username with zero credential verification
api_auth_refresh accepts expired tokens via allow_expired=True — effectively infinite token lifetime
require_auth decorator: returns HTTP 500 (not 401) if jwt_manager attribute is absent from request
jwt_require_role: checks request.claims for role membership without verifying the authentication chain was completed
No revocation list, no token binding, no audience or issuer claims validation
System 3 — RBAC (src/rbac_manager.py + inids_rbac.db)

Separate SQLAlchemy ORM model stored in inids_rbac.db
Manages roles/permissions for user IDs that exist in neither System 1 nor System 2
No foreign key or identity mapping to API keys or JWTs
AuditLog.id generated from ISO timestamp string — collision-prone under concurrent writes
Completely disconnected from the other two systems; never consulted by the main auth decorators
Signing Key Management:

SECRET_KEY=change-me-now committed in .env
settings.py validates non-empty but not minimum entropy
Same SECRET_KEY used for: Flask sessions, CSRF token generation, JWT HS256 signing
Docker Compose reads FLASK_SECRET_KEY from .env.example — different key name, resolves to undefined/default
Gap: There is no single source of identity truth. A caller can be simultaneously authenticated by System 1, issue themselves a System 2 token for admin, and have no entry in System 3. Authorization decisions depend entirely on which decorator a given route uses.

2. Authorization Flow [CRITICALLY UNCLEAR]
Enforcement Points (actual, not documented):

Routes with @require_auth (API key): enforced by auth_service.py
Routes with @jwt_required / @jwt_require_role: enforced by auth_jwt.py
~25 module-level routes: no decorator, no enforcement
RBAC endpoints: enforced by rbac_manager.py checks — disconnected from login state
Privilege Propagation:

API key auth produces a Principal object attached to request context
JWT auth produces claims dict on request.claims
Neither auth system's result is visible to the other system's decorators
The RBAC system operates on user_id strings with no mapping to the above
Missing Enforcement Zones (identified endpoints without auth):

FP suppression (submit/delete suppressions)
Escalation tracker
Behavioral profiling
Drift monitor configuration
Anomaly learning control
Network topology data
Forensic timeline query
Policy enforcement configuration reads
Role Hierarchy: Undefined system-wide. Each auth system defines its own role vocabulary without a master schema.

3. Prevention Pipeline [PARTIALLY DEFINED]
Signal Flow:


Network Traffic / API Submission
    → DetectionService.predict_from_features()
    → EventBus.publish(DetectionEvent)
    → RiskEngine (subscribes to DetectionEvent)
        → Computes composite risk score (weighted: confidence 0.5, severity 0.3, frequency 0.2)
        → EventBus.publish(RiskScoreEvent)
    → PolicyEngine (subscribes to RiskScoreEvent)
        → Consults PolicyConfig thresholds
        → Produces decision: BLOCK / TEMP_BLOCK / RATE_LIMIT / PENDING_BLOCK / ALERT / PASS
        → EventBus.publish(PolicyDecisionEvent)
    → ActionExecutor.execute(PolicyDecisionEvent)
        → Circuit breaker check
        → Idempotency check (ops_store.has_active_block) — TOCTOU window
        → Adapter dispatch: block_ip / rate_limit
        → OpsStore.save_action()
        → EventBus.publish(ActionEvent)
State Dependencies:

PolicyConfig: mutable shared dataclass — no lock, race condition between set_policy() and evaluate()
Default dry_run=True, mode="monitor" — prevention never executes in default configuration
ActionExecutor uses OpsStore for idempotency; PreventionService uses InMemoryPreventionStore — two non-synchronized action stores
Blocking Mechanism:

block_ip() → _call_adapter_with_timeout() → new ThreadPoolExecutor(max_workers=1) per call → adapter.block(ip, ttl)
Circuit breaker: 3 failures → 60-second open window → auto-close
No distributed lock on block operations in multi-instance deployments
Unblocking:

cleanup_expired_actions() called by prevention scheduler (background thread)
Scheduler uses leader election to determine if it should run — fails open if Redis unavailable
4. ML Pipeline [PARTIALLY DEFINED]
Model Loading:

load_models() called at application startup in app.py
joblib.load(path) deserializes .pkl files — no integrity verification (hash/signature)
AnomalyEngine also calls joblib.load() independently on its own model_path
No model versioning, no hot-reload mechanism, no rollback capability
Inference Path:


features dict
    → DEFAULT_FEATURE_ROW.copy() + overlay
    → pd.DataFrame([row], columns=FEATURE_COLUMNS)  [created per request]
    → model.predict(df) → prediction label (0/1)
    → model.predict_proba(df) → confidence array
    → EngineResult(verdict, confidence, severity, attack_type)
MLEngine Failure Modes:

predict_proba() exception: logged then re-raised, crashes the detection pipeline for this request
predict() exception: same re-raise behavior
Model not loaded: is_ready() returns False; caller must check or receives AttributeError
Feature count mismatch: AnomalyEngine silently truncates/pads — no error raised
AnomalyEngine Buffer/Training:

Buffer accumulates samples in _buffer list under _buffer_lock
Auto-fits when buffer_size samples collected
fit() writes self._model under lock
evaluate() reads self._model without lock — data race with fit()
5. Storage Systems [PARTIALLY DEFINED]
Layer 1 — OpsStore (Durable)

SQLite (dev) or PostgreSQL (prod) via dual-path _connect() context manager
Tables: alerts, actions, audits, fp_suppressions, allowlist, schema_version
Migrations run unconditionally on every startup (full-table UPDATE queries)
Schema version: checks but swallows non-RuntimeError exceptions
No connection pooling documented for SQLite path
Layer 2 — InMemoryAlertStore (Ephemeral, Detection Service)

deque(maxlen=1000) — bounded, lost on restart
Appends to front (appendleft) — newest-first ordering
Drops oldest on overflow (correct behavior) but logs the new alert as "dropped" (incorrect log attribution)
Layer 3 — InMemoryPreventionStore (Ephemeral, Prevention Service)

Simple list, no maxlen, no persistence
Lost on restart; diverges from OpsStore immediately
Layer 4 — Redis (Semi-Durable, Optional)

Used exclusively for leader election via SETNX
Not used for session state, caching, or rate limiting (despite two rate limiters existing)
Falls back to always-leader when unavailable
Layer 5 — RBAC Database (inids_rbac.db, Durable)

Separate SQLite file managed by SQLAlchemy ORM
Completely isolated from OpsStore
No cross-database transactions possible
Layer 6 — Adapter In-Memory State (Ephemeral)

WebhookFirewallAdapter.blocked_targets: dict, lost on restart
MockFirewallAdapter.blocked_targets: dict, lost on restart
Data Domain Ownership Conflicts:

Alerts: owned by both InMemoryAlertStore AND OpsStore.alerts
Actions: owned by both InMemoryPreventionStore AND OpsStore.actions
Blocked IPs: owned by adapter in-memory state AND OpsStore.actions
6. Alert Lifecycle [PARTIALLY DEFINED]
Creation Path A (Legacy — DetectionService):


DetectionService.predict_from_features()
    → Alert(id=f"al_{uuid.uuid4().hex[:10]}")  [40-bit ID, collision-prone]
    → InMemoryAlertStore.add(alert)
    → OpsStore.save_alert() [via EventBus subscriber]
Creation Path B (New — Multi-Engine Pipeline via app.py):


multi_engine_detect endpoint
    → engines[].evaluate()
    → EngineResult aggregated
    → OpsStore.save_alert() [direct call]
    → InMemoryAlertStore.add() [possibly, via EventBus]
Suppression:

FP suppression endpoint (unauthenticated) writes to OpsStore.fp_suppressions
All engines check fp_manager.is_suppressed(source_ip) at top of evaluate() — returns "normal" with 100% confidence
Any unauthenticated caller can suppress any source IP
Deduplication: None. Same source IP + same attack type generates a new alert on every evaluation.

Lifecycle States (in OpsStore): open → reviewing → closed / escalated

Alert Count Discrepancy: InMemoryAlertStore holds at most 1000 since restart; OpsStore holds all-time alerts. Dashboard reads from different sources depending on query path.

7. Rate Limiting Systems [PARTIALLY DEFINED]
Limiter 1 — RateLimitMiddleware (WSGI layer)

Applied to all requests before Flask routing
Maintains per-IP sliding window counter
State: in-memory, not shared across processes or instances
No synchronization between instances
Limiter 2 — InMemoryRateLimiter in _before_request_metrics (Flask before_request hook)

Applied after WSGI middleware, before route handler
Separate per-IP counter, separate window, separate threshold
State: in-memory, not shared
Interaction: Both limiters consume quota independently. A request that passes Limiter 1 but is blocked by Limiter 2 was still counted by Limiter 1. Under high load, an IP might be blocked by Limiter 1 but not Limiter 2 (if Limiter 2 has higher threshold), or vice versa. Neither limiter is aware of the other.

production_hardening.py SecurityHardeningManager.enforce_rate_limit():
A third rate-limiting implementation exists but is never called (dead code module). Its counter never resets — a permanent-block-after-first-limit-hit bug.

8. Middleware Chain [PARTIALLY DEFINED]
WSGI Layer (applied to all requests, before Flask routing):


Request →
    [1] SecurityHeadersMiddleware   — sets response headers (CSP with unsafe-inline, HSTS, etc.)
    [2] IPBlockingMiddleware        — checks request.remote_addr against block list
                                       (localhost whitelist non-functional: 'localhost' ≠ '127.0.0.1')
    [3] RateLimitMiddleware         — per-IP sliding window rate limiting
    [4] AuditLogMiddleware          — logs request using datetime.utcnow() (timezone-naive)
→ Flask Application
Flask Application Layer:


Flask request →
    before_request hooks (registration order determines execution):
        [5] CSRF check / token generation (via csrf_protection.py)
        [6] InMemoryRateLimiter check (second, independent rate limiter)
        [7] correlation_tracing middleware (ContextVar, X-Correlation-ID injection, no sanitization)
→ Route handler with optional decorators:
        [@require_auth] — API key check (or bypass if ALLOW_UNAUTHENTICATED=true)
        [@jwt_required] — JWT validation (or 500 if jwt_manager not on request)
        [none]          — ~25 endpoints have no decorator at all
CORSMiddleware:

Defined in src/middleware.py
Never registered with Flask — neither before_request nor after_request hooks attached
Effectively dead code; CORS is completely unenforced
Ordering Consequences:

IP blocking occurs before Flask auth — blocked IPs never reach auth, which is correct
Rate limiting occurs twice — at WSGI layer and Flask layer — independently
Security headers are applied at WSGI layer, so they apply even to error responses
9. Deployment & Runtime Environment [WELL-DEFINED]
Docker Compose Configuration (as-built):


env_file: ../../.env.example          # Wrong file — uses FLASK_SECRET_KEY not SECRET_KEY
volumes:
  - ../../:/app                        # Entire repo mounted — includes .env, DBs, models
command: >
  bash -lc "pip install --no-cache-dir -r requirements.txt && python web_app/app.py"
  # pip install runs on every container start — supply chain risk
  # No user: directive — runs as root (UID 0)
  # No healthcheck directive
  # No mem_limit, cpus, or ulimits
Environment Variable Mapping:

Application expects SECRET_KEY (from settings.py)
Docker Compose injects FLASK_SECRET_KEY (from .env.example)
ALLOW_UNAUTHENTICATED=true present in both .env and .env.example
API key placeholders: replace-admin-key, replace-sensor-key, replace-viewer-key
Startup Sequence:

Container starts as root
pip install -r requirements.txt (live PyPI fetch, no hash verification)
python web_app/app.py
load_settings() — validates SECRET_KEY non-empty; placeholder change-me-now passes
load_models() — joblib.load() on .pkl files — no integrity check
OpsStore.__init__() — unconditional UPDATE migrations run on every start
load_threat_intel() — hardcoded mock RFC-1918 indicators loaded
Prevention scheduler thread started — leader_election referenced before assignment (potential NameError)
Flask-SocketIO serves requests
10. Redis / Leader Election [PARTIALLY DEFINED]
Redis Usage:

Exclusively for leader election via SETNX (SET if Not eXists)
Not used for: sessions, rate limit state, cache, alert deduplication, or distributed locks
Optional dependency — application starts without Redis
Leader Election Mechanism:

ha/leader_election.py uses Redis SETNX with TTL
Renewal interval: time.sleep(self._ttl // 3) — integer division, busy-wait risk with small TTLs
Failure mode: Redis unavailable → is_leader() returns True on all instances
Split-brain consequence: all instances run prevention scheduler simultaneously
Duplicate block actions, duplicate audit records, duplicate firewall rules
Idempotency check in ActionExecutor has TOCTOU window under concurrent execution
11. Firewall Adapter System [WELL-DEFINED]
Adapter Hierarchy:


FirewallAdapter (ABC)
├── MockFirewallAdapter       — in-memory dict, no syscalls, for dev/test
├── UfwFirewallAdapter        — shells out to `ufw` binary, subprocess with timeout
├── NftablesFirewallAdapter   — shells out to `nft` binary, parses text for handle numbers
└── WebhookFirewallAdapter    — HTTP POST to external URL, no TLS enforcement
Adapter Selection: Configured via adapter_name in ActionExecutor constructor — set during app initialization, not changeable at runtime.

State Consistency Issues:

MockFirewallAdapter and WebhookFirewallAdapter: in-memory state lost on restart
UfwFirewallAdapter and NftablesFirewallAdapter: actual OS-level persistent rules (survive restart)
reconcile() compares OpsStore active blocks against adapter.list_rules() — valid only for OS-backed adapters; produces spurious DESYNCED for webhook/mock after restart
nftables Handle Parsing: unblock() parses nft -a list chain text output to extract rule handle numbers. Format-dependent, fragile across nftables versions.

12. Threading & Concurrency Model [PARTIALLY DEFINED]
Background Threads:

Prevention scheduler (leader-gated): cleanup_expired_actions, reconcile
AnomalyEngine auto-fit (triggered from request thread when buffer fills)
Flask-SocketIO worker threads (Gevent/Eventlet or threading mode)
Shared Mutable State and Synchronization:

Component	State	Lock	Thread-Safe?
_RateCounter._timestamps	per-IP timestamp list	threading.Lock	Yes (but TOCTOU in read-modify)
ThresholdEngine._counters	IP → counter map	threading.Lock	Yes
PolicyConfig (all fields)	policy thresholds	None	No — race condition
AnomalyEngine._model	trained model reference	_buffer_lock on write; none on read	No — write-read race
AnomalyEngine._buffer	sample accumulation	_buffer_lock	Yes
InMemoryAlertStore._alerts	alert deque	threading.Lock	Yes
InMemoryPreventionStore.actions	action list	None	No
SecurityHardeningManager.rate_limits	rate limit counters	None	No (dead code)
13. Persistence Boundaries [PARTIALLY DEFINED]
Durable State (survives restart):

OpsStore: alerts, actions, audits, fp_suppressions, allowlist (SQLite/PostgreSQL)
inids_rbac.db: RBAC roles/permissions/users
OS firewall rules (UFW/nftables adapters)
Model .pkl files on filesystem
Semi-Durable State (survives restart if Redis survives):

Leader election key (TTL-bound)
Ephemeral State (lost on restart):

InMemoryAlertStore (up to 1000 alerts)
InMemoryPreventionStore (all prevention actions)
WebhookFirewallAdapter.blocked_targets (all webhook blocks)
MockFirewallAdapter.blocked_targets
ThresholdEngine._counters (all rate tracking)
AnomalyEngine._buffer (collected training samples, unless model_path set)
Both in-process rate limiter states
Unrecoverable on Loss:

Any alert only in InMemoryAlertStore and not yet flushed to OpsStore
Any webhook-blocked IPs (no way to reconstruct what was blocked)
Rate counter state (attackers get fresh window on restart)
14. Configuration Loading [PARTIALLY DEFINED]
Loading Path:


.env file
    → python-dotenv (loaded at module import)
    → os.environ
    → settings.py:load_settings()
    → Settings dataclass
    → Passed to app initialization
    → Distributed to subsystems via __init__ parameters or direct env reads
Validation (current):

SECRET_KEY: non-empty check only
No validation of API key strength, minimum length, or non-placeholder values
No validation that ALLOW_UNAUTHENTICATED is false in non-development environments
No validation that model paths exist
Docker Compose reads wrong env file, causing SECRET_KEY to be undefined/default
Fail Behavior:

Empty SECRET_KEY: raises RuntimeError (fail-closed)
Missing model file: joblib.load() raises FileNotFoundError at startup (fail-closed)
ALLOW_UNAUTHENTICATED=true + placeholder keys: application starts normally (fail-open)
Redis unavailable: application starts normally, leader assumed (fail-open)
15. WebSocket & Event Systems [PARTIALLY DEFINED]
Internal Event Bus (src/core/event_bus.py):

In-process pub/sub: EventBus.publish() → registered handler callbacks
Event types: DetectionEvent, RiskScoreEvent, PolicyDecisionEvent, ActionEvent, AuditEvent
No async delivery — handlers called synchronously on publish()
No dead-letter queue, no retry, no ordering guarantees beyond call order
External WebSocket (Flask-SocketIO):

Real-time dashboard broadcasting via SocketIO events
Connection authentication: not documented in audit — assumed unauthenticated based on general auth patterns
Broadcasting occurs from background threads (leader-gated)
Events: alert creation, system metrics, prevention actions
ContextVar for correlation IDs may bleed between greenlets (Gevent mode)
16. Cross-Service Dependencies [PARTIALLY DEFINED]
Internal:

Detection engines → fp_manager.is_suppressed() (called on every evaluation)
ActionExecutor → OpsStore.has_active_block() (idempotency check)
Prevention scheduler → leader_election.is_leader() (Redis or always-True)
AnomalyEngine → joblib (optional persistence)
External:

WebhookFirewallAdapter → HTTP POST to external URL (no timeout enforcement beyond timeout_seconds)
UfwFirewallAdapter → ufw subprocess (5-second timeout)
NftablesFirewallAdapter → nft subprocess (5-second timeout)
Redis → leader election (optional, fallback to always-leader)
PyPI → pip install on container start (live network dependency at boot)
Failure Propagation:

Redis failure: leader election fails open → all instances act as leader → duplicate actions
Adapter failure (3x): circuit breaker opens → 60-second window of no blocking
predict_proba() exception: re-raised, crashes detection pipeline for that request
OpsStore connection failure at startup: raises exception, application does not start (fail-closed here)
PHASE 2 — CROSS-FINDING CORRELATION ANALYSIS
A. Root Cause Classification
The 72 findings reduce to 7 structural root causes:

RC-1: Identity System Was Never Designed — Only Accumulated
Root cause for: A-01, A-02, A-03, A-04, A-05, A-06, C-01, C-02, D-01, D-02 (indirectly), J-03, M-01, K-01, K-02, C-19

All authentication and authorization failures trace to the absence of a designed identity architecture. Three systems were added incrementally to the same codebase without integration. Each system has independent vulnerabilities, and the lack of integration means fixing one system does not close the gaps in the others.

RC-2: Development/Debug Configuration Committed as Production Configuration
Root cause for: A-03, C-01, C-02, C-05, C-14, C-15, G-01, G-02, G-03, M-03, M-05, L-01, L-02, L-03, L-04, L-05, C-06

.env was committed with all-bypass, all-placeholder values. Docker Compose references .env.example. dry_run=True is the default. Mock threat intel indicators are in production initialization code. The system was designed for developer convenience and never hardened for deployment.

RC-3: No Architectural Boundary Between Detection and Data Input Trust
Root cause for: C-03, C-15, I-01, M-02, C-16, C-17, C-18, C-11

The system trusts network data, user-submitted features, packet captures, and model files without establishing a boundary at which inputs are validated, sanitized, or verified. Arbitrary data flows directly into ML inference, log output, and deserialization paths.

RC-4: Shared Mutable State Without Ownership Model
Root cause for: D-02, A-07, A-08, M-06, J-04, B-04, D-06, D-07, M-08, A-12

Multiple components share mutable state without establishing ownership. PolicyConfig has no lock. AnomalyEngine._model is read without a lock. InMemoryPreventionStore and OpsStore both own prevention action state. InMemoryAlertStore and OpsStore both own alert state. No single-writer discipline was enforced.

RC-5: Persistence Strategy Has No Data Domain Ownership Model
Root cause for: D-05, D-06, D-07, I-02, I-03, I-05, M-07, M-08, B-01, F-03

Multiple storage layers exist for the same data (alerts in two stores, actions in two stores, blocked IPs in adapter memory and OpsStore) without a canonical owner. Migrations run unconditionally. Alert IDs use truncated UUIDs. The RBAC database is entirely isolated. State lost on restart creates reconciliation failures.

RC-6: Supply Chain and Deployment Trust Model Is Absent
Root cause for: C-03, C-04, C-06, H-01, H-02, H-03, H-04, H-05, L-01, L-02, L-03, L-04, L-05

The deployment model fetches packages live at boot, mounts the entire source tree into the container, runs as root, uses no content hash verification, and loads model files without integrity checks. There is no trust boundary between the host and the container, between PyPI and the running process, or between the model file and the inference engine.

RC-7: No Regression Prevention Infrastructure
Root cause for: K-01, K-02, K-03, K-04, and the fact that all RC-1 through RC-6 items exist

The absence of security-focused integration tests, auth boundary tests, and concurrency tests allowed every finding in this report to reach a committed state. The test suite verifies happy paths but does not verify security properties.

B. Finding Families
Family F1 — Authentication Collapse (RC-1)
A-01, A-02, A-03, A-04, A-05, A-06, C-01, C-02, C-19, D-01, J-03, M-01, K-01, K-02

These are not independent bugs. They are 14 symptoms of one root cause: there was never a designed authentication architecture. Each was added independently and each fails independently. Fixing any one in isolation (e.g., adding password validation to login) does not close the others.

Family F2 — Configuration-as-Code Absent (RC-2)
A-03, C-01, C-02, C-05, C-06, C-14, C-15, G-01, G-02, G-03, M-03, L-01, L-02, L-03, L-04, L-05

These are all consequences of never separating development configuration from deployment configuration, never enforcing that placeholder values cannot be used in production, and never treating secrets as secrets.

Family F3 — Input Boundary Absent (RC-3)
C-03, C-11, C-15, C-16, C-17, C-18, I-01, M-02

Family F4 — Concurrency Without Ownership (RC-4)
D-02, A-07, A-08, M-06, J-04, B-04

Family F5 — Storage Duplication and Divergence (RC-5)
D-06, D-07, I-02, I-03, I-05, M-07, M-08, B-01, F-03

Family F6 — Supply Chain and Runtime Trust Absent (RC-6)
C-03, C-04, C-06, H-01, H-02, H-03, H-04, H-05, L-01, L-02, L-03

Family F7 — No Test Safety Net (RC-7)
K-01, K-02, K-03, K-04

C. Cascading Failure Chains
Chain C1 — The Full Authentication Collapse (F1)


A-03 (auth bypass enabled) 
  → All @require_auth routes are open
  → J-03 (25 unauthenticated routes) redundantly confirmed open
  → A-01 (passwordless JWT login) provides Admin JWT for JWT-protected routes
  → D-01 (three disconnected systems) means no single fix closes all vectors
  → M-01 (multi-system privilege escalation) provides residual access even if one system is fixed
Chain C2 — Detection Evasion via FP Suppression


J-03 (FP suppression endpoint unauthenticated)
  → M-02 (attacker suppresses own IP)
  → All engines return "normal" for that IP
  → No alerts generated, no risk score computed
  → No prevention pipeline triggered
  → M-03 (prevention in dry_run anyway) — prevention would have been a no-op
  → Result: attacker is completely invisible AND protected by redundant failure
Chain C3 — Arbitrary Code Execution via Model Substitution


C-04 (entire repo mounted in container)
  → Any LFI/path traversal or container escape exposes model files
  → C-03 (no integrity check on joblib.load)
  → Attacker replaces .pkl file with malicious pickle payload
  → Next model load (startup or hot-reload) executes arbitrary code
  → L-01 (container runs as root) — execution is root-level
  → C-04 (volume mount to host) — root-in-container → potential host escape
Chain C4 — Redis Failure → Split Brain → Duplicate Enforcement


Redis unavailable
  → D-08 (fail-open leader election) — all instances become leader
  → All instances run prevention scheduler
  → M-04 (TOCTOU in idempotency check) — duplicate blocks race through
  → Multiple nftables/UFW rules for same IP created
  → Audit log flooded with duplicate events
  → Reconcile() sees mismatch, marks blocks as DESYNCED
  → Remediation actions compound the problem
Chain C5 — Mock Threat Intel + Auto-Block = Self-Inflicted DoS


C-14 (RFC-1918 IPs in threat intel)
  → Internal host sends traffic, TI engine flags as known malicious
  → Risk score elevated to block threshold
  → M-03 (but dry_run=True by default) — prevented in default config
  → If operator enables auto_block:
  → M-05 (mock TI + auto-block) — internal IPs auto-blocked
  → Internal infrastructure blocked, INIDS itself potentially blocked
Chain C6 — Secret Key Compromise → Session Forgery → Privilege Escalation


C-01 (SECRET_KEY=change-me-now committed to repo)
  → Flask sessions signable by any party with repo access
  → CSRF tokens predictable (derived from same key)
  → C-19 (JWT uses HS256 with same secret) — JWT tokens forgeable
  → Any service with the repo can generate admin JWT
  → A-01 (passwordless login) not even needed — direct token forgery possible
  → M-01 (three auth systems) — all three are compromised by the same key
Finding where fixing one in isolation worsens another:

Fixing A-03 (disabling ALLOW_UNAUTHENTICATED) without fixing J-03 (unauthenticated endpoints): the 25 undecorated routes remain open, giving false confidence that auth is now enforced
Fixing M-03 (enabling live blocking) without fixing C-14 (mock threat intel): immediately triggers self-inflicted DoS on internal IPs
Fixing D-08 (fail-closed leader election) without fixing M-04 (TOCTOU): if Redis fails and all instances stop acting, prevention completely ceases — a different attack surface
D. Attack Chain Analysis
ATTACK CHAIN AC-1 — Zero-Credential Full Administrative Takeover
Severity: Critical

Step	Action	Finding Used
1	Connect to /api/auth/login with {"username": "admin"}	A-01
2	Receive valid JWT for "admin"	A-01
3	Use JWT on all JWT-protected admin endpoints	A-01 + D-01
4	Call FP suppression endpoint to suppress own source IP	J-03 + M-02
5	All detection of attacker's IP ceases	M-02
6	Simultaneously use replace-admin-key for API-key endpoints	C-02
7	Call policy endpoint to disable dry_run	D-02 (race)
Entry point: Any network path to the HTTP API

Minimum attacker capability: HTTP client — zero credentials required

Final impact: Full administrative control, detection evasion, ability to enable/disable prevention

ATTACK CHAIN AC-2 — Unauthenticated Persistent Backdoor via Malicious Model
Severity: Critical

Step	Action	Finding Used
1	Use replace-admin-key or unauthenticated path to access file upload or model update endpoint	C-02 + J-03
2	Write malicious .pkl file to model directory (or replace via volume mount exploit)	C-04
3	Wait for application restart or trigger model hot-reload	—
4	joblib.load() executes payload at root privilege	C-03 + L-01
5	Reverse shell or persistence mechanism establishes from within container	—
6	Volume mount ../../:/app exposes host filesystem	C-04
Entry point: Any API access + filesystem write capability

Minimum attacker capability: Network access + knowledge of placeholder API key

Final impact: Remote code execution as root, potential host escape

ATTACK CHAIN AC-3 — Supply Chain Compromise at Container Boot
Severity: Critical

Step	Action	Finding Used
1	Compromise PyPI package or intercept DNS/TLS for PyPI	H-05
2	Container starts → pip install -r requirements.txt fetches packages	C-06
3	Malicious package executes on install	C-06 + H-05
4	Payload runs as root in container with full repo mounted	L-01 + C-04
Entry point: Container restart (including crash-loop restarts)

Minimum attacker capability: PyPI supply chain position (or MITM at boot time)

Final impact: Full system compromise on every restart

ATTACK CHAIN AC-4 — Unauthenticated Detection Blind Spot Creation
Severity: High

Step	Action	Finding Used
1	POST to /api/fp-suppressions with attacker's source IP	J-03
2	All detection engines return "normal" for that IP	M-02
3	Conduct attack from suppressed IP	—
4	No alerts generated, no prevention triggered	M-03 (redundant)
Minimum attacker capability: HTTP client to INIDS API

Final impact: Complete detection evasion, zero-trace attack capability

E. Duplication & Conflict Inventory
Duplicated System	Instance A	Instance B	Canonical Choice
Alert persistence	InMemoryAlertStore	OpsStore.alerts	OpsStore — remove in-memory store
Action tracking	InMemoryPreventionStore	OpsStore.actions	OpsStore — remove in-memory store
Blocked IP state	Adapter blocked_targets dict	OpsStore.actions	OpsStore — adapters should be stateless
Rate limiting	RateLimitMiddleware (WSGI)	InMemoryRateLimiter (Flask)	Choose one; unify state
Authentication	auth_service.py	auth_jwt.py	Design unified system; migrate
RBAC	rbac_manager.py / inids_rbac.db	Role in Principal (auth_service)	Unified RBAC against OpsStore
Circuit breaker	ActionExecutor._cb_*	CircuitBreaker in production_hardening.py	ActionExecutor version (active); remove dead code version
Conflicts:

Conflict	Component A	Component B	Effect
ALLOW_UNAUTHENTICATED=true vs. security intent	.env	auth_service.py design	Authentication universally bypassed
dry_run=True default vs. prevention intent	PolicyConfig	ActionExecutor	Prevention never executes by default
Mock RFC-1918 TI vs. production safety	load_threat_intel()	real network traffic	Internal IPs flagged as malicious
list_rules() semantics vs. reconcile() assumptions	Stateless adapters (webhook/mock)	ActionExecutor.reconcile()	Spurious DESYNCED state
F. Vulnerability Amplification Matrix
Pair	Individually	Combined	Amplified Impact
A-03 (auth bypass) + J-03 (25 unauth routes)	High each	Critical	Every API endpoint open — bypass renders decorators on remaining endpoints irrelevant
C-01 (weak secret) + C-19 (HS256 downgrade)	High each	Critical	Known secret + symmetric algo = admin JWT forgeable by anyone
M-02 (unauth FP suppression) + M-03 (dry_run default)	High each	Critical	Detection silenced + prevention disabled = zero-friction persistent attack
C-03 (unsafe joblib) + C-04 (repo mounted) + L-01 (runs as root)	High each	Critical	Malicious model → root code execution → host filesystem access
C-06 (pip at boot) + H-05 (no hash verify) + L-01 (root)	High each	Critical	Supply chain compromise → root execution on every restart
D-08 (fail-open leader) + M-04 (TOCTOU)	Medium each	High	Redis outage → split brain → duplicate enforcement → inconsistent firewall state
A-08 (FIFO eviction) + B-04 (O(n) count)	Medium each	High	Long-lived attacker evicted from rate tracker AND counter is slow at scale — rate limiting collapses under sustained attack
C-14 (mock TI) + auto_block enabled	Medium + High	Critical	Enabling prevention causes self-DoS on internal infrastructure
C-11 (log injection) + J-05 (ContextVar bleed)	Medium each	High	Attacker injects forged log entries + context bleeding makes attribution impossible
PHASE 3 — RISK CLASSIFICATION & PRIORITIZATION
TIER 0 — EMERGENCY (Address within 24 hours, formalize within 72 hours)
T0-1: Authentication bypass active in committed configuration

Findings: A-03, C-01, C-02

Justification: ALLOW_UNAUTHENTICATED=true is committed. Every deployment using this .env file has zero authentication. The placeholder API keys are known to anyone with repo access. Combined with A-01 (passwordless JWT login), there is no authentication barrier on any endpoint.

30-day consequence if unfixed: Every deployment is fully open. This is not theoretical — it is the committed default state.

Elevated by: D-01 (all three auth systems are simultaneously bypassed)

T0-2: Passwordless JWT token issuance

Findings: A-01, A-02

Justification: Any caller can obtain a valid admin JWT token with a single HTTP request. Token refresh accepts expired tokens, making token lifetime infinite. Combined with C-01/C-19 (HS256 + known secret), tokens can also be forged directly.

30-day consequence: Complete JWT-protected endpoint compromise.

Elevated by: C-01 + C-19 (token forgery possible without even calling login)

T0-3: ~25 endpoints accessible without any authentication

Findings: J-03, M-02

Justification: FP suppression endpoint is unauthenticated — any caller can silence the entire detection system for any IP. This is a functional kill switch for INIDS's core purpose. The other 24 endpoints expose configuration, forensic data, and control surfaces.

30-day consequence: Active exploitation requires only HTTP access. An attacker with network access can disable INIDS detection and become invisible.

T0-4: Unsafe model deserialization

Findings: C-03

Justification: joblib.load() of unsigned .pkl files is arbitrary code execution waiting for a trigger. Combined with C-04 (repo mount) and L-01 (root), this is a critical RCE path requiring only write access to the model directory.

30-day consequence: Any attacker with model file write access achieves root-level RCE on container startup.

T0-5: Entire repository mounted into running container as root

Findings: C-04, L-01

Justification: The deployment configuration creates conditions where any application-level LFI or code execution reads/writes host filesystem as root. This is not a theoretical risk — the .env, SQLite databases, and model files are all in the mounted path.

30-day consequence: Every application vulnerability now has host-level impact.

T0-6: pip install at container boot without hash verification

Findings: C-06, H-05

Justification: Package resolution at boot is a supply chain attack surface that activates on every restart. A crash-loop restart is an automatic supply chain compromise trigger. No hash verification means package substitution is undetectable.

30-day consequence: Compromise of any package in requirements.txt or PyPI infrastructure → mass deployment compromise on next restart.

T0-7: Mock threat intelligence targeting RFC-1918 space in production initialization

Findings: C-14, M-05

Justification: Enabling auto_block mode (the intended production mode) with default threat intel immediately blocks internal infrastructure. This is a production-critical operational risk that will be triggered the moment an operator enables prevention mode.

30-day consequence: Any production enablement of auto_block causes immediate self-inflicted DoS.

TIER 1 — PRODUCTION-BLOCKING (Fix before any new features)
T1-1: Three disconnected authentication systems (D-01, M-01) — architectural failure enabling residual access after partial fixes

T1-2: PolicyConfig race condition (D-02) — unsynchronized shared mutable state in prevention pipeline

T1-3: Two independent rate limiters (D-03) — unpredictable, inconsistent enforcement

T1-4: Prevention defaults to dry_run=True (M-03) — prevention system non-functional by default

T1-5: Docker Compose uses .env.example not .env (C-05) — Docker deployments use wrong secrets

T1-6: Sensor key mapped to analyst role (A-06) — privilege over-grant at sensor level

T1-7: CSP includes unsafe-inline (C-07) — XSS protection nullified

T1-8: CORS middleware never registered (C-08) — no CORS enforcement

T1-9: JWT algorithm downgrade HS256 (C-19) — symmetric signing with known key

T1-10: MLEngine re-raises exceptions (J-01) — model failures crash detection pipeline

T1-11: AnomalyEngine model read-write race (M-06) — concurrent use of partially-written model

T1-12: Leader election fails open (D-08, M-04) — split-brain on Redis failure

T1-13: Unconditional UPDATE migrations on startup (B-01) — startup latency, write lock DoS on large tables

T1-14: SECRET_KEY validation accepts placeholder values (G-02) — weak key deployed without warning

T1-15: Log injection via X-Correlation-ID (C-11) — audit trail corruption

T1-16: WebhookFirewallAdapter loses state on restart (I-02, M-07) — reconciliation produces spurious DESYNCED

T1-17: Duplicate alert/action stores diverging (D-06, D-07, M-08) — inconsistent data across stores

TIER 2 — ARCHITECTURAL DEBT (Near-term sprint)
T2-1: web_app/app.py god file — 4500+ lines (F-01)

T2-2: production_hardening.py entirely dead code (D-05)

T2-3: input_sanitizer.py SQL_KEYWORDS unused, UUID rejection, incomplete XSS patterns (C-16, C-17, C-18)

T2-4: Alert ID collision risk (truncated UUID) (I-03)

T2-5: _ops_probe writes audit row on every health check (B-07)

T2-6: InMemoryAlertStore drop log attributes wrong alert (J-06)

T2-7: AnomalyEngine silently truncates/pads feature vectors (E-04)

T2-8: IP blocklist whitelist 'localhost' non-functional (C-09)

T2-9: Schema migration version gating absent (F-03)

T2-10: NftablesFirewallAdapter text-parsing for handle numbers (I-04)

T2-11: AuditLogMiddleware uses timezone-naive datetime.utcnow() (C-12)

T2-12: No pagination in list_alerts (I-05)

T2-13: rbac_manager.py AuditLog ID collision under load (F-02)

T2-14: DataFrame created per request in ML path (B-02)

T2-15: New ThreadPoolExecutor per adapter call (B-03)

T2-16: _RateCounter O(n) sliding window under lock (B-04)

TIER 3 — QUALITY & HARDENING (Planned improvements)
T3-1: explain_features deviation distance is not feature importance (E-05)

T3-2: HoneypotEngine returns confidence=0.0 for normal verdict (E-03)

T3-3: HONEYPOT_ENABLED env var has no effect (G-03)

T3-4: api_health exposes OPS_DB_PATH (E-01)

T3-5: system_uptime hardcoded as "4.2h" (E-02)

T3-6: _fetchall private method called from app.py (D-04)

T3-7: Dependency version security audit (H-01, H-02, H-03, H-04)

T3-8: ContextVar bleed under Gevent (J-05)

T3-9: SecurityHardeningManager.enforce_rate_limit never resets (A-12)

T3-10: detect suspiciousflag logic inversion in DetectionService (A-13)   **T3-11:**WebhookFirewallAdapter` no TLS enforcement (C-10)

TIER 4 — LOW PRIORITY (Tracked, not urgent)
T4-1: Integer division in leader election TTL (A-14)

T4-2: _RateCounter now = now or time.monotonic() falsy-zero bug (A-07)

T4-3: PerformanceOptimizer averaging formula uses (a+b)/2 not true mean (G-04)

T4-4: SecurityHardeningManager.audit_logs and metrics unbounded growth (B-05, B-06) — in dead code module

T4-5: AnomalyEngine.evaluate() called without is_ready() guard (A-10)

Findings elevated by correlation:

T0-7 (C-14) appears Medium in isolation but is Tier 0 because enabling the intended production mode (auto_block) causes immediate self-DoS
T0-3 (J-03) appears as a general "missing auth" issue but is Tier 0 because it includes the FP suppression endpoint — the detection kill switch
H-05 (no hash verify) appears Low in isolation but is Tier 0 in context of C-06 (pip at boot)
PHASE 4 — SOLUTION ARCHITECTURE DESIGN
SOLUTION 1 — UNIFIED AUTHENTICATION SYSTEM
What it replaces
src/auth_service.py, src/auth_jwt.py, and the identity portion of src/rbac_manager.py. The three-system architecture is replaced with one.

Design
Identity Model:


User {
    user_id: UUID            # stable, never reused
    username: str            # unique, immutable
    credential_hash: str     # bcrypt hash of password
    roles: list[Role]        # FK to canonical role table
    api_keys: list[APIKey]   # hashed, rotatable
    is_active: bool
    created_at: datetime
    last_login: datetime
}

APIKey {
    key_id: UUID
    key_hash: str            # SHA-256 of actual key — actual key never stored
    user_id: UUID FK
    role: Role               # explicit role for this key (may differ from user default)
    expires_at: datetime | None
    last_used_at: datetime
    is_revoked: bool
}
Token Format (JWT, asymmetric — RS256 or ES256 only):


{
  "sub": "<user_id UUID>",
  "username": "<username>",
  "roles": ["<role>"],
  "jti": "<token UUID>",       // unique per token — enables revocation
  "iss": "inids",
  "aud": "inids-api",
  "iat": <epoch>,
  "exp": <iat + 3600>          // 1-hour hard expiry, non-negotiable
}
Key Management:

RS256 private key: generated at deployment, injected via Docker secret or Vault — never in env var, never committed
Public key: available to all services that need to verify tokens
SECRET_KEY variable eliminated for JWT — separate Flask session secret (strong random, 32+ bytes)
Rotation: new key pair generated, old public key retained for verification during overlap window (max 1 token lifetime = 1 hour), then removed
Session/Token Lifecycle:

Tokens issued only after: username lookup (exists, is_active), credential verification (bcrypt compare)
Refresh: requires valid, non-expired token + valid jti not in revocation list
Revocation: jti stored in OpsStore.revoked_tokens table with expires_at equal to token exp (self-cleaning)
Fail-closed: missing JWT manager → 401, not 500; missing claims → 403; expired token → 401; revoked jti → 401
API Key Path:

API keys hashed with SHA-256 on receipt, only hash stored
Key lookup: SELECT user_id, role FROM api_keys WHERE key_hash = ? AND is_revoked = 0 AND (expires_at IS NULL OR expires_at > NOW())
Role derived from the key's stored role, not from the user's default role
ALLOW_UNAUTHENTICATED variable: eliminated entirely — no bypass path in production code
RBAC Integration:

AuthContext object produced by unified auth carries: user_id, username, roles list
Same AuthContext type regardless of auth method (API key or JWT)
All authorization decisions use AuthContext.roles — one enforcement point
Migration Path:

Add users, api_keys, revoked_tokens tables to OpsStore schema (additive)
Create service accounts for existing placeholder keys with proper role assignments
Implement new AuthService against new schema
Add compatibility shim: existing env-var API keys are hashed and matched against api_keys table during transition
Remove auth_service.py, auth_jwt.py after all routes migrated
Migrate inids_rbac.db role/permission data into OpsStore
SOLUTION 2 — CENTRALIZED AUTHORIZATION MODEL
Design
Single Enforcement Layer: Flask before_request decorator + route metadata


# Decorators declare REQUIRED roles as metadata on the route function
@app.route("/api/fp-suppressions", methods=["POST"])
@require_roles("admin", "analyst")   # single decorator, unified auth
def api_fp_suppression_create():
    ...
require_roles(*roles) decorator:

Extracts bearer token or API key from request (one extraction path)
Validates against unified AuthService
Produces AuthContext or raises AuthError (401/403 — never 500)
Checks AuthContext.roles against the declared required roles
Attaches AuthContext to flask.g.auth (not request — avoids attribute pollution)
On failure: returns JSON {"error": "unauthorized"} with appropriate HTTP status
Role Schema (canonical, single definition):


admin     — full system access
analyst   — read alerts, manage FP suppressions, query data
operator  — acknowledge/close alerts, view actions
sensor    — submit traffic features only (no read access to alerts or config)
viewer    — read-only access to alerts and dashboards (no config)
Audit Trail:

Every authorization decision (allow and deny) logged to OpsStore.audits with: user_id, route, method, roles_required, roles_present, decision, timestamp
Missing/Ambiguous Role Handling:

No role on AuthContext → 403, logged as authorization_denied
Role not in allowed set → 403, logged
No @require_roles decorator on a route → startup validation fails, application does not start (see Solution 3)
Migration Path:

Audit every route for intended access level
Apply @require_roles to all 25 currently undecorated routes
Startup validation loop: iterate all registered routes, assert each has either @require_roles or is explicitly marked @public_route (no-auth whitelist, used only for /health, login endpoints)
Any route missing both decorators raises RuntimeError at startup
SOLUTION 3 — HARDENED CONFIGURATION MANAGEMENT
Design
Principle: Configuration has two kinds of values — secrets and settings. They must never be in the same file and must never be committed to version control.

Settings (non-secret, committed):


# config/settings.toml  — committed to repo
[app]
debug = false
log_level = "INFO"
prevention_mode = "monitor"    # not dry_run — explicit mode name
block_ttl_seconds = 300

[detection]
anomaly_buffer_size = 3000
threshold_window_seconds = 60
connection_rate_limit = 200

[storage]
db_path = "/data/inids_ops.db"    # override via env in container
Secrets (never committed — injected at runtime):


INIDS_SECRET_KEY          — Flask session secret (32+ random bytes, hex or base64)
INIDS_JWT_PRIVATE_KEY     — RS256 PEM key (or path to file)
INIDS_JWT_PUBLIC_KEY      — RS256 PEM key (or path to file)
INIDS_ADMIN_API_KEY       — hashed at startup, original never logged
INIDS_SENSOR_API_KEY      — hashed at startup, original never logged
INIDS_VIEWER_API_KEY      — hashed at startup, original never logged
INIDS_DB_URL              — connection string (may contain password)
Startup Validation (fail-closed on all violations):


REQUIRED_SECRETS = [
    "INIDS_SECRET_KEY",
    "INIDS_JWT_PRIVATE_KEY",
    "INIDS_ADMIN_API_KEY",
]

FORBIDDEN_VALUES = {
    "change-me-now", "replace-admin-key", "replace-sensor-key",
    "replace-viewer-key", "secret", "password", "changeme", "default"
}

def validate_config_at_startup(settings: Settings) -> None:
    for key in REQUIRED_SECRETS:
        value = os.environ.get(key, "")
        if not value:
            raise RuntimeError(f"Required secret {key} is not set. Cannot start.")
        if value.lower() in FORBIDDEN_VALUES:
            raise RuntimeError(f"{key} contains a placeholder value. Cannot start.")
        if len(value) < 32 and key in {"INIDS_SECRET_KEY"}:
            raise RuntimeError(f"{key} is too short (minimum 32 characters).")
    
    # Verify route coverage (see Solution 2)
    validate_all_routes_have_auth_decorator(app)
.env / .env.example Disposition:

.env: added to .gitignore and .dockerignore
.env.example: contains only setting names with empty values and comments — no values whatsoever
Docker Compose: reads secrets from Docker secrets (/run/secrets/) or external vault reference, never from an env file in the repository
ALLOW_UNAUTHENTICATED: Variable is permanently removed. No code path accepts it. If found in environment at startup, log a CRITICAL warning and refuse to start.

SOLUTION 4 — RELIABLE CONCURRENCY & STATE MANAGEMENT
PolicyConfig Race Condition
Design: Replace mutable @dataclass with an immutable snapshot pattern:


from threading import RLock
from dataclasses import replace

class PolicyConfigManager:
    def __init__(self, initial: PolicyConfig):
        self._config = initial
        self._lock = RLock()
    
    def get(self) -> PolicyConfig:
        with self._lock:
            return self._config          # frozen dataclass — safe to read without lock after retrieval
    
    def update(self, **kwargs) -> PolicyConfig:
        with self._lock:
            self._config = replace(self._config, **kwargs)
            return self._config

@dataclass(frozen=True)
class PolicyConfig:
    mode: str = "monitor"
    dry_run: bool = True
    ...
All reads call config_manager.get() which returns an immutable snapshot. Updates replace the entire config atomically. No partial write is visible to concurrent readers.

AnomalyEngine Model Read-Write Race
Design: Replace direct self._model access with atomic reference swap:


import threading

class AnomalyEngine:
    def __init__(self):
        self._model_ref = None           # atomic reference
        self._model_lock = threading.Lock()
    
    def _set_model(self, model):
        with self._model_lock:
            self._model_ref = model      # single assignment — atomic on CPython
    
    def _get_model(self):
        return self._model_ref           # single read — atomic on CPython
    
    def evaluate(self, features):
        model = self._get_model()        # capture snapshot
        if model is None:
            return EngineResult(verdict="unknown", ...)
        # use `model` local variable for all operations — never self._model_ref
Note: CPython's GIL makes single reference assignment atomic, but this pattern is explicit and portable. For non-CPython, wrap the reference in a threading.local or use a RLock.

Leader Election Fail-Closed
Design: Replace fail-open with fail-closed:


def is_leader(self) -> bool:
    try:
        result = self._redis.set(
            self._key, self._instance_id,
            nx=True, ex=self._ttl
        )
        return result is True or self._redis.get(self._key) == self._instance_id
    except RedisError:
        self.logger.error("leader_election_redis_failure: assuming NOT leader (fail-closed)")
        return False    # FAIL CLOSED: no Redis = no prevention actions
Consequence: When Redis fails, prevention scheduler pauses. This is preferable to split-brain. Alert operators to Redis failure via metrics/alerting.

Prevention Action Idempotency
Design: Remove TOCTOU window by using a database-level unique constraint:


ALTER TABLE actions 
ADD CONSTRAINT uq_active_block_target 
UNIQUE (target, status) 
WHERE status IN ('active', 'enforced', 'executed') AND action_type IN ('block', 'temp_block');
The INSERT itself becomes the idempotency check. A duplicate block attempt fails at the database constraint level, not in application code. No read-check-write window.

SOLUTION 5 — SECURE SECRET HANDLING
Injection Model (Docker):


# docker-compose.yml
services:
  inids:
    secrets:
      - inids_secret_key
      - inids_jwt_private_key
      - inids_admin_api_key
    environment:
      - INIDS_SECRET_KEY_FILE=/run/secrets/inids_secret_key
      - INIDS_JWT_PRIVATE_KEY_FILE=/run/secrets/inids_jwt_private_key

secrets:
  inids_secret_key:
    external: true     # managed by Docker secrets, not in compose file
File-based Secret Loading:


def load_secret(env_var: str) -> str:
    file_path = os.environ.get(f"{env_var}_FILE")
    if file_path:
        with open(file_path, "r") as f:
            return f.read().strip()
    direct = os.environ.get(env_var, "")
    if not direct:
        raise RuntimeError(f"Secret {env_var} not found via env or file")
    return direct
What Must Never Appear in Logs:

API keys (even hashed) — log key_id only
JWT tokens — log jti only
Connection strings with passwords — log sanitized version
SECRET_KEY value — never logged
Audit Trail for Secret Access:

Every API key usage → OpsStore.audits with key_id (not key value), timestamp, endpoint
JWT issuance → audit record with jti, user_id, source IP
Failed auth attempts → audit record with source IP and reason
.gitignore additions:


.env
*.pem
*.key
*.p12
*.pfx
models/*.pkl     # model files should not be in VCS
data/*.db        # database files
SOLUTION 6 — PREVENTION ORCHESTRATION HARDENING
Single Source of Truth: OpsStore is the exclusive owner of prevention action state. InMemoryPreventionStore is eliminated.

Adapter State Model: All adapters are treated as stateless appliers — they apply or remove rules and can enumerate current rules, but they do not maintain their own state map. WebhookFirewallAdapter must query the external service (or acknowledge that list_rules() is not supported) rather than maintaining an in-memory dict.


class WebhookFirewallAdapter(FirewallAdapter):
    def list_rules(self) -> list[str]:
        # Option A: query external service for current block list
        # Option B: return [] and mark adapter as "stateless" — reconcile() skips stateless adapters
        raise NotImplementedError("WebhookFirewallAdapter does not support rule enumeration")
reconcile() must check adapter.supports_list_rules() before comparing:


def reconcile(self) -> dict:
    if not self.adapter.supports_list_rules():
        return {"skipped": True, "reason": "adapter_stateless"}
    ...
Blocking Atomicity:

block_ip() must be idempotent at the OS adapter level (UFW/nftables) — adding the same rule twice must not error
ActionExecutor inserts into OpsStore with the unique constraint (Solution 4) — duplicate from split-brain is rejected at DB level
Unblock is also idempotent: if rule doesn't exist, return success
ThreadPoolExecutor Reuse:

ActionExecutor maintains a single ThreadPoolExecutor(max_workers=4) at instance level, not per-call:

self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="adapter_worker")
Default Policy:

mode: must be explicitly configured — no default that silently disables prevention
dry_run: default True in dev environments only; startup validation checks INIDS_ENVIRONMENT and raises if production environment has dry_run=True
SOLUTION 7 — ML PIPELINE STABILITY
Model Integrity Verification:


def load_model_with_verification(path: Path, expected_sha256: str) -> Any:
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise SecurityError(
            f"Model file {path} failed integrity check. "
            f"Expected {expected_sha256}, got {digest}. "
            "Refusing to load potentially malicious model."
        )
    return joblib.load(path)
Model checksums stored in a separate models/checksums.sha256 file (committed to repo, models are not). On deployment, models are downloaded from a trusted artifact store (not the container image or mounted volume), verified, then loaded.

Graceful Degradation (fail-safe):


def evaluate(self, features):
    try:
        pred = int(self._model.predict(df)[0])
        proba = self._model.predict_proba(df)[0]
    except Exception as exc:
        self.logger.error("ml_inference_failed engine=%s error=%s", self._engine_id, exc)
        return EngineResult(
            verdict="unknown",
            confidence=0.0,
            severity="low",
            metadata={"error": "inference_failure", "fallback": True}
        )
        # Do NOT re-raise — degrade to unknown, let other engines vote
Model Versioning:


models/
  primary_v1.pkl          # never loaded directly
  primary_v1.sha256       # checksum for verification
  anomaly_v2.pkl
  anomaly_v2.sha256
  ACTIVE_MODELS           # simple text file: "primary=v1\nanomalY=v2"
Hot Reload:

Model reload is triggered via an authenticated API call (POST /api/models/reload, @require_roles("admin"))
New model loaded to a staging slot, verified against checksum
Atomic swap: self._model_ref = new_model (atomic reference assignment per Solution 4)
Old model held for one TTL window before GC
SOLUTION 8 — ALERTING & STORAGE ARCHITECTURE
Elimination of Duplication:

InMemoryAlertStore is removed. OpsStore is the single alert backend.
InMemoryPreventionStore is removed. OpsStore.actions is the single action backend.
The DetectionService.alert_store parameter is removed; DetectionService writes directly to OpsStore.
Alert ID Generation:


alert_id = f"al_{uuid.uuid4().hex}"   # full 128-bit UUID hex, no truncation
Deduplication Strategy:

Deduplication window: alerts for the same (source_ip, attack_type) within a 60-second window are suppressed (one alert emitted, subsequent increments a count field)
Implemented as: SELECT id FROM alerts WHERE source_ip=? AND attack_type=? AND timestamp > ? with index on (source_ip, attack_type, timestamp)
Pagination:


def list_alerts(self, limit: int = 50, offset: int = 0, ...) -> tuple[list[dict], int]:
    # Returns (rows, total_count_for_pagination)
Retention:

audits table: 90-day rolling delete (configurable via settings)
alerts table: 365-day retention, configurable
Cleanup job runs in prevention scheduler (leader-gated, once per day)
_ops_probe in Health Check:

Remove the audit INSERT from _ops_probe
Validate DB connectivity by a read-only query: SELECT 1 FROM schema_version LIMIT 1
SOLUTION 9 — RATE LIMITING UNIFICATION
Single Rate Limiter Design:

Remove RateLimitMiddleware (WSGI layer) and the InMemoryRateLimiter in _before_request_metrics.

Replace with a single RateLimiter registered as a Flask before_request hook:


class RateLimiter:
    """Unified rate limiter with per-IP, per-user, and per-route limits."""
    
    def __init__(self, store: RateLimitStore):
        self._store = store    # Redis-backed in production, in-memory for dev
    
    def check(self, identifier: str, limit: int, window_s: int) -> RateLimitResult:
        ...
    
    def limit(self, per_minute: int = 100, per_hour: int | None = None):
        """Decorator for per-route limits."""
        def decorator(f):
            f._rate_limit = RateLimitSpec(per_minute=per_minute, per_hour=per_hour)
            return f
        return decorator
Enforcement Point: Single before_request hook, after auth (so user identity is available for per-user limits):


# Applied in order:
# 1. IP-based: 1000/minute global hard limit (DDoS protection)
# 2. User-based: 200/minute for authenticated users
# 3. Route-based: per-route decorator overrides if more restrictive
State: Redis-backed sliding window counters for multi-instance deployments. Falls back to in-memory with a warning log if Redis is unavailable (acceptable degradation — rate limiting is best-effort under Redis failure).

SOLUTION 10 — DEPLOYMENT & RUNTIME HARDENING
Container Build (Dockerfile — image baked, not runtime-installed):


FROM python:3.11-slim

# Create non-root user
RUN groupadd -r inids && useradd -r -g inids inids

WORKDIR /app

# Copy only application code (NOT repo root)
COPY src/ ./src/
COPY web_app/ ./web_app/
COPY rules/ ./rules/
COPY requirements.txt .

# Install dependencies at build time with hash verification
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

# Model files NOT baked into image — downloaded at startup from trusted artifact store
# Database files NOT in image — mounted from persistent volume

USER inids

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "-b", "0.0.0.0:5000", "web_app.app:app"]
Docker Compose (production-hardened):


version: "3.8"
services:
  inids:
    build: .
    user: "inids:inids"
    read_only: true
    tmpfs:
      - /tmp
    volumes:
      - inids-data:/data            # only data directory, not source tree
      - inids-models:/models        # model files only
    secrets:
      - inids_secret_key
      - inids_jwt_private_key
      - inids_admin_api_key
    environment:
      - INIDS_SECRET_KEY_FILE=/run/secrets/inids_secret_key
      - INIDS_ENVIRONMENT=production
      - INIDS_DB_URL=sqlite:////data/inids_ops.db
    deploy:
      resources:
        limits:
          memory: 2g
          cpus: "2.0"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s
Startup Ordering:

Secrets injected via Docker secrets mechanism
validate_config_at_startup() — fails closed on missing/weak secrets, missing auth decorators, placeholder values
Model files verified (checksum) before load
OpsStore initialized with version-gated migrations (not unconditional UPDATE)
Threat intelligence loaded from external feed (no hardcoded mock indicators)
Prevention scheduler started only after leader election confirmed
PHASE 5 — REGRESSION PREVENTION STRATEGY
1. Integration Test Strategy
Mandatory test coverage gates (CI must pass before merge):

Every PR touching auth_service.py, auth_jwt.py, any route handler, or any middleware must pass:

tests/integration/test_auth_end_to_end.py — full auth flow for all roles
tests/integration/test_unauthorized_access.py — every route tested without credentials → must return 401/403
tests/integration/test_route_coverage.py — asserts every route has @require_roles or @public_route
Preserved behavior verification:

All existing API contracts (request/response schema) verified against snapshot tests
Alert creation → storage → retrieval round-trip
Prevention pipeline: detection → risk → policy → action → audit record
2. Security Regression Tests
For each attack chain:

AC-1 (zero-credential JWT takeover):


def test_login_requires_valid_credentials():
    resp = client.post("/api/auth/login", json={"username": "admin"})
    assert resp.status_code == 401    # no password = rejected

def test_login_with_wrong_password_rejected():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401

def test_expired_token_not_refreshable():
    token = create_expired_token("admin")
    resp = client.post("/api/auth/refresh", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
AC-4 (unauthenticated FP suppression):


def test_fp_suppression_requires_auth():
    resp = client.post("/api/fp-suppressions", json={"source_ip": "1.2.3.4"})
    assert resp.status_code in (401, 403)

def test_all_routes_require_auth():
    """Parametrized test: every non-public route returns 401 without auth."""
    for rule in app.url_map.iter_rules():
        if rule.endpoint in PUBLIC_ENDPOINTS:
            continue
        resp = client.get(str(rule))
        assert resp.status_code in (401, 403), f"Route {rule} is open without auth"
AC-2 (unsafe model loading):


def test_model_load_rejects_bad_checksum():
    with pytest.raises(SecurityError):
        load_model_with_verification(malicious_pkl_path, expected_sha256="deadbeef...")

def test_model_load_accepts_verified_model():
    model = load_model_with_verification(good_pkl_path, correct_sha256)
    assert model is not None
Placeholder config rejection:


def test_placeholder_secret_key_rejected():
    with patch.dict(os.environ, {"INIDS_SECRET_KEY": "change-me-now"}):
        with pytest.raises(RuntimeError, match="placeholder"):
            validate_config_at_startup(settings)
3. Concurrency & Race Condition Tests
PolicyConfig race:


def test_policy_config_concurrent_update_and_read():
    manager = PolicyConfigManager(PolicyConfig(dry_run=True))
    results = []
    
    def updater():
        for _ in range(1000):
            manager.update(dry_run=False)
            manager.update(dry_run=True)
    
    def reader():
        for _ in range(1000):
            cfg = manager.get()
            # dry_run must always be either True or False — never corrupted
            assert isinstance(cfg.dry_run, bool), f"Corrupted: {cfg.dry_run!r}"
            results.append(cfg.dry_run)
    
    threads = [Thread(target=updater), Thread(target=reader), Thread(target=reader)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    # No assertion errors above means no corruption
AnomalyEngine model swap:


def test_anomaly_engine_concurrent_fit_and_evaluate():
    engine = AnomalyEngine(buffer_size=10)
    engine.set_model(pretrained_model)
    
    errors = []
    def evaluator():
        for _ in range(500):
            try:
                engine.evaluate(sample_features)
            except Exception as e:
                errors.append(e)
    
    def fitter():
        for _ in range(5):
            engine.fit(np.random.rand(100, len(NUMERIC_FEATURES)))
    
    threads = [Thread(target=evaluator) for _ in range(4)] + [Thread(target=fitter)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, f"Concurrent access errors: {errors}"
4. Deployment Validation Gates
Pre-deployment checklist (automated, CI-enforced):


# 1. Container runs as non-root
docker inspect <image> | jq '.[0].Config.User' | grep -v root

# 2. No .env or secrets committed
git diff --name-only HEAD~1 | grep -E '\.(env|pem|key|p12)$' && exit 1

# 3. No placeholder values in any config file
grep -r "change-me-now\|replace-admin-key" . --include="*.yml" --include="*.toml" && exit 1

# 4. All routes have auth decorators (startup validation)
docker run --rm <image> python -c "from web_app.app import app; from src.auth import validate_route_coverage; validate_route_coverage(app)"

# 5. Model checksums present and valid
python scripts/verify_model_checksums.py

# 6. Requirements hash file present
grep '\-\-hash=' requirements.txt | wc -l | grep -v '^0$'
Automatic rollback triggers:

Health check fails 3 consecutive times within 90-second window → rollback
Any RuntimeError from validate_config_at_startup() → rollback (container exits non-zero)
Model integrity check failure → rollback
OpsStore migration failure (raises RuntimeError) → rollback
5. Migration Safety Checks
Auth System Migration:

Dual-write window: new AuthService validates against new schema; old API key env vars still accepted during transition (compatibility shim, 2-sprint window maximum)
Rollback: re-enable env-var auth path; new users table is additive — no data loss
Data corruption risk: none (additive schema changes only during migration)
OpsStore Schema Migration (Solution 8):

Migrations now version-gated: each migration has a version number; _verify_schema_version() applies only unapplied migrations in order
Rollback: down-migration script for each migration that added columns
Corruption risk: column ADD is reversible; data inserted in new columns is orphaned on rollback (acceptable — new columns are nullable)
Alert Store Consolidation (Solution 8):

InMemoryAlertStore drained to OpsStore before removal: drain_to_ops_store(alert_store, ops_store) called at startup if in-memory store has content
Rollback: re-add InMemoryAlertStore as a second write target (compatibility shim for one sprint)
6. Phased Rollout Plan
Feature-flaggable (zero-downtime):

Model integrity verification (configurable: INIDS_MODEL_VERIFY=strict|warn|disabled)
Alert deduplication window size
Rate limiter backend (in-memory vs. Redis)
Placeholder value rejection (configurable: INIDS_STRICT_CONFIG=true|false)
Requires maintenance window:

Auth system cutover (cannot run both systems simultaneously against the same sessions)
OpsStore schema migration with unique constraint addition (brief write lock in SQLite)
InMemoryAlertStore elimination (brief gap if in-flight alerts exist)
Hot-patchable (zero downtime, config change only):

Disabling ALLOW_UNAUTHENTICATED (env var change + restart)
Replacing placeholder API keys (env var change + restart)
Setting dry_run=false in PolicyConfig after validating threat intel is correct
Must be sequenced:

Fix threat intel (C-14) BEFORE enabling auto_block
Fix auth bypass (A-03) BEFORE auditing which endpoints are exposed
Unified auth system BEFORE removing old auth systems (compatibility window)
OpsStore unique constraint BEFORE removing idempotency check code
PHASE 6 — SURGICAL IMPLEMENTATION ROADMAP
PHASE A — Emergency Stabilization (0–72 hours)

┌─────────────────────────────────────────────────────────────────┐
│ STEP 1 — DISABLE AUTHENTICATION BYPASS                TIER: 0  │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  .env, src/auth_service.py, src/settings.py         │
│ ACTION:     1. Change .env: ALLOW_UNAUTHENTICATED=false        │
│             2. Add .env to .gitignore and .dockerignore         │
│             3. Add startup check: if ALLOW_UNAUTHENTICATED=true │
│                → raise RuntimeError("Cannot start. Auth bypass  │
│                  is enabled. Check ALLOW_UNAUTHENTICATED.")     │
│             4. Generate strong random keys (32+ bytes) for all  │
│                three API key env vars; rotate SECRET_KEY        │
│             5. Document new keys securely outside repo          │
│ WHY NOW:    This is the master bypass that disables everything  │
│             else. Every other auth fix is irrelevant while this │
│             is active. Must be first.                           │
│ DEPENDS ON: Nothing. Standalone change.                        │
│ ENABLES:    Steps 2-5 (auth fixes now have effect)             │
│ RISK:       Any service using the bypass flag breaks. Check all │
│             automated tests and scripts for ALLOW_UNAUTH=true.  │
│             Monitor logs for 401 errors after change.          │
│ ROLLBACK:   Revert .env change (immediate, no code change)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 2 — REMOVE HARDCODED MOCK THREAT INTEL           TIER: 0  │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  web_app/app.py — load_threat_intel()               │
│ ACTION:     Replace hardcoded mock indicators with empty-list   │
│             initialization. Add a config flag                   │
│             INIDS_TI_FEED_PATH pointing to an external TI file │
│             (CSV/STIX). If unset, TI engine starts with no     │
│             indicators (operational but non-alerting for TI).  │
│             Log a warning: "No threat intel feed configured."  │
│ WHY NOW:    Enabling auto_block (Step 3 prerequisite) with     │
│             current TI causes immediate self-DoS. Must fix TI   │
│             before any prevention mode changes.                │
│ DEPENDS ON: Nothing.                                           │
│ ENABLES:    Step 3 (safe to enable auto_block)                 │
│ RISK:       TI engine produces no matches until feed is        │
│             configured. This is safe — false negatives from TI │
│             are preferable to false positive self-blocking.    │
│ ROLLBACK:   Revert load_threat_intel() to empty list           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 3 — ADD AUTH TO ALL UNDECORATED ROUTES           TIER: 0  │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  web_app/app.py — ~25 unauthenticated route handlers│
│ ACTION:     Audit every route in app.py. For each route missing │
│             an auth decorator:                                  │
│             1. Determine correct required role per Solution 2  │
│             2. Apply @require_auth with appropriate role check  │
│             Priority order:                                     │
│             - FP suppression: analyst + admin only             │
│             - Policy config: admin only                        │
│             - Forensic timeline: analyst + admin               │
│             - Escalation tracker: analyst + operator + admin   │
│             - All others: viewer minimum                       │
│             3. Add startup validation that no unauthenticated  │
│             routes exist (except explicit whitelist)           │
│ WHY NOW:    Step 1 re-enables auth checks; Step 3 adds the     │
│             checks that were never there. FP suppression is an │
│             active attack vector until this step is done.      │
│ DEPENDS ON: Step 1 (auth must be enabled for decorators to    │
│             have effect)                                       │
│ ENABLES:    Steps 4, 5 (detection pipeline integrity restored) │
│ RISK:       Existing callers of these endpoints (scripts,      │
│             integrations) break. Audit all API consumers first.│
│ ROLLBACK:   Remove added decorators (git revert)              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 4 — BREAK JWT PASSWORDLESS LOGIN                 TIER: 0  │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  web_app/app.py — api_auth_login route              │
│ ACTION:     Temporarily: gate api_auth_login to only work for  │
│             known API key holders:                             │
│             1. Require a valid API key in the request          │
│             2. Derive username from the API key's principal    │
│             3. Issue JWT for that principal only               │
│             This is a bridge until Solution 1 (full user DB)  │
│             is implemented. Not the permanent solution — but it│
│             closes the critical gap immediately.               │
│             Fix api_auth_refresh: add 5-minute "fresh window"  │
│             check — only refresh tokens issued within last 5m  │
│             of expiry; reject already-expired tokens.         │
│ WHY NOW:    Passwordless JWT issuance is an open admin door.  │
│             Bridge fix takes hours; permanent fix takes days.  │
│ DEPENDS ON: Step 1 (API keys are now genuine after rotation)  │
│ ENABLES:    Authenticated JWT tokens carry real identity      │
│ RISK:       Any service using JWT login without API key breaks.│
│             Monitor /api/auth/login 401 rate.                  │
│ ROLLBACK:   Revert api_auth_login handler                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 5 — HARDEN DOCKER DEPLOYMENT BASICS             TIER: 0   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  deploy/compose/docker-compose.yml, Dockerfile      │
│ ACTION:     1. Add USER inids to Dockerfile (non-root)         │
│             2. Change volume mount from ../../:/app to:        │
│                - /data:/data (persistent storage only)         │
│                - /models:/models (model files only)            │
│                Remove source tree mount entirely               │
│             3. Move pip install from container command to      │
│                Dockerfile RUN layer (build-time only)          │
│             4. Change env_file from .env.example to reference  │
│                Docker secrets (or explicit env vars, no file)  │
│             5. Add memory limit: 2g and cpu limit: 2.0        │
│             6. Add healthcheck directive                       │
│ WHY NOW:    Volume mount + root = host-level blast radius for │
│             any exploit. Must close before model integrity      │
│             step (Step 6) as model substitution via mount is  │
│             the delivery vector.                               │
│ DEPENDS ON: Step 1 (secrets are outside repo, safe to stop    │
│             mounting repo root)                                │
│ ENABLES:    Steps 6, 7 (model security now meaningful)        │
│ RISK:       Container rebuild required. Validate app starts   │
│             correctly without source mount. Test all file      │
│             paths that previously referenced /app/...         │
│ ROLLBACK:   Revert docker-compose.yml and rebuild             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 6 — ADD MODEL INTEGRITY VERIFICATION            TIER: 0   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  web_app/app.py — load_models(); src/detection/     │
│             engines/anomaly_engine.py — model load path        │
│ ACTION:     1. Create scripts/generate_model_checksums.py:     │
│                iterates all .pkl files, writes SHA-256 to      │
│                models/checksums.sha256                         │
│             2. Run script against current known-good models    │
│             3. Commit checksums.sha256 to repo                 │
│             4. Modify load_models() and anomaly_engine joblib  │
│                load to call load_model_with_verification()     │
│                (per Solution 7 design)                         │
│             5. Startup fails if any model fails checksum check │
│ WHY NOW:    Step 5 removed the volume mount attack path for   │
│             model substitution. This step closes the remaining │
│             path (model files in /models volume or direct API).│
│ DEPENDS ON: Step 5 (model directory is now a controlled       │
│             volume, not the open source tree)                  │
│ ENABLES:    ML pipeline is safe to run                        │
│ RISK:       If models have been modified (legitimately or not) │
│             since last checksum generation, startup fails.     │
│             Verify all model checksums before deploying.      │
│ ROLLBACK:   Set INIDS_MODEL_VERIFY=warn (log only, no fail)   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 7 — GENERATE PINNED REQUIREMENTS WITH HASHES   TIER: 0   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  requirements.txt                                   │
│ ACTION:     1. In clean Python virtualenv, install all deps    │
│             2. pip-compile --generate-hashes requirements.in   │
│                → produces requirements.txt with --hash entries │
│             3. Update Dockerfile: pip install --require-hashes │
│             4. Commit new requirements.txt                     │
│             5. Review hash list against known CVEs in pinned  │
│                versions (cryptography, numpy, Werkzeug, scapy) │
│             6. Upgrade any packages with known CVEs (H-01      │
│                through H-04) — regenerate hashes after upgrade │
│ WHY NOW:    pip install is now at build time (Step 5). Hash    │
│             verification prevents supply chain compromise at   │
│             the one remaining package fetch point (image build)│
│ DEPENDS ON: Step 5 (pip install moved to build time)          │
│ ENABLES:    Supply chain integrity for all subsequent builds   │
│ RISK:       Hash generation requires network access to PyPI.  │
│             Do in clean environment. Validate app starts after │
│             package version upgrades (H-01 through H-04).     │
│ ROLLBACK:   Revert to previous requirements.txt (no hashes)   │
│             and rebuild image                                  │
└─────────────────────────────────────────────────────────────────┘
PHASE B — Architectural Foundations (Week 1–2)

┌─────────────────────────────────────────────────────────────────┐
│ STEP 8 — FIX ML INFERENCE FAILURE HANDLING          TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/detection/engines/ml_engine.py                 │
│ ACTION:     Replace the re-raise pattern in both predict() and │
│             predict_proba() with graceful degradation:         │
│             - Catch exception, log with full context           │
│             - Return EngineResult(verdict="unknown",           │
│               confidence=0.0, metadata={"error": ...,          │
│               "fallback": True})                               │
│             Add similar guard to AnomalyEngine.evaluate() for  │
│             the case where _model is None (A-10 fix)          │
│ WHY NOW:    Step 6 ensures models are clean. This step makes   │
│             the pipeline resilient to future transient         │
│             inference failures without crashing requests.      │
│             Must be done before the detection pipeline is      │
│             trusted to handle all traffic.                     │
│ DEPENDS ON: Step 6 (model integrity established)              │
│ ENABLES:    Reliable multi-engine pipeline operation          │
│ RISK:       Inference failures now silently degrade to        │
│             "unknown". Ensure monitoring alerts on elevated    │
│             "unknown" verdict rates (metric: ml_unknown_rate) │
│ ROLLBACK:   Revert ml_engine.py and anomaly_engine.py changes │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 9 — FIX POLICYGONFIG RACE CONDITION            TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/prevention_service.py — PolicyConfig,          │
│             PolicyConfigManager                                │
│ ACTION:     1. Convert PolicyConfig to frozen=True dataclass   │
│             2. Introduce PolicyConfigManager (per Solution 4)  │
│                with RLock + immutable snapshot swap            │
│             3. Update all callers of self.policy.X to use     │
│                config_manager.get().X                         │
│             4. Update set_policy() to use                     │
│                config_manager.update(**kwargs)                 │
│             5. Update ActionExecutor and RiskEngine to         │
│                receive PolicyConfigManager, not PolicyConfig   │
│ WHY NOW:    Prevention pipeline is about to be trusted with    │
│             real blocking (Phase A removed dry_run default).  │
│             A race that partially writes policy during live    │
│             blocking is dangerous.                             │
│ DEPENDS ON: Nothing (pure concurrency fix)                    │
│ ENABLES:    Step 15 (policy enforcement hardening)            │
│ RISK:       All callers of PolicyConfig must be updated.      │
│             grep for all .policy. references in codebase.     │
│ ROLLBACK:   Revert prevention_service.py                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 10 — FIX ANOMALY ENGINE MODEL RACE             TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/detection/engines/anomaly_engine.py            │
│ ACTION:     1. Replace direct self._model access with atomic   │
│                reference pattern (per Solution 4):             │
│                - _model_ref as the stored reference            │
│                - _get_model() captures snapshot in local var   │
│                - evaluate() uses local `model` var exclusively │
│             2. _set_model() does single assignment (GIL-atomic)│
│             3. fit() calls _set_model(new_model) atomically   │
│ WHY NOW:    Anomaly engine is being used for inference while   │
│             the buffer auto-fits on high traffic. Race is      │
│             active in production.                              │
│ DEPENDS ON: Nothing (self-contained)                          │
│ ENABLES:    Reliable AnomalyEngine operation                  │
│ RISK:       Low — change is additive, only affects access      │
│             pattern to _model attribute                        │
│ ROLLBACK:   Revert anomaly_engine.py                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 11 — FIX LEADER ELECTION FAIL-OPEN             TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/ha/leader_election.py                          │
│ ACTION:     1. Change Redis connection failure behavior from    │
│                return True to return False                     │
│             2. Log CRITICAL: "leader_election_unavailable —    │
│                prevention scheduler paused" when Redis fails   │
│             3. Add a metric: leader_election_state (0/1) that  │
│                monitoring can alert on                         │
│             4. Fix integer division: replace ttl // 3 with    │
│                max(1, int(self._ttl / 3.0))                   │
│ WHY NOW:    Redis failures in multi-instance deployments cause │
│             split-brain that produces duplicate firewall rules. │
│             Fail-closed is the correct behavior for a          │
│             prevention system.                                 │
│ DEPENDS ON: Step 9 (prevention scheduler uses                 │
│             PolicyConfigManager, so pause is clean)           │
│ ENABLES:    Step 16 (safe multi-instance prevention)          │
│ RISK:       Single-instance deployments without Redis: if Redis│
│             was never configured, is_leader() now returns False│
│             → prevention scheduler never runs. Set             │
│             INIDS_REDIS_REQUIRED=false in single-instance mode │
│             to retain always-leader behavior for that topology.│
│ ROLLBACK:   Revert leader_election.py                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 12 — ELIMINATE DUPLICATE ALERT/ACTION STORES   TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/detection_service.py — InMemoryAlertStore;     │
│             src/prevention_service.py — InMemoryPreventionStore│
│ ACTION:     1. Before removal: drain_to_ops_store() — flush    │
│                any in-memory alerts/actions to OpsStore at     │
│                startup (if OpsStore is available)              │
│             2. Remove InMemoryAlertStore from DetectionService │
│                constructor; DetectionService writes to         │
│                OpsStore directly via injected reference        │
│             3. Remove InMemoryPreventionStore. ActionExecutor  │
│                is already the canonical source via OpsStore.   │
│             4. Update all API routes that read from in-memory  │
│                stores to read from OpsStore instead            │
│             5. Add OpsStore.list_alerts() offset pagination    │
│                (I-05 fix)                                      │
│ WHY NOW:    Dashboard shows different data depending on query  │
│             path. With auth now working (Steps 1-3) and        │
│             persistence being trusted, eliminate the split     │
│             source of truth.                                   │
│ DEPENDS ON: Steps 1-3 (auth), OpsStore stability (B-01 fix in │
│             Step 13)                                           │
│ ENABLES:    Consistent API responses, reliable dashboard       │
│ RISK:       Any code reading from alert_store or               │
│             prevention_store directly breaks. Full audit of    │
│             all references required. Monitor OpsStore write    │
│             latency after consolidation.                       │
│ ROLLBACK:   Re-inject InMemoryAlertStore as secondary write    │
│             target temporarily                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 13 — FIX STARTUP MIGRATION UNCONDITIONAL UPDATES TIER: 1 │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/ops_store.py — _migrate_actions_table(),       │
│             _migrate_alerts_table()                            │
│ ACTION:     1. Add schema_version tracking per migration step  │
│                (each migration has an integer version number)  │
│             2. On startup: load current DB version from        │
│                schema_version table                            │
│             3. Apply only migrations with version > current    │
│             4. Remove the unconditional UPDATE queries that    │
│                run on every startup                            │
│             5. Data normalization UPDATEs moved into their     │
│                respective version-gated migration functions,   │
│                run exactly once.                               │
│             6. Fix _verify_schema_version() to re-raise all   │
│                exceptions that indicate startup should abort  │
│ WHY NOW:    Step 12 increases OpsStore write volume. Full-     │
│             table UPDATEs on every restart under increased     │
│             load will cause startup latency spikes.            │
│ DEPENDS ON: Nothing (standalone OpsStore change)              │
│ ENABLES:    Faster restarts, reliable schema management        │
│ RISK:       Migration ordering must be preserved. Existing DB  │
│             treated as "version 2" (current SCHEMA_VERSION).  │
│             Test migration upgrade path in staging first.      │
│ ROLLBACK:   Keep old migration functions as fallback for one   │
│             sprint; revert by re-enabling them                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 14 — REGISTER CORS MIDDLEWARE CORRECTLY        TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  web_app/app.py (or src/middleware.py)              │
│ ACTION:     1. Determine correct CORS policy for the API:      │
│                - Dashboard UI origin (specific domain, not *)  │
│                - Allowed methods: GET, POST, PUT, DELETE       │
│                - Allowed headers: Authorization, Content-Type, │
│                  X-API-Key                                     │
│                - Credentials: allowed (for session auth)       │
│             2. Register CORSMiddleware by calling              │
│                app.before_request(cors.handle_before) and      │
│                app.after_request(cors.handle_after)            │
│             3. Alternatively: replace with flask-cors          │
│                configured with the above policy                │
│             4. Remove CSP 'unsafe-inline' from               │
│                SecurityHeadersMiddleware — requires audit of   │
│                templates to externalize inline scripts         │
│ WHY NOW:    Auth is now enforced (Steps 1-3). An unenforced   │
│             CORS policy allows cross-origin requests to        │
│             authenticated endpoints from any webpage.          │
│ DEPENDS ON: Steps 1-3 (auth establishes what to protect)      │
│ ENABLES:    Full browser security model for the dashboard      │
│ RISK:       Removing unsafe-inline from CSP requires all       │
│             inline scripts to be externalized. This may break  │
│             the dashboard. Phase this: register CORS first,    │
│             address CSP unsafe-inline in Step 22 (Phase D).   │
│ ROLLBACK:   Remove cors registration (CORS enforcement removed)│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 15 — FIX SENSOR KEY ROLE ASSIGNMENT            TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/auth_service.py:53                             │
│ ACTION:     1. Define "sensor" role with permissions:          │
│                - POST /api/detect (submit features only)       │
│                - POST /api/stream (stream ingestion only)      │
│                - No read access to alerts, config, or actions  │
│             2. Change sensor key principal mapping from        │
│                role="analyst" to role="sensor"                 │
│             3. Ensure @require_auth on detection submission     │
│                endpoints accepts "sensor" role                 │
│ WHY NOW:    Sensor nodes are compromised by attackers or       │
│             malware regularly in network environments. A        │
│             compromised sensor must not have analyst access.   │
│             This is a single-line config change with high      │
│             security value.                                    │
│ DEPENDS ON: Step 1 (auth is enabled and role matters)         │
│ ENABLES:    Principle of least privilege for sensor nodes      │
│ RISK:       Any sensor that uses the API key for analyst-      │
│             level operations (reads, queries) breaks. Audit    │
│             all sensor integrations before deploying.         │
│ ROLLBACK:   Revert auth_service.py:53                         │
└─────────────────────────────────────────────────────────────────┘
PHASE C — System Unification (Week 2–4)

┌─────────────────────────────────────────────────────────────────┐
│ STEP 16 — IMPLEMENT UNIFIED AUTHENTICATION SYSTEM   TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  New: src/auth/auth_service.py,                     │
│             src/auth/jwt_manager.py, src/auth/models.py;       │
│             OpsStore schema (new tables)                       │
│ ACTION:     Implement Solution 1 in full:                      │
│             1. Add users, api_keys, revoked_tokens tables to   │
│                OpsStore (version-gated migration, Step 13      │
│                infrastructure)                                 │
│             2. Implement UnifiedAuthService:                   │
│                - Validates credentials against users table     │
│                - Issues RS256 JWT with jti claim               │
│                - Validates token: signature, expiry, not       │
│                  revoked (jti in revoked_tokens)               │
│                - API key path: SHA-256 lookup in api_keys      │
│             3. Implement require_roles() decorator             │
│                (Solution 2) replacing require_auth and         │
│                jwt_require_role                                │
│             4. Add startup route coverage validation           │
│             5. Run old and new auth systems in parallel during │
│                transition: old system handles requests if new  │
│                system would reject (2-sprint compatibility     │
│                window with INIDS_AUTH_COMPAT=true flag)        │
│ WHY NOW:    Bridge fixes in Steps 1 and 4 are temporary. This │
│             step implements the permanent auth architecture.    │
│             Must happen before RBAC migration (Step 17).       │
│ DEPENDS ON: Steps 1-4 (bridge auth), Step 13 (migration       │
│             infrastructure), Step 9 (config immutability for  │
│             auth config)                                       │
│ ENABLES:    Step 17 (RBAC migration), Step 18 (JWT hardening) │
│ RISK:       Auth system cutover is high-risk. Use             │
│             INIDS_AUTH_COMPAT flag for dual-run. Validate all  │
│             integration tests pass before removing old systems.│
│ ROLLBACK:   Set INIDS_AUTH_COMPAT=true (old system takes over)│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 17 — MIGRATE RBAC INTO UNIFIED AUTH SYSTEM     TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/rbac_manager.py, inids_rbac.db                 │
│ ACTION:     1. Export current inids_rbac.db data               │
│             2. Map RBAC user IDs to new users table entries    │
│             3. Map RBAC roles to unified role schema           │
│             4. Add roles to api_keys entries in OpsStore       │
│             5. Migrate RBAC audit log to OpsStore.audits        │
│             6. Update any RBAC authorization checks to use     │
│                UnifiedAuthService (Step 16)                    │
│             7. After validation: remove rbac_manager.py and    │
│                inids_rbac.db                                   │
│ WHY NOW:    After Step 16, there are now two auth systems: new │
│             unified system and the legacy RBAC. A system with  │
│             two disconnected auth sources is still vulnerable. │
│             Must collapse to one.                              │
│ DEPENDS ON: Step 16 (unified auth must exist first)           │
│ ENABLES:    Single authorization truth: every access check     │
│             goes to one system                                 │
│ RISK:       RBAC data must be migrated without data loss. Run  │
│             parallel audit log writes during transition.       │
│ ROLLBACK:   Restore rbac_manager.py and inids_rbac.db from    │
│             backup; add back RBAC auth checks                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 18 — HARDEN JWT: RS256, REVOCATION, NO EXPIRE  TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/auth/jwt_manager.py (from Step 16)             │
│ ACTION:     Step 16 implements RS256 and jti. This step:       │
│             1. Generate RSA-2048 or EC-P256 keypair for        │
│                production deployment                           │
│             2. Remove all references to INIDS_SECRET_KEY for  │
│                JWT purposes (JWT uses its own keypair)         │
│             3. Implement token revocation endpoint:            │
│                POST /api/auth/revoke — requires valid token,   │
│                inserts jti into revoked_tokens                 │
│             4. Implement revoked_tokens auto-cleanup (delete   │
│                rows where expires_at < NOW() — run daily)      │
│             5. Eliminate api_auth_refresh allow_expired path   │
│                entirely                                        │
│ WHY NOW:    Step 16 builds the auth system. Step 18 eliminates│
│             the remaining JWT-specific vulnerabilities (C-19,  │
│             A-02) by switching to the correct algorithm and    │
│             adding revocation.                                 │
│ DEPENDS ON: Step 16 (JWT infrastructure must exist)           │
│ ENABLES:    Correct JWT security posture system-wide           │
│ RISK:       All existing JWT tokens issued under HS256 become  │
│             invalid. Coordinate with all API consumers.        │
│             Provide migration notice. Use short cutover window.│
│ ROLLBACK:   Fall back to HS256 with new (non-placeholder) key │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 19 — UNIFY RATE LIMITING                       TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/middleware.py — RateLimitMiddleware;           │
│             web_app/app.py — InMemoryRateLimiter usage         │
│ ACTION:     1. Remove RateLimitMiddleware (WSGI layer)         │
│             2. Remove InMemoryRateLimiter from before_request  │
│             3. Implement unified RateLimiter (per Solution 9):

┌─────────────────────────────────────────────────────────────────┐
│ STEP 19 — UNIFY RATE LIMITING (continued)           TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/middleware.py, web_app/app.py                  │
│ ACTION:     1. Remove RateLimitMiddleware (WSGI layer)         │
│             2. Remove InMemoryRateLimiter from before_request  │
│             3. Implement single RateLimiter (Solution 9):      │
│                - Registered as one Flask before_request hook   │
│                - Three tiers: global-IP (1000/min), user-IP    │
│                  (200/min authenticated), route-specific       │
│                - Redis-backed if available; in-memory fallback │
│                - @rate_limit(per_minute=N) decorator for routes│
│             4. Verify new limiter fires once per request        │
│ WHY NOW:    After Step 16 (unified auth), user identity is     │
│             available in before_request — per-user limits now  │
│             possible. Two conflicting limiters must be removed  │
│             before the system can have a reliable contract.    │
│ DEPENDS ON: Step 16 (user identity in request context)        │
│ ENABLES:    Predictable, auditable rate enforcement            │
│ RISK:       Removing both old limiters simultaneously creates  │
│             a window with no rate limiting. Deploy new limiter  │
│             first, verify it fires, then remove old ones.      │
│ ROLLBACK:   Re-register old middleware if new limiter fails    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 20 — FIX PREVENTION IDEMPOTENCY AT DB LEVEL    TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/ops_store.py — actions table schema;           │
│             src/ips/action_executor.py — execute()             │
│ ACTION:     1. Add partial unique index to actions table:      │
│                CREATE UNIQUE INDEX IF NOT EXISTS               │
│                uq_active_block ON actions(target)              │
│                WHERE lower(status) IN                          │
│                ('active','enforced','executed')                │
│                AND lower(action_type) IN                       │
│                ('block','temp_block','rate_limit');            │
│             2. In save_action(): catch IntegrityError from     │
│                duplicate insert — return existing record       │
│                instead of re-inserting                         │
│             3. Remove the has_active_block() read-check-write  │
│                TOCTOU pattern from execute() — idempotency is  │
│                now guaranteed at the INSERT level              │
│ WHY NOW:    Step 11 fixed leader election fail-closed, but     │
│             split-brain can still occur transiently. DB-level  │
│             uniqueness is the correct idempotency mechanism.   │
│ DEPENDS ON: Step 13 (migration infrastructure for schema       │
│             change)                                            │
│ ENABLES:    Safe multi-instance prevention operation           │
│ RISK:       SQLite partial index syntax differs slightly from  │
│             PostgreSQL. Test both backends. Partial indexes     │
│             require SQLite 3.8.9+.                             │
│ ROLLBACK:   DROP INDEX uq_active_block; restore has_active_    │
│             block() check in execute()                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 21 — FIX WEBHOOK ADAPTER STATE + RECONCILE     TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/firewall_adapters.py — WebhookFirewallAdapter; │
│             src/ips/action_executor.py — reconcile()           │
│ ACTION:     1. Add adapter capability flag:                    │
│                WebhookFirewallAdapter.supports_rule_query = False│
│                UfwFirewallAdapter.supports_rule_query = True   │
│                NftablesFirewallAdapter.supports_rule_query = True│
│             2. reconcile() checks adapter.supports_rule_query: │
│                if False → log "reconcile skipped: stateless    │
│                adapter" and return                             │
│             3. Remove blocked_targets dict from               │
│                WebhookFirewallAdapter — it was the source of   │
│                spurious post-restart DESYNCED state            │
│             4. WebhookFirewallAdapter.list_rules() raises      │
│                NotImplementedError (forces caller to check cap)│
│ WHY NOW:    reconcile() runs on every prevention scheduler     │
│             cycle. With Step 11 (fail-closed leader election)  │
│             now active, the scheduler runs reliably — spurious │
│             DESYNCED records from webhook adapter must be      │
│             eliminated before they trigger false remediation.  │
│ DEPENDS ON: Steps 11, 13                                      │
│ ENABLES:    Clean reconciliation for all adapter types         │
│ RISK:       Any code calling list_rules() on a webhook adapter │
│             must now check supports_rule_query first. Audit   │
│             all adapter.list_rules() call sites.              │
│ ROLLBACK:   Revert firewall_adapters.py and action_executor.py│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 22 — SANITIZE CORRELATION ID + AUDIT TIMESTAMP TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/correlation_tracing.py;                        │
│             src/middleware.py — AuditLogMiddleware             │
│ ACTION:     1. In correlation_tracing.py: sanitize the         │
│                X-Correlation-ID header before storage:         │
│                - Strip all non-printable and control chars     │
│                - Reject values with \n, \r, or null bytes      │
│                - Truncate to 64 chars maximum                  │
│                - If invalid: generate server-side UUID instead │
│             2. In AuditLogMiddleware: replace datetime.utcnow()│
│                with datetime.now(timezone.utc) throughout       │
│             3. In rbac_manager.py AuditLog: replace ISO        │
│                timestamp ID with uuid.uuid4().hex (F-02)       │
│ WHY NOW:    Log injection (C-11) is an active attack path for  │
│             covering tracks. With real authentication now in   │
│             place (Step 16), the audit log has legal standing. │
│             Its integrity must be protected.                   │
│ DEPENDS ON: Step 17 (unified audit trail exists in OpsStore)  │
│ ENABLES:    Forensically reliable audit trail                  │
│ RISK:       Low. Sanitization is additive. Monitor for         │
│             correlation IDs being stripped unexpectedly.       │
│ ROLLBACK:   Revert correlation_tracing.py                     │
└─────────────────────────────────────────────────────────────────┘
PHASE D — Hardening & Depth (Month 2)

┌─────────────────────────────────────────────────────────────────┐
│ STEP 23 — ELIMINATE production_hardening.py DEAD CODE TIER: 2 │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/production_hardening.py                        │
│ ACTION:     This module is entirely unreferenced. The circuit  │
│             breaker it defines duplicates ActionExecutor's.    │
│             The encrypt_sensitive_data function is a SHA-256   │
│             hash masquerading as encryption.                   │
│             1. Delete src/production_hardening.py              │
│             2. Confirm no imports exist anywhere in codebase   │
│             3. Ensure ActionExecutor's circuit breaker is the  │
│                canonical implementation (it is active)         │
│ WHY NOW:    Leaving it risks a future developer importing and  │
│             using its broken "encryption" function (SHA-256    │
│             with hardcoded key "master-key") or the            │
│             non-resetting rate limiter.                        │
│ DEPENDS ON: Steps 16-17 (real auth replaces any future         │
│             temptation to use this module)                     │
│ ENABLES:    No confusion about which circuit breaker to use    │
│ RISK:       Zero — module is provably unreferenced             │
│ ROLLBACK:   Restore from git                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 24 — FIX INPUT SANITIZER                       TIER: 2   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/input_sanitizer.py                             │
│ ACTION:     1. Fix sanitize_string UUID handling: allow        │
│                hyphens by default; remove '-' from the special │
│                character deny set                              │
│             2. Wire up SQL_KEYWORDS check: add               │
│                detect_sql_injection(value) function that       │
│                checks for SQL keywords in unexpected positions  │
│             3. Extend XSS patterns: add <svg, <math,           │
│                <details, all on* event handler patterns        │
│                (onerror, onload, onclick, onmouseover, etc.)   │
│             4. Apply sanitizer to all user-submitted string    │
│                fields in detection feature endpoints (not to   │
│                numeric fields — those are float-cast already)  │
│ WHY NOW:    Auth is enforced (Phase A/B). The remaining        │
│             injection surfaces are at the feature submission   │
│             and configuration endpoints. Fix before hardening  │
│             the endpoints further (Step 25).                   │
│ DEPENDS ON: Step 3 (routes have auth; now harden their input) │
│ ENABLES:    Step 25 (packet capture sanitization)             │
│ RISK:       Overly aggressive sanitization may reject valid    │
│             inputs. Test with realistic feature payloads.      │
│ ROLLBACK:   Revert input_sanitizer.py                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 25 — SANITIZE LIVE PACKET CAPTURE OUTPUT       TIER: 2   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/capture_live_traffic.py                        │
│ ACTION:     1. Remove hardcoded logged_in=1 — derive from      │
│                session analysis or leave as 0 (unknown) until  │
│                implemented                                     │
│             2. For all string fields extracted from packets:   │
│                apply sanitize_string() before writing to CSV   │
│             3. For all numeric fields: enforce float()/int()   │
│                conversion with fallback to 0 on failure        │
│             4. Clamp all numeric fields to valid NSL-KDD       │
│                feature ranges (prevents adversarial feature    │
│                manipulation via crafted packets)               │
│ WHY NOW:    Step 24 fixes the sanitizer. Step 25 applies it    │
│             to the highest-risk input path (raw packet data).  │
│ DEPENDS ON: Step 24 (sanitizer must be correct first)         │
│ ENABLES:    Trustworthy ML feature pipeline from live traffic  │
│ RISK:       Clamping may affect detection accuracy for         │
│             extreme-value attacks. Validate against test PCAP  │
│             files before deploying.                            │
│ ROLLBACK:   Revert capture_live_traffic.py                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 26 — HARDEN PREVENTION DEFAULTS + TI PIPELINE  TIER: 2  │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/prevention_service.py — PolicyConfig;          │
│             web_app/app.py — TI initialization                 │
│ ACTION:     1. Remove dry_run=True default from PolicyConfig.  │
│                dry_run is now a deployment-time setting:        │
│                INIDS_DRY_RUN=true|false (required config, no   │
│                default — startup fails if not set explicitly)  │
│             2. Add startup validation: if INIDS_ENVIRONMENT=   │
│                production AND INIDS_DRY_RUN=true → log         │
│                WARNING: "Prevention is in dry-run mode in a    │
│                production environment"                         │
│             3. Implement TI feed loader from external file:    │
│                parse INIDS_TI_FEED_PATH (CSV or STIX/JSON)     │
│                Load indicators at startup and on SIGHUP        │
│             4. Add TI feed validation: reject private/loopback │
│                ranges from TI feed (RFC-1918, 127.x, ::1)     │
│ WHY NOW:    Step 2 cleared mock TI. Step 26 replaces the void  │
│             with a real feed loader and ensures dry_run is     │
│             intentional, not a forgotten default.              │
│ DEPENDS ON: Steps 2 (mock TI removed), 9 (PolicyConfig frozen)│
│ ENABLES:    Production prevention mode safe to enable          │
│ RISK:       Operators must explicitly set INIDS_DRY_RUN.       │
│             Document this in runbook. Alert if missing.        │
│ ROLLBACK:   Restore dry_run=True default in PolicyConfig       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 27 — HARDEN NFTABLES HANDLE PARSING            TIER: 2   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/firewall_adapters.py — NftablesFirewallAdapter │
│ ACTION:     Replace text-based handle parsing in unblock() with│
│             nft --json output mode:                            │
│             nft -j list chain inet filter input                │
│             Parse JSON output to extract handle numbers.       │
│             JSON output is a stable, versioned nftables API.   │
│             Add fallback: if JSON parse fails, log error and   │
│             return False (do not attempt text parsing).        │
│ WHY NOW:    Step 21 marked nftables as a supported rule-query  │
│             adapter. Its unblock() path is relied upon for     │
│             expiry cleanup. Text parsing breakage = permanent  │
│             IP blocks that can't be cleared.                   │
│ DEPENDS ON: Step 21                                           │
│ ENABLES:    Reliable nftables unblock operations               │
│ RISK:       nft --json requires nftables 0.9.1+. Verify        │
│             target system nftables version before deploying.   │
│ ROLLBACK:   Revert to text parsing (accepts the fragility)    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 28 — FIX ALERT ID COLLISION + DEDUPLICATION    TIER: 2   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/detection_service.py — Alert ID generation;    │
│             src/ops_store.py — save_alert(), list_alerts()     │
│ ACTION:     1. Change Alert ID to full UUID:                   │
│                f"al_{uuid.uuid4().hex}" (no truncation)        │
│             2. Add alert deduplication in save_alert():        │
│                Before INSERT, check:                           │
│                SELECT id FROM alerts                           │
│                WHERE source_ip=:ip AND attack_type=:type       │
│                AND timestamp > datetime('now', '-60 seconds')  │
│                If found: UPDATE count field (add count column  │
│                to alerts table) instead of inserting new row   │
│             3. Add index: CREATE INDEX idx_alert_dedup ON      │
│                alerts(source_ip, attack_type, timestamp)       │
│ WHY NOW:    Step 12 eliminated the in-memory store. All alerts │
│             now go to OpsStore. With full write volume, the    │
│             truncated UUID collision probability becomes real   │
│             on high-volume deployments.                        │
│ DEPENDS ON: Step 12 (OpsStore is now the only alert store),   │
│             Step 13 (migration infrastructure)                 │
│ ENABLES:    Reliable alert identity and reduced noise           │
│ RISK:       Existing short alert IDs in OpsStore remain.       │
│             API consumers must handle both formats during       │
│             transition. New IDs are longer — check UI display. │
│ ROLLBACK:   Revert ID format; drop dedup index                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 29 — FIX HEALTH CHECK PROBE AUDIT WRITE        TIER: 2   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  web_app/app.py — _ops_probe()                      │
│ ACTION:     Replace the audit INSERT in _ops_probe with a      │
│             read-only schema_version query:                     │
│             SELECT version FROM schema_version LIMIT 1         │
│             Remove add_audit() call from health check path.    │
│             Add a separate /api/health/deep endpoint           │
│             (admin-only) that does perform a write test when   │
│             explicitly needed.                                 │
│ WHY NOW:    Step 12 moved all alerts to OpsStore. The audits   │
│             table now carries real security events. Polluting   │
│             it with health check rows at 2/minute undermines   │
│             its forensic value and query performance.          │
│ DEPENDS ON: Step 13 (schema_version table guaranteed to exist) │
│ ENABLES:    Clean audit trail with only meaningful events      │
│ RISK:       Minimal — health check behavior is unchanged from  │
│             the caller's perspective                           │
│ ROLLBACK:   Revert _ops_probe() to restore audit write        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 30 — ADD ALERT PAGINATION + RETENTION POLICY   TIER: 2   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/ops_store.py — list_alerts();                  │
│             web_app/app.py — all alert listing routes          │
│ ACTION:     1. Add offset parameter to list_alerts():          │
│                list_alerts(limit=50, offset=0, ...) → rows, total│
│             2. Update all API routes that call list_alerts to  │
│                pass offset from query param (?offset=N)        │
│             3. Add cleanup job to prevention scheduler:        │
│                DELETE FROM alerts WHERE timestamp < NOW() - N  │
│                days (N from INIDS_ALERT_RETENTION_DAYS setting)│
│             4. Same retention job for audits table             │
│             5. Add index on alerts.timestamp for efficient     │
│                retention queries                               │
│ WHY NOW:    Step 28 adds dedup which reduces volume. But the   │
│             table grows forever without retention. Pagination  │
│             and retention must be added together so the API    │
│             contract (consistent page sizes) holds.            │
│ DEPENDS ON: Steps 12, 28                                      │
│ ENABLES:    Sustainable long-term OpsStore operation           │
│ RISK:       Retention job deletes data. Test on staging with   │
│             production-scale data first. Log rows deleted.     │
│ ROLLBACK:   Disable retention job; keep pagination (additive)  │
└─────────────────────────────────────────────────────────────────┘
PHASE E — Quality & Observability (Month 2–3)

┌─────────────────────────────────────────────────────────────────┐
│ STEP 31 — FIX IP BLOCKLIST LOCALHOST WHITELIST      TIER: 3   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/middleware.py — IPBlockingMiddleware            │
│ ACTION:     Replace string 'localhost' in the whitelist with   │
│             both '127.0.0.1' and '::1'. Extend to resolve      │
│             string hostnames to IPs at startup for any         │
│             other whitelist entries that are hostnames.        │
│ DEPENDS ON: Nothing. RISK: Low. ROLLBACK: Revert one line.    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 32 — REMOVE unsafe-inline FROM CSP             TIER: 3   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/middleware.py — SecurityHeadersMiddleware;     │
│             web_app/templates/*.html                           │
│ ACTION:     1. Audit all HTML templates for inline <script>    │
│                and <style> blocks                              │
│             2. Move inline scripts to static .js files         │
│             3. Move inline styles to static .css files         │
│             4. Remove 'unsafe-inline' from script-src and      │
│                style-src in CSP header                         │
│             5. For any dynamically-generated JS values:        │
│                use data-* attributes + external JS to read them│
│ DEPENDS ON: Step 14 (CORS registered). RISK: Dashboard may    │
│             break if inline scripts missed — test all pages.   │
│ ROLLBACK:   Re-add 'unsafe-inline' temporarily                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 33 — FIX SYSTEM METRICS (UPTIME, HEALTH)       TIER: 3   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  web_app/app.py — dashboard metrics route           │
│ ACTION:     1. Record _process_start_time = time.monotonic()   │
│                at app startup                                  │
│             2. system_uptime computed as:                      │
│                seconds = time.monotonic() - _process_start_time│
│                formatted as "Xd Yh Zm"                        │
│             3. Remove hardcoded system_health: 98              │
│             4. Compute health from: OpsStore connectivity,     │
│                ML model is_ready(), prevention scheduler       │
│                running flag → aggregate to 0-100 score         │
│ DEPENDS ON: Nothing. RISK: Low. ROLLBACK: Revert metric fields.│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 34 — REMOVE OPS_DB_PATH FROM HEALTH ENDPOINT  TIER: 3   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  web_app/app.py — api_health route                  │
│ ACTION:     Remove OPS_DB_PATH from the JSON response body.    │
│             Replace with a boolean: "db_connected": true/false │
│             The /api/health/deep endpoint (Step 29) may return │
│             path info but only to admin-authenticated callers. │
│ DEPENDS ON: Step 29. RISK: None. ROLLBACK: Re-add field.      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 35 — INSTRUMENT RATE COUNTER PERFORMANCE       TIER: 3   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/detection/engines/threshold_engine.py          │
│ ACTION:     Replace _RateCounter list with deque-based         │
│             sliding window:                                    │
│             self._timestamps = collections.deque()             │
│             count(): popleft() while deque[0] <= cutoff        │
│             (O(k) where k=expired entries, not O(n) total).   │
│             Fix now = now or time.monotonic() → now is None    │
│             check. Fix FIFO eviction to LRU: track last_used   │
│             per IP, evict lowest last_used entry.              │
│ DEPENDS ON: Nothing. RISK: Low. ROLLBACK: Revert counter impl. │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 36 — FIX HONEYPOT ENGINE CONFIDENCE + ENV VAR  TIER: 3   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/detection/engines/honeypot_engine.py           │
│ ACTION:     1. Normal verdict: return confidence=100.0 (engine │
│                is certain there is no honeypot involvement)    │
│             2. Read HONEYPOT_ENABLED env var in __init__:      │
│                if os.environ.get("HONEYPOT_ENABLED","1") in    │
│                {"0","false","no"}: self._enabled = False        │
│             3. log a warning if HONEYPOT_ENABLED=false is set  │
│                but IPs/ports are also configured (conflict)    │
│ DEPENDS ON: Nothing. RISK: Low. ROLLBACK: Revert __init__.    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 37 — ENFORCE WEBHOOK TLS                       TIER: 3   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/firewall_adapters.py — WebhookFirewallAdapter  │
│ ACTION:     In __post_init__: validate webhook_url scheme:     │
│             if not self.webhook_url.startswith("https://"):    │
│               raise ValueError("Webhook URL must use HTTPS")  │
│             In _post(): create explicit SSLContext with        │
│             verify=True and set minimum TLS 1.2.              │
│ DEPENDS ON: Step 21. RISK: Any http:// webhook configs break.  │
│ ROLLBACK:   Remove scheme validation.                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 38 — MOVE OpsStore._fetchall CALL TO PUBLIC API TIER: 3  │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/ops_store.py; web_app/app.py — api_actions_    │
│             pending                                            │
│ ACTION:     Add public method to OpsStore:                     │
│             def list_pending_approvals(self) -> list[dict]:    │
│               return self._fetchall(                           │
│                 "SELECT * FROM actions WHERE                   │
│                 lower(COALESCE(status,''))='pending_approval'" │
│               )                                               │
│             Update api_actions_pending to call this method.    │
│ DEPENDS ON: Step 13. RISK: None. ROLLBACK: Revert call site.  │
└─────────────────────────────────────────────────────────────────┘
PHASE F — Long-Term Debt Elimination (Quarter 2)

┌─────────────────────────────────────────────────────────────────┐
│ STEP 39 — DECOMPOSE web_app/app.py GOD FILE         TIER: 2   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  web_app/app.py (4500+ lines)                       │
│ ACTION:     Decompose into Flask Blueprints, one per domain:   │
│             web_app/blueprints/                                │
│               auth_bp.py       — /api/auth/* routes            │
│               detection_bp.py  — /api/detect/* routes          │
│               alerts_bp.py     — /api/alerts/* routes          │
│               prevention_bp.py — /api/prevention/* routes      │
│               ops_bp.py        — /api/ops/* routes             │
│               admin_bp.py      — /api/admin/* routes           │
│               health_bp.py     — /health, /api/health routes   │
│             web_app/app.py retains only:                       │
│               - Application factory create_app()               │
│               - Blueprint registration                         │
│               - Middleware registration                        │
│               - Startup initialization (load_models, etc.)     │
│             Migration: one Blueprint at a time, each verified  │
│             with existing integration tests before next.       │
│ WHY NOW:    All structural fixes (Phases A-E) are complete.    │
│             Decomposition is now safe because the auth model,  │
│             storage model, and concurrency model are stable.   │
│             Decomposing before stability would have required   │
│             re-doing structural changes across multiple files. │
│ DEPENDS ON: All Phase A-E steps complete                      │
│ ENABLES:    Independent testing per domain, parallel           │
│             development, auditable per-module auth coverage    │
│ RISK:       URL routing changes if blueprint URL prefixes      │
│             differ from current. Use url_prefix to preserve    │
│             existing paths during migration.                   │
│ ROLLBACK:   Each Blueprint extraction is independently         │
│             revertable — revert one blueprint at a time        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 40 — IMPLEMENT FULL TEST REGRESSION SUITE      TIER: 1   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  tests/ directory                                   │
│ ACTION:     Implement the full regression suite from Phase 5:  │
│             tests/security/                                    │
│               test_auth_bypass_impossible.py  — all attack     │
│               chains AC-1 through AC-4 as negative tests       │
│               test_route_auth_coverage.py     — every route    │
│               requires auth or explicit public declaration     │
│               test_placeholder_config_rejected.py             │
│               test_model_integrity.py                         │
│               test_jwt_no_password_no_token.py                │
│             tests/concurrency/                                 │
│               test_policy_config_race.py                      │
│               test_anomaly_engine_fit_evaluate.py             │
│               test_rate_counter_concurrent.py                 │
│             tests/migration/                                   │
│               test_schema_migration_idempotent.py             │
│               test_migration_rollback.py                      │
│             CI gate: all tests must pass; coverage ≥ 80% on   │
│             src/auth/, src/detection/, src/ips/               │
│ WHY NOW:    All architectural fixes are in place. Tests now    │
│             verify correctness and serve as regression guards. │
│             Adding tests before fixes would have been testing  │
│             broken behavior.                                   │
│ DEPENDS ON: All Phase A-E steps                               │
│ ENABLES:    Sustainable development: every future change runs  │
│             this suite before merge                            │
│ RISK:       Discovering untested breakage from earlier steps.  │
│             Treat test failures as bugs to fix, not to skip.  │
│ ROLLBACK:   N/A — test suite additions are non-breaking       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 41 — IMPLEMENT ML MODEL VERSIONING + HOT RELOAD TIER: 2  │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  web_app/app.py — load_models(); src/detection/     │
│             engines/ml_engine.py, anomaly_engine.py            │
│ ACTION:     1. Implement ACTIVE_MODELS manifest (per Solution 7│
│             2. POST /api/admin/models/reload (admin only)      │
│                → reads ACTIVE_MODELS, verifies checksums,      │
│                → atomic swap via _set_model() (Step 10 infra)  │
│             3. Model staging slot: load new model to          │
│                _pending_model, run 10 inference checks against │
│                known-good samples, if pass → swap to active    │
│             4. Old model retained for 1 hour after swap        │
│                (still valid for in-flight requests)            │
│ DEPENDS ON: Steps 6, 8, 10 (integrity check, graceful degrade,│
│             atomic reference swap)                             │
│ ENABLES:    Zero-downtime model updates with rollback safety   │
│ RISK:       Staging validation samples must be representative. │
│             Document which samples are used for validation.    │
│ ROLLBACK:   Roll back ACTIVE_MODELS manifest entry            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 42 — DEPENDENCY AUDIT AND CVE REMEDIATION      TIER: 3   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  requirements.txt                                   │
│ ACTION:     Run pip-audit or safety check against current      │
│             requirements.txt (generated with hashes in Step 7).│
│             For each CVE identified in cryptography==41.0.7,   │
│             numpy==1.24.3, Werkzeug==3.0.1, scapy==2.5.0:     │
│             1. Identify the patched version                    │
│             2. Test application against patched version        │
│             3. Regenerate requirements.txt hashes after upgrade│
│             4. Add pip-audit to CI — fail build on known CVEs  │
│             Target: cryptography ≥ 42.0.8, numpy ≥ 1.26.4,   │
│             Werkzeug ≥ 3.0.6                                   │
│ DEPENDS ON: Step 7 (hashed requirements exist as baseline)    │
│ RISK:       API changes between versions may break code.       │
│             Run full test suite after each upgrade.            │
│ ROLLBACK:   Revert to previous pinned version and its hash    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 43 — FIX SUSPICIOUS LABEL INVERSION            TIER: 4   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/detection_service.py:124                       │
│ ACTION:     Redefine suspicious to mean what operators expect: │
│             suspicious = (prediction == "Attack" and           │
│                           confidence < threshold)              │
│             # True = attack prediction with uncertain confidence│
│             Update reason field accordingly:                   │
│             reason = "low_confidence_attack" if suspicious     │
│                      else "high_confidence_attack" if pred==   │
│                      "Attack" else "normal"                    │
│ DEPENDS ON: Nothing. RISK: Changes alert reason field content. │
│             Update any dashboards or filters keyed on reason.  │
│ ROLLBACK:   Revert detection_service.py:124                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 44 — IMPLEMENT CONTEXTVAR REQUEST ISOLATION    TIER: 3   │
├─────────────────────────────────────────────────────────────────┤
│ COMPONENT:  src/correlation_tracing.py                         │
│ ACTION:     If running under Gevent/Eventlet:                  │
│             1. Wrap each request in a new contextvars.copy_    │
│                context().run() call to ensure ContextVars      │
│                are isolated per greenlet                       │
│             2. Alternatively: replace ContextVar with          │
│                flask.g (Flask's per-request context, which     │
│                Flask-SocketIO handles correctly)               │
│             3. Add a test: two concurrent requests with        │
│                different X-Correlation-IDs must never mix      │
│                (tests/concurrency/test_correlation_isolation.py)│
│ DEPENDS ON: Step 22 (correlation sanitization), Step 40       │
│             (test infrastructure)                              │
│ RISK:       Medium — greenlet context behavior is subtle.      │
│             Test under load before deploying.                  │
│ ROLLBACK:   Revert to raw ContextVar usage                    │
└─────────────────────────────────────────────────────────────────┘
ROADMAP SUMMARY TABLE
Step	Title	Tier	Phase	Depends On
1	Disable auth bypass	0	A	—
2	Remove mock threat intel	0	A	—
3	Auth all undecorated routes	0	A	1
4	Break passwordless JWT login	0	A	1
5	Harden Docker (non-root, no repo mount)	0	A	1
6	Model integrity verification	0	A	5
7	Pin requirements with hashes + CVE upgrades	0	A	5
8	ML inference graceful degradation	1	B	6
9	PolicyConfig race → frozen + manager	1	B	—
10	AnomalyEngine model swap race	1	B	—
11	Leader election fail-closed	1	B	9
12	Eliminate duplicate alert/action stores	1	B	1, 13
13	Version-gated schema migrations	1	B	—
14	Register CORS middleware	1	B	1–3
15	Fix sensor key role	1	B	1
16	Unified auth system	1	C	1–4, 13, 9
17	Migrate RBAC into unified auth	1	C	16
18	RS256 + revocation + no-expired-refresh	1	C	16
19	Unify rate limiters	1	C	16
20	DB-level idempotency for prevention	1	C	13
21	Webhook adapter state + reconcile fix	1	C	11, 13
22	Sanitize correlation ID + audit timestamps	1	C	17
23	Delete production_hardening.py dead code	2	D	16–17
24	Fix input sanitizer	2	D	3
25	Sanitize packet capture output	2	D	24
26	Prevention defaults + TI feed loader	2	D	2, 9
27	nftables JSON handle parsing	2	D	21
28	Full alert IDs + deduplication	2	D	12, 13
29	Health check probe → read-only	2	D	13
30	Alert pagination + retention	2	D	12, 28
31	Localhost whitelist fix	3	E	—
32	Remove CSP unsafe-inline	3	E	14
33	Fix hardcoded uptime/health metrics	3	E	—
34	Remove OPS_DB_PATH from /health	3	E	29
35	Rate counter deque + LRU eviction	3	E	—
36	Honeypot confidence + env var	3	E	—
37	Enforce webhook TLS	3	E	21
38	OpsStore public API for pending approvals	3	E	13
39	Decompose app.py into Blueprints	2	F	all Phase A–E
40	Full regression + security test suite	1	F	all Phase A–E
41	ML model versioning + hot reload	2	F	6, 8, 10
42	Dependency CVE audit + remediation	3	F	7
43	Fix suspicious label inversion	4	F	—
44	ContextVar greenlet isolation	3	F	22, 40
CRITICAL PATH
The minimum sequence to close all Tier 0 attack chains in 72 hours:


Step 1 (auth bypass) 
  → Step 2 (mock TI)
  → Steps 3, 4 in parallel (route auth + JWT bridge fix)
  → Step 5 (Docker hardening — requires rebuild)
  → Steps 6, 7 in parallel (model integrity + hash requirements)
After this sequence, attack chains AC-1 through AC-4 are closed. The system remains architecturally fragile but is no longer trivially exploitable without credentials. Phases B through F then systematically eliminate the remaining structural debt in dependency order.

The single rule that governs the entire sequence: never enable production prevention mode (Step 26 / disable dry_run) before Step 2 (mock TI removed) is confirmed deployed. That ordering constraint supersedes all other scheduling flexibility.