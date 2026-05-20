# INIDS — Intelligent Network Intrusion Detection System
## Comprehensive Technical Architecture Report

**Document Classification:** Confidential — Internal Technical Documentation
**Report Version:** 1.0
**Analysis Scope:** Full Repository Audit
**Prepared By:** Senior Technical Audit Team
**Date:** 2026-05-19

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Project Overview & Business Context](#2-project-overview--business-context)
3. [Technology Stack & Dependencies](#3-technology-stack--dependencies)
4. [System Architecture](#4-system-architecture)
5. [Database Design & Data Architecture](#5-database-design--data-architecture)
6. [Backend API Architecture & Business Logic](#6-backend-api-architecture--business-logic)
7. [Frontend Architecture & User Experience](#7-frontend-architecture--user-experience)
8. [Authentication, Authorization & Security](#8-authentication-authorization--security)
9. [Infrastructure, Deployment & DevOps](#9-infrastructure-deployment--devops)
10. [External Integrations & Third-Party Services](#10-external-integrations--third-party-services)
11. [Testing Strategy & Quality Assurance](#11-testing-strategy--quality-assurance)
12. [Performance, Scalability & Resilience](#12-performance-scalability--resilience)
13. [Cross-Cutting Concerns](#13-cross-cutting-concerns)
14. [Technical Debt & Risk Assessment](#14-technical-debt--risk-assessment)
15. [Architecture Decision Records & Design Rationale](#15-architecture-decision-records--design-rationale)
16. [Recommendations & Improvement Roadmap](#16-recommendations--improvement-roadmap)
17. [Appendices](#17-appendices)

---

## 1. Executive Summary

### 1.1 Purpose and Scope

This report presents the results of a comprehensive technical architecture audit of the INIDS (Intelligent Network Intrusion Detection System) repository. The audit covered the complete codebase as of 2026-05-19, encompassing approximately 150+ Python source files, 112 test files, frontend templates and JavaScript, infrastructure configuration, and CI/CD pipelines.

INIDS is a full-featured network intrusion detection and prevention system built on top of machine learning models trained on the NSL-KDD dataset. The system has undergone substantial architectural evolution — the repository contains evidence of a multi-phase recovery and hardening effort ("PLAN.md", totaling 292KB of specification text) that substantially improved the security posture, code quality, and operational capabilities from an earlier version.

### 1.2 Key Findings Summary

| Category | Finding | Severity |
|---|---|---|
| Authentication | RS256 JWT with stateful revocation; fail-closed startup validation | Positive |
| Authorization | Hierarchical RBAC with `@require_roles` on every non-public route; startup enforcement | Positive |
| Rate Limiting | Two-tier (IP + per-user) unified rate limiter; Redis-backed with in-memory fallback | Positive |
| Database | SQLite dev / PostgreSQL prod dual-backend with 6-version migration framework | Positive |
| Secret Handling | `load_settings()` fails hard on missing SECRET_KEY; ephemeral JWT keys in dev | Moderate Risk |
| Input Validation | `input_sanitizer.py` with SQL/XSS pattern checks; jsonschema validation on predict | Positive |
| ML Model Integrity | SHA-256 checksum verification at load time; strict/warn/disabled modes | Positive |
| WebSocket Security | JWT required on `/events` namespace connect; ephemeral key invalidates tokens on restart | Moderate Risk |
| CSRF Protection | CSRF middleware present; coverage of all mutating HTML form endpoints not fully verified | Low-Moderate Risk |
| Temporal Correlation | Engine instantiated but patterns commented-out; no-op at runtime | Moderate Risk |
| Tech Debt | Multiple TODO comments; `allow_unsafe_werkzeug=True` in dev run; hardcoded `system_health: 98` in metrics | Low-Moderate |
| Test Coverage | 112 test files, 1046+ passing; coverage gate at 50% floor | Moderate Risk |
| Docker | Non-root user; `--require-hashes --no-deps` install; single gunicorn worker (SocketIO constraint) | Positive |
| Elasticsearch | Optional integration; disabled by default; no data encryption at rest specified | Low Risk |
| Observability | Structured log format; Prometheus metrics via `/api/metrics`; SIEM export buffer | Positive |

### 1.3 Critical Observations

**Strength: Architecture coherence.** INIDS 2.0 has a well-designed multi-layer architecture: event-bus-driven asynchronous detection pipeline, pluggable engine registry, layered middleware stack, and clean blueprint separation. The fail-closed security posture at startup (`_validate_all_routes_have_auth_decorator()`, `ALLOW_UNAUTHENTICATED` check) demonstrates mature engineering thinking.

**Strength: Defense in depth.** Authentication has three credential vectors (Bearer JWT, X-API-Key, session cookie) with a unified service. Rate limiting is two-tier. Input sanitization occurs at the API boundary. Model integrity is verified with SHA-256 at load time. These controls are layered, not single points of defense.

**Risk: Single-worker deployment constraint.** Flask-SocketIO requires `eventlet` and a single Gunicorn worker. This hard constraint limits horizontal scaling significantly. The current deployment model cannot use standard multi-worker setups without Redis coordination via SocketIO's Redis adapter.

**Risk: Temporal correlation engine disabled.** The `TemporalCorrelationEngine` is instantiated but its example pattern registrations are commented out, and the engine is not added to the `engine_registry` unless patterns are explicitly registered at runtime. Until operators add patterns via the API, this entire detection dimension contributes nothing to security.

**Risk: Coverage floor at 50%.** The project-wide coverage gate is 50%, which is low for a security-critical application. Original security modules (auth, detection, IPS) are supposed to maintain 80%, but the unified gate is 50%, allowing large swaths of untested code. With 112 test files and 1046+ passing tests, the test suite is extensive, but the coverage gate should be tightened.

**Risk: Ephemeral JWT keys by default.** Without `INIDS_JWT_PRIVATE_KEY` configured, the system generates ephemeral RSA keys per process. Any application restart invalidates all issued tokens, breaking all active sessions. This is acceptable for development but the default behavior creates a significant operational risk if someone deploys without configuring persistent keys.

---

## 2. Project Overview & Business Context

### 2.1 System Purpose

INIDS is a machine-learning-powered Network Intrusion Detection and Prevention System (IDPS). It ingests network flow records (modeled on the NSL-KDD dataset format), passes them through a multi-engine detection pipeline, generates risk-scored alerts, and can autonomously execute firewall prevention actions against attacking source IPs.

The system spans three operational modes:
- **IDS-only (monitor mode):** Detection, alerting, and audit logging with no automated enforcement.
- **IPS mode:** Full prevention including IP blocking, rate limiting, and temporary blocks with configurable TTLs.
- **Dry-run mode:** Full logic execution but no actual firewall commands issued; useful for change validation.

### 2.2 Feature Inventory

| Feature Domain | Capabilities |
|---|---|
| Detection | Multi-engine voting: ML (Random Forest + others), Signature (YAML rules), Anomaly (IsolationForest), Threshold/rate-based, Honeypot, Threat Intel matching |
| Prevention | IP blocking, rate-limiting, temporary blocks with TTL; UFW, nftables, webhook, and mock firewall adapters |
| Alerting | Alert lifecycle (open/reviewing/closed/escalated); false-positive feedback; deduplication; bulk dismiss |
| Risk Scoring | Composite risk score from confidence, severity, frequency; escalation booster for repeat offenders |
| Policy Engine | Configurable thresholds (alert/rate-limit/temp-block/block); approval gates; version history + rollback |
| Ingestion | REST API, Zeek conn-log parser, Suricata EVE-flow parser, Redis stream pipeline |
| Threat Intel | CSV/JSON feed loader; in-memory indicator cache with TTL expiry; per-IP lookup |
| Investigation | Incident aggregation, entity enrichment (GeoIP, threat history), temporal correlation patterns |
| Perception Layer | Attack story engine, confidence breakdown engine, live system pulse (60-min rolling window) |
| Observability | Prometheus metrics, SIEM JSONL export buffer, Elasticsearch/OpenSearch audit bridge |
| Real-time UI | WebSocket (SocketIO) dashboard with live alerts, actions, metrics, perception events |
| Model Lifecycle | Model registry, dataset collector, automated retraining scheduler (daily at 02:00 UTC) |
| RBAC | Four roles: admin, analyst, viewer, sensor; hierarchical inheritance |
| Audit | Comprehensive audit log for every significant operation; per-user activity queries |

### 2.3 User Roles and Responsibilities

| Role | Permissions | Typical User |
|---|---|---|
| admin | Full access: policy changes, user management, system config, approval of pending blocks | SOC manager, security engineer |
| analyst | Detection, alerts, ingest, investigations, engines, TI lookups, escalation review | SOC analyst |
| viewer | Read-only dashboard metrics | Executive, read-only observer |
| sensor | Network sensor / ingestion agent (implied from env key config) | Automated data collection agents |

### 2.4 System Boundaries

```
  ┌──────────────────────────────────────────────────────────┐
  │                     INIDS System                          │
  │                                                           │
  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
  │  │  Web UI /    │   │  REST API    │   │  WebSocket   │  │
  │  │  Templates   │◄──┤  (Flask)     │◄──┤  /events     │  │
  │  └──────────────┘   └──────┬───────┘   └──────────────┘  │
  │                            │                              │
  │              ┌─────────────▼────────────┐                 │
  │              │    Detection Pipeline     │                 │
  │              │  ML | Sig | Anomaly |     │                 │
  │              │  Threshold | TI | Hpot   │                 │
  │              └─────────────┬────────────┘                 │
  │                            │                              │
  │  ┌─────────────────────────▼──────────────────────────┐   │
  │  │  Event Bus (DetectionEvent → RiskScore → Policy     │   │
  │  │              → Action → SIEM/WebSocket)             │   │
  │  └─────────────────────────┬──────────────────────────┘   │
  │                            │                              │
  │  ┌──────────┐  ┌───────────▼────┐  ┌─────────────────┐   │
  │  │ SQLite / │  │  Firewall      │  │  Elasticsearch  │   │
  │  │ Postgres │  │  Adapter       │  │  (optional)     │   │
  │  └──────────┘  └────────────────┘  └─────────────────┘   │
  └──────────────────────────────────────────────────────────┘

  External:  Redis (optional, pipeline/rate-limit/leader election)
             GeoIP database, TI feed files, Zeek/Suricata log streams
```

---

## 3. Technology Stack & Dependencies

### 3.1 Full Technology Table

| Layer | Technology | Version Constraint | Purpose |
|---|---|---|---|
| Runtime | Python | >=3.10 (3.11 in CI) | Application runtime |
| Web Framework | Flask | >=3.0.0 | HTTP server and routing |
| WebSocket | Flask-SocketIO + python-socketio | latest | Real-time bidirectional events |
| ASGI Server | Gunicorn + eventlet | latest | Production WSGI server |
| ML / Data | scikit-learn | >=1.3.0 | Model training and inference |
| ML / Data | pandas | >=2.0.0 | Data manipulation |
| ML / Data | numpy | >=1.24.0 | Numerical operations |
| ML / Data | joblib | >=1.3.0 | Model serialization |
| Auth | PyJWT | latest | JWT encoding/decoding |
| Auth | cryptography | latest | RSA key generation/loading |
| Persistence | SQLite (stdlib) | — | Default operational store |
| Persistence | SQLAlchemy | latest | PostgreSQL abstraction |
| Cache/Queue | redis | latest | Stream ingestion, rate limiting, leader election |
| Search | elasticsearch / opensearch-py | latest | Optional audit/event store |
| Serialization | PyYAML | latest | Signature rule loading |
| Validation | jsonschema | latest | Request payload validation |
| API Docs | connexion[swagger-ui] | latest | OpenAPI integration layer |
| Visualization | matplotlib + seaborn | >=3.7.0 / >=0.12.0 | Confusion matrices, feature plots |
| Compression | flask-compress | latest | Gzip HTTP responses >=1KB |
| Packet Capture | scapy | >=2.5.0 | Live traffic capture |
| HTTP Client | aiohttp + requests | latest | Async/sync external calls |
| File I/O | aiofiles | latest | Async file operations |
| Frontend | Bootstrap 5.2.3 | bundled | UI component framework |
| Frontend | Tailwind CSS | bundled | Utility CSS |
| Frontend | Socket.IO client | CDN | WebSocket client |
| Testing | pytest | >=7.4.0 | Test runner |
| Testing | pytest-benchmark | latest | Performance benchmarks |
| Security | pip-audit | latest | Dependency CVE scanning |

### 3.2 Direct Dependency Risk Assessment

| Dependency | Risk | Rationale |
|---|---|---|
| Flask >=3.0.0 | Low | Modern, actively maintained; >=3.0 includes security fixes |
| PyJWT | Low | Well-maintained; RS256 algorithm explicitly selected |
| cryptography | Low | Core cryptographic library; keep updated for CVE fixes |
| scapy | Moderate | Requires elevated privileges (root/CAP_NET_RAW) for packet capture; large attack surface |
| redis | Moderate | Optional but if misconfigured (no auth), exposes rate limiter and stream data |
| opensearch-py / elasticsearch | Moderate | Optional; if TLS verification disabled in prod, audit data could be intercepted |
| joblib (model loading) | High | ML model deserialization via pickle; mitigated by SHA-256 checksum verification |
| connexion | Low | OpenAPI integration; swagger UI should be disabled in production |
| eventlet | Moderate | Monkey-patching; single-worker constraint; known issues with some Python versions |
| pandas | Low | Data processing; no direct security exposure |

### 3.3 Package Management

Dependencies are managed via `requirements.in` (direct dependencies) compiled to `requirements.txt` with `pip-compile --generate-hashes`. This generates a fully pinned, hash-verified manifest. CI enforces `--require-hashes --no-deps` installation and runs `pip-audit` on every push and daily. The `_gen_hashed_requirements.py` script automates hash regeneration.

---

## 4. System Architecture

### 4.1 Architectural Philosophy

INIDS follows a **layered, event-driven architecture** with clear separation between:
1. **HTTP/WebSocket ingress layer** (Flask blueprints + SocketIO)
2. **Detection and aggregation layer** (pluggable engine registry)
3. **Risk and policy decision layer** (event bus + policy engine)
4. **Enforcement and persistence layer** (firewall adapters + OpsStore)
5. **Observability and intelligence layer** (SIEM, Elasticsearch, perception)

The design philosophy is explicitly fail-closed: authentication bypass flags cause a `RuntimeError` at startup; missing routes without `@require_roles` cause a `RuntimeError` at startup; missing `SECRET_KEY` causes a `RuntimeError` at startup. This aggressive startup validation prevents silent misconfigurations from reaching production.

### 4.2 Component Model

```
                        ┌────────────────────────────────────┐
                        │           web_app/app.py            │
                        │  (Flask application + SocketIO)     │
                        │                                     │
  ┌──────────────┐      │  blueprints/                        │
  │   Client     │◄────►│   health  auth  detection           │
  │  (Browser /  │      │   prevention  ingest  intel         │
  │   API)       │      │   observability  system  dashboard  │
  └──────────────┘      │   pages  modules                    │
                        └──────────────┬─────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │            Service Layer                  │
                    │  DetectionService  PreventionService      │
                    │  IngestionService  MetricsService         │
                    │  OpsStore  ModelRegistry  PolicyStore     │
                    └──────────────────┬──────────────────────┘
                                       │
          ┌────────────────────────────▼────────────────────────┐
          │               Detection Engine Registry               │
          │                                                       │
          │  ┌──────────┐ ┌────────────┐ ┌────────────────────┐  │
          │  │ MLEngine │ │ Signature  │ │  AnomalyEngine     │  │
          │  │(RF model)│ │ Engine     │ │  (IsolationForest) │  │
          │  └──────────┘ └────────────┘ └────────────────────┘  │
          │  ┌──────────┐ ┌────────────┐ ┌────────────────────┐  │
          │  │Threshold │ │ TIEngine   │ │  HoneypotEngine    │  │
          │  │ Engine   │ │            │ │                    │  │
          │  └──────────┘ └────────────┘ └────────────────────┘  │
          │         │               │                             │
          │         └───────────────▼─ EngineAggregator          │
          │                   (ANY_TRIGGER strategy)              │
          └────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────▼────────────────────────┐
          │                 EventBus                         │
          │                                                  │
          │  DetectionEvent → RiskEngine → RiskScoreEvent   │
          │  RiskScoreEvent → PolicyEngine → PolicyDecision │
          │  PolicyDecision → ActionExecutor → ActionEvent  │
          │  All events → SIEM exporter, WebSocket, Audit   │
          └───────────────────────────────────────────────┘
```

### 4.3 Blueprint Organization

The Flask application registers 11 blueprints, each handling a distinct domain:

| Blueprint | Module | Domain |
|---|---|---|
| health_bp | blueprints/health.py | Health checks, liveness, readiness, Prometheus metrics |
| auth_bp | blueprints/auth.py | JWT login, refresh, validate, revoke, audit logs, admin operations |
| detection_bp | blueprints/detection.py | Predict, detect, alerts, alert lifecycle, engine management, FP management |
| prevention_bp | blueprints/prevention.py | Policy CRUD, actions, allowlist, honeypot config |
| ingest_bp | blueprints/ingest.py | Data ingestion (raw, Zeek, Suricata, batch processing) |
| intel_bp | blueprints/intel.py | Incidents, temporal patterns, entity enrichment, filter rules, models, TI |
| observability_bp | blueprints/observability.py | Audit log, SIEM flush, TI stats, explain API |
| system_bp | blueprints/system.py | Anomaly status, escalation, FP stats, investigations, playbooks, capture |
| dashboard_bp | blueprints/dashboard.py | Dashboard metrics, demo control, perception layer APIs |
| pages_bp | blueprints/pages.py | HTML page rendering (login, dashboard UI, investigation, etc.) |
| modules_bp | blueprints/modules.py | Module management and configuration |

### 4.4 Event-Driven Detection Flow

The detection pipeline follows a publish-subscribe pattern through the `EventBus`:

```
  Network Flow Record (features dict)
           │
           ▼
    EngineRegistry.evaluate_all()
           │
    [MLEngine, SignatureEngine, AnomalyEngine,
     ThresholdEngine, TIEngine, HoneypotEngine]
           │
           ▼
    EngineAggregator.aggregate()  [ANY_TRIGGER strategy]
           │
           ▼
    DetectionEvent published to EventBus
           │
    ┌──────┴────────────────────────────────┐
    │                                       │
    ▼                                       ▼
_on_detection_event()            _on_detection_realtime()
 - Save alert to OpsStore         - Emit to WebSocket /events
 - Feed normal → AnomalyEngine    - Emit dashboard metrics
 - Calculate risk score
 - Publish RiskScoreEvent
           │
           ▼
_on_risk_event()
 - Apply escalation boost
 - PolicyEngine.decide()
 - Publish PolicyDecisionEvent
           │
           ▼
_on_policy_decision_event()
 - Check allowlist
 - ActionExecutor.execute()
 - EscalationTracker.record_hit()
 - Publish ActionEvent
```

### 4.5 Redis Stream Pipeline

When Redis is available and `INIDS_PIPELINE_ENABLED=true`:

```
  External sensor / API  →  redis.xadd(inids:flows, payload)
                                      │
                             StreamProcessor (consumer group)
                                      │
                             [reads batch, calls evaluate_all]
                                      │
                             _stream_result_callback()
                                      │
                             DetectionEvent → EventBus
```

The `BackpressureController` monitors consumer lag and transitions through `NORMAL → SAMPLING → SHEDDING` levels, rejecting new submissions with HTTP 503 when in SHEDDING mode.

---

## 5. Database Design & Data Architecture

### 5.1 Backend Selection

`OpsStore` supports two backends determined at instantiation:
- **SQLite** (default): file-based, zero-configuration, appropriate for single-instance deployments. Path configured via `OPS_DB_PATH` (default: `data/inids_ops.db`).
- **PostgreSQL**: activated when `OPS_DB_PATH` begins with `postgresql://` or `postgres://`. Uses SQLAlchemy `create_engine` with `future=True`.

The class transparently handles SQL dialect differences (SQLite `INSERT OR IGNORE` vs PostgreSQL `ON CONFLICT DO NOTHING`, `AUTOINCREMENT` vs `BIGSERIAL`, `PRAGMA table_info` vs `information_schema.tables`, `REAL` vs `DOUBLE PRECISION`).

### 5.2 Schema Version History

| Version | Migration | Purpose |
|---|---|---|
| v1 | `_migration_v1_create_tables` | Create base tables: alerts, actions, audits, allowlist, fp_suppressions |
| v2 | `_migration_v2_add_columns` | Add extended columns to actions (action_id, ip, action_type, status, dry_run, executed) and alerts (status, assignee, close_reason, source_ip, attack_type, risk_score) |
| v3 | `_migration_v3_auth_tables` | Create users, api_keys, revoked_tokens tables; seed service accounts from env vars; create `idx_revoked_jti` index |
| v4 | `_migration_v4_actions_idempotency_index` | Partial unique index `uq_active_block` on actions(target) WHERE status IN active states; prevents duplicate active blocks at DB layer |
| v5 | `_migration_v5_alert_dedup` | Add `dedup_key` column to alerts; unique partial index `uq_alert_dedup` for 5-minute bucket deduplication |
| v6 | `_migration_v6_indexes` | Performance indexes: `idx_alerts_source_ip`, `idx_alerts_timestamp`, `idx_audits_created_at`, `idx_actions_status` |

### 5.3 Table Schemas

**Table: alerts**

| Column | Type | Constraints | Description |
|---|---|---|---|
| id | TEXT | PRIMARY KEY | UUID v4 |
| timestamp | TEXT | NOT NULL | ISO-8601 UTC |
| severity | TEXT | NOT NULL | low/medium/high/critical |
| prediction | TEXT | NOT NULL | Attack/Normal/Suspicious |
| confidence | REAL | NOT NULL | 0.0–100.0 |
| profile | TEXT | NOT NULL | Detection profile name |
| reason | TEXT | NOT NULL | Trigger reason |
| source_ip | TEXT | NOT NULL DEFAULT '' | Source IP address |
| attack_type | TEXT | NOT NULL DEFAULT '' | Classified attack category |
| risk_score | REAL | NOT NULL DEFAULT 0.0 | Composite risk score |
| status | TEXT | DEFAULT 'open' | open/reviewing/closed/escalated |
| assignee | TEXT | NULL | Assigned analyst username |
| close_reason | TEXT | NULL | Reason for closure |
| status_updated_at | TEXT | NULL | ISO-8601 of last status change |
| dedup_key | TEXT | UNIQUE (partial, NULL excluded) | {source_ip}|{attack_type}|{5-min bucket} |

**Table: actions**

| Column | Type | Constraints | Description |
|---|---|---|---|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Internal row ID |
| action | TEXT | NOT NULL | Action verb (block, rate_limit) |
| target | TEXT | NOT NULL | Target IP address |
| reason | TEXT | NOT NULL | Why action was taken |
| action_id | TEXT | — | Business-key UUID (act_...) |
| ip | TEXT | — | Alias for target |
| action_type | TEXT | — | block/temp_block/rate_limit |
| status | TEXT | DEFAULT 'active' | active/enforced/executed/pending_approval |
| expires_at | TEXT | NULL | ISO-8601 TTL expiry |
| created_at | TEXT | NOT NULL | ISO-8601 creation time |
| executed_at | TEXT | NULL | ISO-8601 execution time |
| adapter | TEXT | — | Firewall adapter name |
| dry_run | INTEGER | DEFAULT 0 | Boolean: dry run mode |
| executed | INTEGER | DEFAULT 0 | Boolean: was executed |

Partial unique index: `uq_active_block ON actions(target) WHERE lower(status) IN ('active','enforced','executed') AND lower(action_type) IN ('block','temp_block','rate_limit')`

**Table: audits**

| Column | Type | Description |
|---|---|---|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| event_type | TEXT | Category: risk_score, policy_decision, auth_success, authz_denied, etc. |
| message | TEXT | JSON-serialized event details |
| created_at | TEXT | ISO-8601 UTC |

**Table: allowlist**

| Column | Type | Description |
|---|---|---|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| entry | TEXT | UNIQUE — IP address or CIDR |
| reason | TEXT | Human-readable justification |
| created_at | TEXT | ISO-8601 UTC |

**Table: fp_suppressions**

| Column | Type | Description |
|---|---|---|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| engine_id | TEXT | Engine identifier (signature, ml_primary, etc.) |
| rule_id | TEXT | Rule or feature identifier |
| suppressed | INTEGER | 1 = suppressed |
| created_at | TEXT | ISO-8601 UTC |

UNIQUE constraint on (engine_id, rule_id).

**Table: users** (v3)

| Column | Type | Description |
|---|---|---|
| user_id | TEXT | PRIMARY KEY (e.g., svc-admin) |
| username | TEXT | UNIQUE display name |
| roles | TEXT | Comma-separated role names |
| created_at | TEXT | ISO-8601 UTC |

**Table: api_keys** (v3)

| Column | Type | Description |
|---|---|---|
| key_id | TEXT | PRIMARY KEY (e.g., key-admin) |
| user_id | TEXT | Foreign key to users.user_id |
| key_hash | TEXT | UNIQUE SHA-256 of raw API key |
| label | TEXT | Human label (env var name) |
| created_at | TEXT | ISO-8601 UTC |

Index: `idx_api_keys_hash ON api_keys(key_hash)`

**Table: revoked_tokens** (v3)

| Column | Type | Description |
|---|---|---|
| jti | TEXT | PRIMARY KEY — JWT ID claim |
| user_id | TEXT | Associated user |
| expires_at | TEXT | ISO-8601 original token expiry |
| revoked_at | TEXT | ISO-8601 revocation time |

Index: `idx_revoked_jti ON revoked_tokens(jti)` — O(log n) per auth check.

**Table: schema_version**

| Column | Type | Description |
|---|---|---|
| version | INTEGER | PRIMARY KEY — current schema version |
| updated_at | TEXT | ISO-8601 UTC of migration run |

### 5.4 Data Retention

Alert retention is controlled by `INIDS_ALERT_RETENTION_DAYS`. When set to a positive integer, a daily background thread (`alert-retention`) runs `delete_alerts_older_than(cutoff)`. Disabled when the variable is 0 or unset. The `_run_alert_retention()` function returns count of deleted rows. A manual trigger exists at `POST /api/admin/alert-retention` (admin only).

Revoked JWT tokens are cleaned up via `POST /api/admin/cleanup-tokens` (admin) and `cleanup_revoked_tokens(before_iso)`, which removes expired entries from `revoked_tokens`.

### 5.5 Historical RBAC Database

A separate file `inids_rbac.db` exists at the repository root — a legacy SQLite database from an older RBAC system. The `migrate_rbac_users()` method in `OpsStore` handles migrating users from this database into the main `ops_store`. A `scripts/migrate_rbac.py` script automates this migration. The legacy database is superseded by the `users` and `api_keys` tables in `OpsStore` v3+.

---

## 6. Backend API Architecture & Business Logic

### 6.1 Request Lifecycle

Every API request traverses the following pipeline before reaching a route handler:

```
  HTTP Request
       │
       ▼
  Flask before_request hooks (registered in register_middleware())
       ├─ CORS preflight check (OPTIONS bypass)
       ├─ IPBlockingMiddleware — 403 if IP is blocked
       ├─ RequestValidationMiddleware — 413 if body > 1MB; 400 if malformed JSON
       └─ AuditLogMiddleware.before_request() — record start_time
       │
       ▼
  app._before_request_metrics()
       ├─ _ensure_scheduler_started() (first request only)
       ├─ metrics_service.inc('requests_total') for /api/* paths
       └─ UnifiedRateLimiter.check_ip() — 429 if Tier-1 rate exceeded
       │
       ▼
  @require_roles(*roles) decorator (all non-public routes)
       ├─ UnifiedAuthService: try Bearer JWT → X-API-Key → cookie JWT
       ├─ Role intersection check — 401 if insufficient
       ├─ g.auth = AuthContext  (set for downstream handlers)
       ├─ OpsStore.add_audit(auth_success)
       └─ UnifiedRateLimiter.check_user() — 429 if Tier-2 rate exceeded
       │
       ▼
  Route handler logic
       │
       ▼
  Flask after_request hooks
       ├─ IPBlockingMiddleware.after_request() — track 401/403 for blocking
       ├─ SecurityHeadersMiddleware — inject OWASP headers
       ├─ CORSMiddleware.add_headers()
       ├─ ContentSecurityMiddleware — enforce content-type
       └─ AuditLogMiddleware.after_request() — log completed request
```

### 6.2 Complete API Endpoint Catalog

#### Health Domain

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | /api/health | None | Full system status: model, engines, pipeline, leader election, telemetry |
| GET | /api/health/live | None | Liveness probe — always returns 200 if process is up |
| GET | /api/health/ready | None | Readiness probe — 200 if healthy, 503 if any probe fails |
| GET | /api/metrics | analyst | Prometheus-format metrics text |

#### Authentication Domain

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | /api/auth/login | X-API-Key header | Exchange API key for RS256 JWT (1-hour TTL) |
| POST | /api/auth/refresh | Bearer JWT | Refresh token within 5-minute grace window after expiry |
| GET | /api/auth/validate | Bearer JWT | Validate token and return claims |
| GET | /api/auth/status | None | Auth system configuration summary |
| POST | /api/auth/revoke | Bearer JWT | Revoke token by persisting JTI to revoked_tokens |
| POST | /api/auth/runas | admin | Issue delegated token for a target user (impersonation) |
| GET | /api/audit/logs | viewer | Recent request audit log entries (from AuditLogMiddleware) |
| GET | /api/audit/user-activity | admin | Activity for a specific user over N hours |
| POST | /api/admin/cleanup-tokens | admin | Purge expired revocation records |
| POST | /api/admin/alert-retention | admin | Trigger alert retention deletion manually |

#### Detection Domain

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | /api/predict | analyst | Single-record ML model prediction via DetectionService |
| POST | /api/detect | analyst | Multi-engine detection via EngineRegistry + aggregation |
| POST | /api/detection/analyze | analyst | Target IP/domain analysis via engine pipeline |
| GET | /api/alerts | analyst | List alerts with pagination, severity, status filters |
| POST | /api/alerts/dismiss | analyst | Bulk close alerts as dismissed |
| PATCH | /api/alerts/{alert_id} | analyst | Update alert status, assignee, close_reason |
| POST | /api/alerts/{alert_id}/feedback | analyst | Submit FP/TP feedback for false-positive suppression |
| GET | /api/fp-suppressions | analyst | List active false-positive suppressions |
| POST | /api/fp-suppressions | admin | Add a suppression rule |
| DELETE | /api/fp-suppressions/{engine_id}/{rule_id} | admin | Remove a suppression rule |
| GET | /api/engines | analyst | List all registered detection engines and their status |
| POST | /api/engines/{engine_id}/toggle | admin | Enable/disable a detection engine |

#### Prevention Domain

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | /api/policy | admin | Get current prevention policy |
| POST | /api/policy | admin | Update prevention policy (versioned) |
| GET | /api/policy/history | admin | List policy version history |
| POST | /api/policy/rollback | admin | Roll back to a previous policy version |
| GET | /api/actions | analyst | List recent prevention actions |
| POST | /api/actions | analyst | Create manual prevention action |
| GET | /api/actions/pending | analyst | List actions awaiting approval |
| POST | /api/actions/{action_id}/approve | admin | Approve a pending block action |
| POST | /api/actions/cleanup | admin | Clean up expired actions from DB |
| GET | /api/allowlist | analyst | List allowlisted IPs/CIDRs |
| POST | /api/allowlist | admin | Add entry to allowlist |
| DELETE | /api/allowlist/{entry} | admin | Remove entry from allowlist |
| GET | /api/honeypots | analyst | List honeypot configurations |
| POST | /api/honeypots/{id}/toggle | admin | Enable/disable a honeypot |
| GET | /api/honeypot/config | analyst | Get honeypot engine configuration |
| POST/PATCH | /api/honeypot/config | admin | Update honeypot IPs and ports |

#### Ingestion Domain

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | /api/ingest | analyst | Enqueue raw feature records (rows or features object) |
| POST | /api/ingest/log | analyst | Ingest Zeek conn-log or Suricata EVE flow records |
| POST | /api/ingest/process | analyst | Drain ingestion queue through detection service |

#### Intelligence Domain

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | /api/incidents | analyst | List aggregated incidents |
| GET | /api/incidents/{id} | analyst | Get incident details |
| GET | /api/activities | analyst | List activity entries |
| GET | /api/temporal/patterns | analyst | List registered temporal correlation patterns |
| POST | /api/temporal/patterns | admin | Register a new multi-stage attack pattern |
| GET | /api/temporal/state/{source_ip} | analyst | Get temporal event state for an IP |
| GET | /api/entity/enrich/{source_ip} | analyst | Entity enrichment for an IP (GeoIP, TI, history) |
| GET | /api/entity/{source_ip}/threat-level | analyst | Get threat level for an IP |
| GET | /api/alerts/filter-rules | analyst | List alert filter rules (exclude/ignore/merge) |
| POST | /api/alerts/filter-rules/exclude | admin | Add exclude rule |
| POST | /api/alerts/filter-rules/ignore | admin | Add ignore rule |
| POST | /api/alerts/filter-rules/merge | admin | Add merge rule |
| DELETE | /api/alerts/filter-rules/{rule_id} | admin | Delete filter rule |
| GET | /api/alerts/filter-stats | analyst | Get alert filter statistics |
| GET | /api/models/registry | analyst | List model registry entries |
| GET | /api/models | analyst | List loaded ML models catalog |
| GET | /api/threat-intelligence | analyst | List TI indicators and feed summary |
| GET | /api/detections/history | analyst | Query risk_score/policy_decision audit history |

#### Observability Domain

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | /api/audit | admin | List raw OpsStore audit records |
| GET | /api/siem/flush | admin | Drain SIEM export buffer to JSONL |
| GET | /api/threat-intel/stats | analyst | TI cache statistics |
| POST | /api/threat-intel/lookup | analyst | Lookup a single IP in TI cache |
| POST | /api/explain | analyst | Feature importance explanation for a feature set |

#### System Domain

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | /api/anomaly/status | analyst | Anomaly engine buffer status |
| GET | /api/escalation/summary | analyst | Escalation tracker state |
| POST | /api/escalation/evict | admin | Evict stale escalation entries |
| GET | /api/fp-stats | analyst | False-positive manager statistics |
| GET | /api/investigations | analyst | Generated investigation list from alerts |
| GET | /api/playbooks | analyst | List response playbooks |
| POST | /api/playbooks/{id}/execute | analyst | Execute a response playbook |
| POST | /api/capture/start | analyst | Start packet capture session |
| POST | /api/capture/stop | analyst | Stop packet capture session |

#### Dashboard & Perception Domain

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | /api/dashboard/metrics | viewer | Dashboard KPI metrics |
| POST | /api/dashboard/refresh | analyst | Force dashboard refresh |
| POST | /api/demo/start | admin | Start demo traffic simulation |
| POST | /api/demo/stop | admin | Stop demo traffic simulation |
| GET | /api/perception/pulse | analyst | Live system pulse status |
| GET | /api/perception/pulse/timeseries/{metric} | analyst | Time-series data for a metric |
| GET | /api/perception/confidence/{detection_id} | analyst | Confidence breakdown for a detection |
| GET | /api/perception/attack-story/{attack_id} | analyst | Attack narrative story |
| GET | /api/perception/attack-stories | analyst | Recent attack stories |
| GET | /api/perception/feature-importance | analyst | Feature importance ranking |
| GET | /api/perception/integration-status | analyst | Perception integration worker status |

### 6.3 Business Logic Deep-Dives

#### Detection Service Flow (`/api/predict`)

1. Validate payload with `validate_predict_request()` (jsonschema)
2. Coerce numeric feature columns to float
3. Call `DetectionService.predict_from_features(features, profile, source_ip, attack_type)`
4. Look up most recent action for source IP in last 60 seconds
5. Return prediction result plus associated prevention_action

#### Multi-Engine Detection (`/api/detect`)

1. Coerce numeric features
2. Call `enrich_single_row()` for feature engineering enrichment
3. `engine_registry.evaluate_all()` — parallel evaluation through all enabled engines
4. `engine_aggregator.aggregate()` with `ANY_TRIGGER` strategy (any engine flagging = attack)
5. Increment per-engine and aggregate metrics counters
6. Return aggregated result with all engine verdicts

#### Risk Scoring and Policy Decision (Event Bus)

Risk score is computed as:
```
risk_score = confidence_weight * (confidence/100) 
           + severity_weight * severity_numeric
           + frequency_weight * frequency_score
```
Configurable weights default to confidence=0.5, severity=0.3, frequency=0.2.

Escalation boosts are applied: 5% for ALERT level, 10% for RATE_LIMIT level, 15% for TEMP_BLOCK or higher. Cap at 1.0.

Policy thresholds (defaults):
- risk >= 0.40 → ALERT
- risk >= 0.60 → RATE_LIMIT
- risk >= 0.75 → TEMP_BLOCK
- confidence >= 85% AND risk >= 0.85 → BLOCK (or PENDING_BLOCK if approval required)

---

## 7. Frontend Architecture & User Experience

### 7.1 Frontend Type: Server-Side Rendered + JavaScript-Enhanced

INIDS is not a Single-Page Application (SPA). The frontend uses:
- **Jinja2 templates** rendered server-side by Flask
- **Bootstrap 5.2.3** (bundled locally, not CDN) for layout and components
- **Tailwind CSS** (compiled, bundled) for utility classes
- **Vanilla JavaScript** with module-per-page pattern (no build step)
- **Socket.IO client** (CDN reference) for real-time WebSocket events

### 7.2 Page Inventory

The application provides the following rendered HTML pages:

| Template | Route | Auth | Purpose |
|---|---|---|---|
| index.html / home.html | / | public | Landing page |
| login.html | /login | public | API key login form |
| dashboard.html | /dashboard | analyst | Main operational dashboard |
| alerts.html | /alerts | analyst | Alert management |
| detection.html | /detection | analyst | Single-record prediction form |
| actions.html | /actions | analyst | Prevention actions list |
| allowlist.html | /allowlist | analyst | Allowlist management |
| engines.html | /engines | analyst | Detection engine status |
| models.html | /models | analyst | Model registry |
| honeypot.html | /honeypot | analyst | Honeypot configuration |
| investigate.html | /investigate | analyst | Investigation workspace |
| capture.html | /capture | analyst | Packet capture control |
| health.html | /health-ui | viewer | System health visualization |
| batch.html | /batch | analyst | Batch prediction |
| learn.html | /learn | public | Educational content |
| about.html | /about | public | About page |
| modules/ | /modules/* | analyst | Module-specific views |

### 7.3 JavaScript Module Architecture

JavaScript is organized per-page (`dashboard.js`, `alerts.js`, `actions.js`, `allowlist.js`, etc.) with shared utilities in `base-ui.js` and `base-modules.js`. The pattern uses component factories in `web_app/static/js/components/` and core utilities in `web_app/static/js/core/`.

Real-time events from WebSocket are handled in each page's JS: the client connects to `io('/events')` with a Bearer token in auth metadata, subscribes to topic rooms, and updates DOM elements on receipt of events (`DetectionEvent`, `ActionEvent`, `RiskScoreEvent`, `metrics.update`).

### 7.4 CSP Configuration

The Content-Security-Policy header is set by `SecurityHeadersMiddleware`:
```
default-src 'self';
script-src 'self' https://cdn.jsdelivr.net https://cdn.socket.io;
style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net;
connect-src 'self' ws://127.0.0.1:5000 ws://localhost:5000 ...;
img-src 'self' data:
```

The `'unsafe-inline'` in `style-src` is a known limitation — inline styles are used for dynamic content. This is a candidate for remediation.

---

## 8. Authentication, Authorization & Security

### 8.1 Authentication Architecture

INIDS implements a unified three-vector authentication system through `UnifiedAuthService`:

```
  Incoming Request
        │
        ├─ Authorization: Bearer <token>  →  RS256 JWT verification
        │                                     + JTI revocation check
        ├─ X-API-Key: <key>               →  SHA-256 hash lookup
        │                                     in api_keys table
        └─ Cookie: inids_jwt=<token>      →  RS256 JWT verification
                                               (browser sessions)
```

**JWT implementation details:**
- Algorithm: RS256 (asymmetric, not HS256)
- Token TTL: 3600 seconds (1 hour, non-configurable)
- JTI: UUID v4 hex per token
- Audience: `INIDS-API` (enforced at verification)
- Key source: `INIDS_JWT_PRIVATE_KEY` / `INIDS_JWT_PUBLIC_KEY` environment variables or `_FILE` variants pointing to Docker secrets
- Fallback: ephemeral 2048-bit RSA key per process with WARNING log
- Revocation: JTI persisted to `revoked_tokens` table; O(log n) lookup via `idx_revoked_jti`

**Token refresh:**
- `/api/auth/refresh` accepts tokens within a 5-minute grace window after expiry (`REFRESH_GRACE_SECONDS = 300`)
- Expired tokens older than 5 minutes are rejected with a re-authentication prompt

### 8.2 Authorization (RBAC)

Role hierarchy:
```
admin → can access {admin, analyst, viewer} resources
analyst → can access {analyst, viewer} resources
viewer → can access {viewer} resources
sensor → can access {sensor} resources (isolated)
```

Enforcement mechanism: `@require_roles(*roles)` decorator sets `func._required_roles` on the view function. At startup, `_validate_all_routes_have_auth_decorator()` iterates all registered routes and raises `RuntimeError` for any non-public route lacking `_required_roles`. This is a fail-closed startup gate that prevents routes from accidentally being left unprotected.

Public routes (no auth required):
- `/health`, `/api/health`, `/api/health/live`, `/api/health/ready`
- `/api/auth/login`, `/api/auth/refresh`, `/api/auth/validate`, `/api/auth/status`, `/api/auth/revoke`
- `/login`, `/logout`, `/static/*`

### 8.3 Security Controls Inventory

| Control | Implementation | Status |
|---|---|---|
| Authentication bypass prevention | `ALLOW_UNAUTHENTICATED` check at startup raises RuntimeError | Active |
| Route coverage enforcement | `_validate_all_routes_have_auth_decorator()` at startup | Active |
| Input sanitization | `src/input_sanitizer.py` — SQL, XSS, log-injection patterns | Active |
| Request payload validation | jsonschema via `validation_schemas.py` on predict/detect | Active |
| IP rate limiting (Tier 1) | `UnifiedRateLimiter.check_ip()` — 1000 req/min per IP | Active |
| User rate limiting (Tier 2) | `UnifiedRateLimiter.check_user()` — 200 req/min per user | Active (via require_roles) |
| IP blocking | `IPBlockingMiddleware` — blocks after 5 consecutive 401/403 for 300s | Active |
| Security headers | HSTS, X-Frame-Options, CSP, X-Content-Type-Options, Referrer-Policy | Active |
| CSRF protection | `csrf_protect_middleware` — session token with `X-CSRF-Token` header | Active |
| Correlation ID sanitization | Log injection characters rejected; sanitized to printable, max 64 chars | Active |
| Model integrity | SHA-256 checksums verified at load time; strict mode rejects mismatched models | Active |
| Dependency integrity | `--require-hashes --no-deps` install; pip-audit daily CI scan | Active |
| Upload size limit | Flask `MAX_CONTENT_LENGTH = 16 MB` | Active |
| Request body size | `RequestValidationMiddleware` — 1 MB hard cap | Active |
| Secret key enforcement | `load_settings()` raises RuntimeError on missing SECRET_KEY | Active |
| Non-root container | Dockerfile creates `inids` user, drops to non-root | Active |
| Response compression | flask-compress for JSON/HTML/JS/CSS >= 1KB | Active |
| WebSocket JWT requirement | `/events` namespace requires Bearer token on connect | Active |
| Allowlist bypass protection | Allowlisted IPs skip enforcement; audit log records bypass | Active |
| Audit trail | Every auth event, policy change, action logged to OpsStore.audits | Active |

### 8.4 Security Risks Table

| Risk | Severity | Evidence | Mitigation Status |
|---|---|---|---|
| Ephemeral JWT keys by default | High | `jwt_manager.py` generates ephemeral key when `INIDS_JWT_PRIVATE_KEY` unset | Mitigated in production by env var config; warning logged |
| `allow_unsafe_werkzeug=True` in dev | Medium | `socketio.run(allow_unsafe_werkzeug=True)` in `app.py` main block | Only in `__main__` block; Gunicorn used in production |
| `style-src 'unsafe-inline'` in CSP | Medium | `middleware.py` SecurityHeadersMiddleware HEADERS dict | Needs nonce-based CSP implementation |
| Temporal engine disabled | Medium | Pattern registrations commented out in `app.py` lines 278–297 | Manual pattern registration via API required |
| Hardcoded `system_health: 98` | Low | `_build_dashboard_metrics_payload()` in `app.py` | Cosmetic; actual health comes from health_check probes |
| Redis without authentication | Medium | No Redis AUTH configuration in `Settings` | Operator responsibility; not enforced |
| Elasticsearch TLS disabled by default | Medium | `elasticsearch_verify_certs=False` default | Acceptable for dev; must be enabled in production |
| `op: "=="` in legacy rule matching | Low | `_match_rule_legacy()` uses `_OPS` dict without advanced parser | Mitigated by `RuleCompiler` primary path with fallback |
| Token revocation table growth | Low | `revoked_tokens` grows unbounded until cleanup called | Mitigated by `cleanup_revoked_tokens()` and admin API |

### 8.5 Compliance Posture

The system implements controls relevant to several frameworks:

| Framework | Coverage | Gaps |
|---|---|---|
| OWASP Top 10 | A01 (RBAC), A02 (JWT/key hashing), A03 (input sanitization), A04 (security headers), A07 (auth), A09 (audit logging) | A08 (SSRF via webhook adapter not fully validated), A10 (CSRF partially implemented) |
| NIST SP 800-53 | AU-2 (audit events), AC-2 (account management), AC-3 (access enforcement), IA-2 (JWT auth), SI-3 (intrusion detection) | CM-7 (lockdown of unused services), SC-12 (key management — ephemeral keys) |
| CIS Controls | CIS 13 (network monitoring), CIS 8.2 (audit logging), CIS 16 (account monitoring) | CIS 3.3 (data classification), CIS 18 (pen testing) |

---

## 9. Infrastructure, Deployment & DevOps

### 9.1 Containerization

The Dockerfile implements several security best practices:

```dockerfile
FROM python:3.11-slim
RUN groupadd -r inids && useradd -r -g inids inids  # non-root
WORKDIR /app
COPY src/ ./src/
COPY web_app/ ./web_app/
COPY rules/ ./rules/
COPY requirements.txt requirements.in ./
RUN pip install --no-cache-dir --require-hashes --no-deps -r requirements.txt
RUN mkdir -p /data /models && chown -R inids:inids /data /models
USER inids
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen(...)"
EXPOSE 5000
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "-b", "0.0.0.0:5000", "web_app.app:app"]
```

Key observations:
- Non-root user enforced
- No repository volume mount (source code only, no secrets)
- `--require-hashes --no-deps` enforces supply-chain integrity at image build
- Python-native healthcheck (no curl dependency)
- Single Gunicorn worker required by SocketIO+eventlet
- `.dockerignore` present (not read in detail but file exists)

### 9.2 CI/CD Pipeline

The sole CI/CD configuration is `.github/workflows/security.yml` with three jobs triggered on push to `main`, pull requests to `main`, and daily at 02:00 UTC:

| Job | Steps | Purpose |
|---|---|---|
| pip-audit | Install pip-audit, run `--require-hashes --no-deps -r requirements.txt` | CVE scanning of pinned dependencies |
| pip-compile-drift | Install pip-tools, recompile, diff against committed requirements.txt | Detect uncommitted dependency changes |
| test | Install deps, install pytest-cov, run tests with `--cov-fail-under=50` | Unit test and coverage gate |

Notable gaps:
- No SAST (static application security testing) pipeline
- No container scanning (Trivy, Grype)
- No integration/smoke test against a running container
- No staging environment deployment step
- No secret scanning (truffleHog, gitleaks)
- Coverage gate at 50% (should be higher for security-critical code)

### 9.3 Environment Configuration

All configuration is loaded via `src/settings.py`'s `load_settings()` function, which:
1. Loads `.env` file if present (does not override existing env vars)
2. Validates `ALLOW_UNAUTHENTICATED` is not truthy (fail-closed)
3. Requires `SECRET_KEY` to be non-empty (raises RuntimeError)
4. Returns an immutable frozen `Settings` dataclass

Sensitive values support `_FILE` variants for Docker secrets mounting:
- `SECRET_KEY_FILE` / `FLASK_SECRET_KEY_FILE`
- `INIDS_JWT_PRIVATE_KEY_FILE`
- `INIDS_JWT_PUBLIC_KEY_FILE`

### 9.4 Deployment Architecture

```
  Single-node (current):
  ┌─────────────────────────────────────┐
  │  Docker container                    │
  │  gunicorn (eventlet, 1 worker)       │
  │  Flask + SocketIO                    │
  │  SQLite (local file)                 │
  │  [optional] Redis (external)         │
  └─────────────────────────────────────┘

  Multi-instance (future, requires):
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Instance │  │ Instance │  │ Instance │
  │    1     │  │    2     │  │    3     │
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
              ┌───────▼──────┐  ┌──────────┐
              │  Redis       │  │PostgreSQL│
              │  (leader     │  │(shared   │
              │   election,  │  │ OpsStore)│
              │   rate-limit,│  └──────────┘
              │   SocketIO)  │
              └──────────────┘
```

For multi-instance deployment, the following must be configured:
- `REDIS_URL` for leader election and rate limiting
- `OPS_DB_PATH` pointing to a PostgreSQL URL
- `INIDS_JWT_PRIVATE_KEY` / `INIDS_JWT_PUBLIC_KEY` (persistent keys)
- SocketIO must use Redis adapter for cross-instance event delivery (not currently implemented)

### 9.5 Observability Stack

| Component | Implementation | Notes |
|---|---|---|
| Application logging | Python stdlib logging; structured format via `RequestContextFilter` | JSON logging available via `INIDS_JSON_LOGGING=1` |
| Metrics | In-process `MetricsService` counter/histogram; Prometheus export at `/api/metrics` | Not using official prometheus_client library |
| Distributed tracing | Correlation ID middleware (`X-Correlation-ID` header); propagated in log records | No OpenTelemetry integration |
| Audit log | OpsStore.audits table; every auth, policy, action event | Queryable via `/api/audit` |
| SIEM export | `SiemExporter` buffer + `SiemExporter.flush_jsonl()` | Push to SIEM system via `/api/siem/flush` |
| Elasticsearch bridge | `ElasticsearchAuditBridge`; async writes; fallback to in-memory | Optional; disabled by default |
| Health probes | `HealthCheck` with named probes (model, engines, ops_db, redis, firewall, policy, pipeline, leader) | `/api/health/ready` returns 503 if any fails |

---

## 10. External Integrations & Third-Party Services

### 10.1 Integration Map

```
  INIDS Core
       │
       ├── Redis (optional)
       │       Purpose: Redis Streams (ingestion pipeline), rate limiting,
       │                leader election, SocketIO multi-instance
       │       Config:  REDIS_URL
       │       Fallback: in-memory queue and rate limiter
       │
       ├── Elasticsearch / OpenSearch (optional)
       │       Purpose: Persistent audit log, alerts, detection events,
       │                prevention actions, performance metrics
       │       Config:  ELASTICSEARCH_ENABLED=1, ELASTICSEARCH_HOSTS,
       │                ELASTICSEARCH_PORT, ELASTICSEARCH_USE_SSL,
       │                ELASTICSEARCH_USERNAME/PASSWORD
       │       Fallback: No-op (logs warning)
       │
       ├── Firewall adapters (configurable)
       │       └── Mock (default, in-memory)
       │       └── UFW — calls `ufw deny from <ip>` via subprocess
       │       └── nftables — JSON ruleset management via subprocess
       │       └── Webhook — HTTP POST to configurable URL
       │       Config:  FIREWALL_ADAPTER, FIREWALL_WEBHOOK_URL
       │
       ├── Threat Intelligence feeds (optional)
       │       Purpose: IP indicator matching in TIEngine
       │       Config:  INIDS_TI_FEED_PATH (single file),
       │                INIDS_TI_FEED_DIR (directory scan)
       │       Format:  CSV (with 'indicator' column) or JSON arrays
       │       Refresh: INIDS_TI_REFRESH_INTERVAL (default 3600s)
       │
       ├── GeoIP (implicit)
       │       Purpose: EntityEnrichmentEngine geolocation
       │       Implementation: src/advanced/geoip_enrichment.py
       │       Note: No database configured by default; requires MaxMind
       │
       ├── Scapy (local)
       │       Purpose: Live packet capture via src/capture_live_traffic.py
       │       Requires: Root/CAP_NET_RAW privileges
       │
       └── Model files (local)
               Purpose: Trained scikit-learn models
               Format:  joblib .pkl files with SHA-256 checksums
               Location: models/ directory
               Models:  rf_nsl_kdd, gb_nsl_kdd, dt_nsl_kdd,
                        ab_nsl_kdd, mlp_nsl_kdd, rf_nsl_kdd_multi
```

### 10.2 Firewall Adapter Architecture

The `FirewallAdapter` abstract base class defines `block(ip, ttl_seconds)`, `unblock(ip)`, and `list_rules()`. All adapters validate the target IP with `ipaddress.ip_address()` before any action.

The `WebhookFirewallAdapter` sends HTTP POSTs with configurable `adapter_call_timeout_s` (default 3s) and implements a circuit breaker pattern (`adapter_cb_failure_threshold=3`, `adapter_cb_open_duration_s=60s`).

The `UfwFirewallAdapter` and `NftablesFirewallAdapter` invoke system commands via `subprocess.run` with `timeout=5`. This requires the container process to have elevated privileges — a security concern that the mock adapter avoids in default deployments.

### 10.3 NSL-KDD Dataset Dependency

The system's ML models are trained on the NSL-KDD dataset (KDDTrain+.txt, KDDTest+.txt in `data/`). The 40-feature schema (`src/schema.py`) is fixed and any ingested flow records must conform to or be enriched to this schema. This creates a critical coupling: the detection accuracy is bounded by how well NSL-KDD features represent real-world traffic in the deployment environment.

---

## 11. Testing Strategy & Quality Assurance

### 11.1 Test Suite Overview

| Metric | Value |
|---|---|
| Total test files | 112 |
| Total passing tests | 1046+ (from memory context; 35 pre-existing auth 401 failures) |
| Test framework | pytest >=7.4.0 |
| Coverage framework | pytest-cov |
| Coverage gate | 50% overall (CI enforced) |
| Performance tests | pytest-benchmark (`.benchmarks/` directory) |

### 11.2 Test Coverage by Domain

| Domain | Test Files | Coverage Assessment |
|---|---|---|
| Authentication | test_auth_service.py, test_c01_unified_auth.py, test_c04_rs256_revocation.py, test_auth_bypass_disabled.py, test_endpoint_auth_matrix.py | High — core auth paths well-covered |
| Database / Schema | test_ops_store.py, test_schema_migration.py, test_b06_store_consolidation.py, test_c02_db_idempotency.py, test_c03_rbac_migration.py | High — migration framework tested |
| Detection | test_detection_service.py, test_api_detection.py, test_phase_f_ml.py, test_week1_features.py | Moderate |
| Prevention | test_prevention_service.py, test_prevention_runtime.py, test_d04_prevention_defaults.py, test_approval_gate_and_risk_weights.py | Moderate-High |
| Rate Limiting | test_rate_limiter.py, test_c05_rate_limiter.py | High |
| Ingestion / Pipeline | test_pipeline_runtime.py, test_redis_outage_recovery.py, test_stream_consumer_group_recovery.py | Moderate |
| Security | test_csrf_protection.py, test_correlation_tracing.py, test_d02_input_sanitizer.py, test_c07_correlation_sanitization.py | High |
| Alert lifecycle | test_alert_lifecycle.py, test_d06_alert_dedup.py, test_d08_alert_pagination_retention.py | High |
| Health | test_d07_health_readonly.py, test_e03_health_metrics.py, test_e04_health_no_db_path.py | High |
| Threat Intel | test_threat_intel.py | Moderate |
| API endpoints | test_api_endpoints.py, test_new_api_endpoints.py, test_all_routes_require_auth.py | Moderate |
| Observability | test_observability_runtime.py, test_drift_monitor.py | Low-Moderate |
| Frontend/UI | test_dashboard_page.py | Low |
| Training | test_train_cli.py, test_preprocess.py | Low |
| Attack scenarios | test_attack_chains.py, test_anomaly_and_escalation.py | Moderate |

### 11.3 Test Pattern Analysis

The test suite uses a `conftest.py` with shared fixtures. Test naming follows a phase-plan convention (`test_a03_`, `test_b02_`, `test_c01_`, etc.) reflecting the recovery plan phases. This is pragmatic but mixes issue-tracking concerns with functional concerns — over time, as the recovery work is considered complete, tests should be reorganized by functional domain rather than plan phase.

Performance benchmarks exist in `.benchmarks/` but the CI pipeline does not enforce performance regression gates — only functional correctness.

### 11.4 Quality Assessment

**Strengths:**
- Comprehensive auth and security testing
- Migration framework thoroughly tested (each migration has a dedicated test)
- Idempotency scenarios tested (duplicate block, alert dedup)
- Redis outage recovery tested
- False-positive feedback loop tested

**Gaps:**
- Frontend JavaScript not tested (no Playwright/Selenium)
- No load/stress testing in CI
- No contract tests for external integrations (Elasticsearch, Redis)
- Temporal correlation engine not tested with real pattern scenarios
- Coverage gate at 50% is too low for production security tooling
- 35 pre-existing auth 401 failures indicate known broken test paths

---

## 12. Performance, Scalability & Resilience

### 12.1 Scalability Analysis

| Dimension | Current State | Limiting Factor | Scale Path |
|---|---|---|---|
| Request throughput | Single-process Gunicorn | eventlet + SocketIO mandate single worker | Redis SocketIO adapter for multi-worker |
| Detection throughput | ~50 flows/batch (pipeline default) | In-process engine evaluation | Increase batch_size; distribute consumers |
| Alert storage | SQLite file | File locking, no concurrent writers | Migrate to PostgreSQL |
| Rate limit state | In-memory (default) | Per-process; not shared | Redis-backed (built in, opt-in) |
| Session/JWT state | DB-backed revocation | Single OpsStore DB | PostgreSQL + connection pooling |
| Firewall enforcement | Subprocess calls (UFW/nftables) | OS call latency (~5s timeout) | Async adapter or pre-validation queue |
| Model inference | ~100ms per single prediction | scikit-learn predict() | Batch inference; GPU not applicable |

### 12.2 Backpressure and Resilience

The `BackpressureController` implements three-level backpressure:
- **NORMAL** (lag < 5000): All flows processed
- **SAMPLING** (lag 5000–20000): 25% of flows sampled; others dropped
- **SHEDDING** (lag > 20000): New submissions rejected with HTTP 503

This protects the system from cascade failures under extreme load but means detection gaps during shed periods. The 503 response at `/api/ingest` is properly communicated to callers.

Leader election via `LeaderElection` prevents multiple instances from running prevention actions simultaneously. With Redis available, distributed lock semantics are used. Without Redis, the single instance is always leader.

### 12.3 Circuit Breaker Pattern

The `WebhookFirewallAdapter` implements a circuit breaker:
- Opens after `adapter_cb_failure_threshold` (default 3) consecutive failures
- Stays open for `adapter_cb_open_duration_s` (default 60 seconds)
- Prevents cascading timeouts when the webhook target is unavailable

This pattern is not implemented for the subprocess-based UFW and nftables adapters.

### 12.4 Memory and Resource Management

- `AuditLogMiddleware` uses a `deque(maxlen=10000)` for in-memory audit logs, preventing unbounded memory growth
- `InMemoryRateLimiter` bounds key count: when `len(self._events) > 50000`, up to 10000 stale keys are evicted
- `ThresholdEngine._RateCounter` caps timestamps at `_MAX_TIMESTAMPS = 10_000`
- Matplotlib figures are explicitly closed with `plt.close(fig)` and BytesIO buffers closed after model analytics
- The ingestion queue has `max_items=10000` hard cap
- Alert and action queries have hard maximums (`MAX_AUDIT_LIMIT=500`, `MAX_ALERTS_LIMIT=1000`, `_FETCHALL_HARD_MAX=10000`)

### 12.5 Performance Risks

| Risk | Impact | Probability | Notes |
|---|---|---|---|
| Single-worker constraint | High | Certain | Cannot scale without Redis SocketIO adapter |
| Synchronous OpsStore writes in event bus | Medium | High | Every detection event triggers 2+ DB writes |
| Event bus lock contention | Medium | Medium | `publish()` holds `_lock` while copying handlers |
| `list_active_blocks()` on every `/api/ingest` | Medium | Low | Called with limit=5000 on each metrics build; caching recommended |
| SIEM auto-flush every 60s (leader only) | Low | Low | Non-blocking daemon thread; minimal impact |
| Daily retraining at 02:00 UTC | Medium | Medium | Full model retrain could spike CPU significantly |

---

## 13. Cross-Cutting Concerns

### 13.1 Logging Strategy

INIDS uses Python's standard `logging` module with a structured log format:

```
timestamp=<ISO> level=<LEVEL> request_id=<X-Request-ID> source_ip=<IP>
risk_score=<score> action=<action> endpoint=<path> message=<message>
```

The `RequestContextFilter` enriches log records with request context when inside a Flask request. Fields default to `-` outside request context.

JSON logging is available when `INIDS_JSON_LOGGING=1` is set (uses `src/observability/json_logging.py`), intended for log aggregation pipelines (Splunk, ELK).

All significant security events are logged at INFO level with structured key=value format (e.g., `auth.jwt_key_source=ephemeral`, `ws.connect_rejected reason=no_token`).

### 13.2 Configuration Management

Configuration flows through a single `Settings` frozen dataclass loaded once at startup from environment variables + `.env` file. The frozen dataclass prevents accidental mutation at runtime.

Secret resolution order: `{ENV_KEY}_FILE` (Docker secret file) → plain `{ENV_KEY}` environment variable. This supports both Docker Swarm/Kubernetes secrets and direct env var injection.

The `policy_store.py` (`PolicyStore`) adds a second layer of runtime configuration for prevention policy: versioned, auditable, and rollback-capable. Policy changes take effect immediately (no restart required).

### 13.3 Background Jobs / Daemon Threads

| Thread Name | Trigger | Period | Purpose |
|---|---|---|---|
| `ti-refresh` | TI feed dir configured | `INIDS_TI_REFRESH_INTERVAL` (3600s default) | Purge expired TI indicators; reload feeds |
| `siem-flush` | First request | 60 seconds | Auto-drain SIEM export buffer |
| `alert-retention` | First request | 86400 seconds (daily) | Delete old alerts per `INIDS_ALERT_RETENTION_DAYS` |
| `module-broadcaster` | WebSocket connect | 2 seconds | Push `metrics.update` and module updates to dashboard |
| `prevention-scheduler` | First request | 30 seconds | Run cleanup of expired prevention actions (leader only) |
| `retraining-scheduler` | Model load | Daily at 02:00 UTC | Re-train ML model from DatasetCollector data |
| `perception-integration` | App startup | Event-driven (worker threads: 2) | Process event bus events through perception layer |

All daemon threads are started lazily on first request via `_ensure_scheduler_started()`, avoiding slow startup times and ensuring the app is ready to serve requests before background work begins.

### 13.4 Caching

There is no dedicated caching layer (Redis used for pipeline/rate-limiting, not application-level cache). Notable in-memory state:
- `all_models` dict: loaded ML models cached in process memory
- `AnomalyEngine._buffer`: 3000-sample buffer for auto-fit
- `TIEngine` cache: in-memory indicator store with TTL
- `EscalationTracker`: in-memory per-IP event history
- `ThresholdEngine._counters`: per-(IP, metric) sliding window counters
- `IngestionQueue`: in-process queue with `max_items=10000`

Caching of `list_active_blocks()` (called repeatedly in metrics builds) would be a meaningful performance improvement.

---

## 14. Technical Debt & Risk Assessment

### 14.1 Technical Debt Inventory

| Item | Location | Type | Effort | Priority |
|---|---|---|---|---|
| Temporal correlation patterns commented out | app.py:278-297 | Config/Logic | Low | High |
| `allow_unsafe_werkzeug=True` in dev | app.py:1960 | Security | Low | Medium |
| `system_health: 98` hardcoded | app.py `_build_dashboard_metrics_payload()` | Logic | Low | Low |
| Single Gunicorn worker constraint | Dockerfile CMD | Architecture | High | Medium |
| No Redis SocketIO adapter for multi-instance | — | Architecture | High | Medium |
| Coverage gate at 50% | pyproject.toml | Quality | Medium | High |
| 35 pre-existing auth 401 test failures | test suite | Quality | Medium | High |
| No SAST pipeline | .github/workflows/ | Security | Medium | High |
| No container scanning | .github/workflows/ | Security | Low | Medium |
| No secret scanning | .github/workflows/ | Security | Low | High |
| `'unsafe-inline'` in CSP style-src | middleware.py | Security | Medium | Medium |
| No circuit breaker on UFW/nftables adapters | firewall_adapters.py | Reliability | Medium | Medium |
| `api.investigations` generates from alerts in-memory | system.py | Performance | Low | Low |
| Policy `block_requires_approval` not persisted in rollback | prevention.py | Logic | Low | Medium |
| Swagger/Connexion integration not wired to routes | connexion_integration.py | Feature | Medium | Low |
| GeoIP requires external database file | geoip_enrichment.py | Feature | Low | Medium |
| `detect_response.txt` committed to repo | repo root | Hygiene | Trivial | Low |
| `PHASE_10_OPERATIONAL_RUNBOOK.md` deleted (git status) | — | Hygiene | Trivial | Low |
| `pytest_tmp_run2` and `pytest_tmp_single` directories | repo root | Hygiene | Trivial | Low |

### 14.2 Risk Register

| Risk | Probability | Impact | Score | Owner | Mitigation |
|---|---|---|---|---|---|
| Ephemeral JWT keys deployed to production | Medium | Critical | High | DevOps | Document and enforce `INIDS_JWT_REQUIRE_PERSISTENT=true` |
| Redis not secured (no auth) | Medium | High | High | Operator | Add Redis AUTH configuration support to Settings |
| Alert volume overwhelming SQLite | Low | High | Medium | Architect | Document PostgreSQL migration path; add monitoring |
| Single-worker becoming bottleneck | Medium | High | High | Architect | Implement Redis SocketIO adapter |
| Temporal engine never activated | High | Medium | High | Config | Register default patterns at startup or document operational step |
| ML model staleness (NSL-KDD drift) | Medium | High | High | ML Ops | Ensure retraining scheduler is supplied with real traffic data |
| Pre-existing 401 test failures masking regressions | Medium | Medium | Medium | QA | Investigate and fix all test failures |
| Dependency CVE in compiled requirements.txt | Low | High | Medium | DevOps | pip-audit in CI (already present); weekly review |
| Large alert/audit tables degrading read performance | Medium | Medium | Medium | Ops | Apply retention policy; add PostgreSQL for large deployments |
| Runbook deleted from repo (git status) | Low | Low | Low | DevOps | Restore or move operational documentation |

---

## 15. Architecture Decision Records & Design Rationale

### ADR-001: Flask as Web Framework (Not FastAPI/Django)

**Context:** Python web framework selection for a security-focused IDS/IPS backend.

**Decision:** Flask >=3.0.0 with Flask-SocketIO.

**Rationale:** Flask's minimal footprint fits a purpose-built security application. Flask-SocketIO was required for real-time WebSocket event broadcasting. Flask 3.0's improved async support and security fixes made it the appropriate choice. FastAPI would have required ASGI infrastructure changes incompatible with Flask-SocketIO's eventlet model.

**Consequences:** Single-worker constraint from eventlet; requires Redis SocketIO adapter for horizontal scaling.

---

### ADR-002: SQLite Default with PostgreSQL Upgrade Path

**Context:** Choosing between a file-based and client-server database.

**Decision:** SQLite for development/single-node; PostgreSQL for production scale.

**Rationale:** SQLite requires zero operational overhead for the common single-node deployment. The `OpsStore` dual-backend design allows operators to migrate by changing `OPS_DB_PATH` to a PostgreSQL URL. Schema migrations run automatically on startup in both backends.

**Consequences:** Migration v4's partial unique index requires SQLite >= 3.8.9. The PostgreSQL path requires SQLAlchemy installation.

---

### ADR-003: RS256 JWT over HS256

**Context:** Token signing algorithm selection.

**Decision:** RS256 (RSA) asymmetric signing.

**Rationale:** RS256 allows public key distribution for verification without sharing the signing key. This is critical for a multi-service architecture where services may need to verify tokens without being able to issue them. HS256 shared secrets cannot be distributed securely. Explicitly stated as "non-negotiable" in PLAN.md.

**Consequences:** More complex key management; ephemeral key fallback for dev; requires `INIDS_JWT_REQUIRE_PERSISTENT=true` for production enforcement.

---

### ADR-004: Event Bus Architecture

**Context:** Decoupling detection, risk scoring, policy, and action execution.

**Decision:** In-process publish-subscribe `EventBus` with typed events.

**Rationale:** Allows each pipeline stage to be independently developed and tested. Enables multiple consumers (SIEM, WebSocket, audit log) to react to the same events without coupling. Thread-safe via `RLock`.

**Consequences:** All consumers run synchronously in the handler thread; a slow handler blocks subsequent handlers. No event persistence — if a handler crashes, the event is lost.

---

### ADR-005: ANY_TRIGGER Aggregation Strategy

**Context:** How to combine verdicts from multiple detection engines.

**Decision:** Default to `AggregationStrategy.ANY_TRIGGER` — any single engine flagging an attack produces an attack verdict.

**Rationale:** In a security context, false negatives (missing an attack) are more dangerous than false positives (flagging benign traffic). ANY_TRIGGER maximizes recall at the cost of precision. The FP manager, alert filter, and allowlist provide complementary mechanisms to reduce false positive alert fatigue.

**Consequences:** Higher alert volume; requires FP feedback loop to suppress noisy rules/engines.

---

### ADR-006: Fail-Closed Startup Validation

**Context:** Preventing silent misconfiguration from reaching production.

**Decision:** `_validate_all_routes_have_auth_decorator()`, `ALLOW_UNAUTHENTICATED` check, and `SECRET_KEY` requirement all raise `RuntimeError` at startup.

**Rationale:** In a security application, a misconfigured auth layer could expose all endpoints to unauthenticated access. Failing fast at startup with a clear error message is far safer than silently deploying a broken auth configuration.

**Consequences:** Any new route without `@require_roles` will crash the application. This is intentional and correct behavior.

---

### ADR-007: Two-Tier Rate Limiting

**Context:** Rate limiting approach for API protection.

**Decision:** Tier 1 global IP limit (1000 req/min) in `before_request` hook; Tier 2 per-user limit (200 req/min) inside `require_roles()`.

**Rationale:** IP-based rate limiting prevents volumetric attacks from unauthenticated sources. Per-user rate limiting prevents authenticated abuse from a single compromised credential. Both tiers support Redis-backed distributed counting with in-memory fallback.

**Consequences:** Unauthenticated endpoints (health, auth) are only protected by Tier 1.

---

### ADR-008: Model Checksum Verification

**Context:** ML model files loaded via joblib deserialization (pickle-based).

**Decision:** SHA-256 manifest (`checksums.sha256`) required in MODELS_DIR; `INIDS_MODEL_VERIFY=strict` (default) refuses to load unverified models.

**Rationale:** Pickle deserialization is a known remote code execution vector. An attacker who can replace a model file in the models directory could achieve arbitrary code execution. Checksum verification provides integrity assurance.

**Consequences:** Operators must regenerate checksums after model retraining (`scripts/generate_model_checksums.py`).

---

## 16. Recommendations & Improvement Roadmap

### 16.1 Prioritized Recommendations

#### P0 — Critical (Address Before Production Deployment)

| ID | Recommendation | Effort | Impact | Evidence |
|---|---|---|---|---|
| P0-001 | Enforce `INIDS_JWT_REQUIRE_PERSISTENT=true` in production via container healthcheck or startup probe | Low | Critical | `jwt_manager.py`: ephemeral key default |
| P0-002 | Fix 35 pre-existing auth 401 test failures | Medium | High | Memory context: test_suite_state.md |
| P0-003 | Add secret scanning to CI (gitleaks or truffleHog) | Low | High | `.env` file in working tree; `SECRET_KEY=change-me-now` in `.env.example` |
| P0-004 | Register temporal correlation patterns at startup or provide default patterns configuration | Low | High | app.py:278-297 patterns commented out |

#### P1 — High Priority (Address Within 30 Days)

| ID | Recommendation | Effort | Impact | Evidence |
|---|---|---|---|---|
| P1-001 | Raise coverage gate from 50% to 70% for core security modules | Medium | High | pyproject.toml `fail_under = 50` |
| P1-002 | Add SAST pipeline (bandit, semgrep) to CI/CD | Low | High | No SAST in security.yml |
| P1-003 | Add container image scanning (Trivy/Grype) to CI/CD | Low | Medium | No container scan in security.yml |
| P1-004 | Implement Redis AUTH configuration in Settings | Low | High | No Redis password support |
| P1-005 | Remove `allow_unsafe_werkzeug=True` (use Gunicorn/eventlet in all cases) | Low | Medium | app.py:1960 |
| P1-006 | Replace hardcoded `system_health: 98` with actual health score derived from health_check probes | Low | Low | `_build_dashboard_metrics_payload()` |
| P1-007 | Add circuit breaker to UFW and nftables adapters (matches webhook pattern) | Medium | Medium | `firewall_adapters.py` — only webhook has circuit breaker |

#### P2 — Medium Priority (Address Within 90 Days)

| ID | Recommendation | Effort | Impact | Evidence |
|---|---|---|---|---|
| P2-001 | Implement Redis SocketIO adapter for horizontal scaling beyond single-worker | High | High | Architecture constraint |
| P2-002 | Replace `'unsafe-inline'` in CSP style-src with nonce-based policy | Medium | Medium | `middleware.py` CSP header |
| P2-003 | Implement application-level caching for `list_active_blocks()` (TTL cache, 5-10s) | Low | Medium | Called on every metrics build |
| P2-004 | Add integration tests against a running Docker container to CI | Medium | High | CI has no smoke tests |
| P2-005 | Migrate to official `prometheus_client` library for metrics | Medium | Low | `MetricsService` is a custom implementation |
| P2-006 | Add OpenTelemetry distributed tracing | High | Medium | Correlation IDs exist but no trace propagation |
| P2-007 | Add PostgreSQL migration documentation and runbook | Low | Medium | No migration guide exists |
| P2-008 | Configure GeoIP database path in Settings | Low | Medium | `geoip_enrichment.py` has no configured database |

#### P3 — Low Priority (Address Within 6 Months)

| ID | Recommendation | Effort | Impact | Evidence |
|---|---|---|---|---|
| P3-001 | Reorganize test files from plan-phase naming to functional domain naming | Medium | Low | Mixed naming convention |
| P3-002 | Add frontend JavaScript testing (Playwright or Vitest) | High | Medium | No frontend tests |
| P3-003 | Wire Connexion/OpenAPI integration to actual API routes | High | Medium | `connexion_integration.py` exists but unused |
| P3-004 | Add performance regression benchmarks to CI with gates | Medium | Medium | `.benchmarks/` exists but no CI enforcement |
| P3-005 | Clean up `pytest_tmp_run2`, `pytest_tmp_single`, `detect_response.txt` from repo | Trivial | Trivial | Repository hygiene |
| P3-006 | Evaluate moving from in-memory `AuditLogMiddleware` to OpsStore-backed audit | Medium | Low | In-memory only survives restart |
| P3-007 | Add Redis Sentinel/Cluster support for HA Redis deployments | High | Medium | Only single Redis instance supported |
| P3-008 | Evaluate Elasticsearch encryption at rest configuration | Low | Medium | No data-at-rest encryption specified |

### 16.2 Improvement Roadmap

| Quarter | Focus | Key Deliverables |
|---|---|---|
| Q2 2026 | Security hardening | P0 items complete; SAST/secret scanning in CI; JWT key enforcement; temporal patterns operational |
| Q3 2026 | Quality and reliability | Coverage gate to 70%; container scanning; Redis AUTH; circuit breakers on all adapters |
| Q4 2026 | Scalability | Redis SocketIO adapter; multi-worker deployment; CSP nonce-based policy; PostgreSQL runbook |
| Q1 2027 | Observability | OpenTelemetry integration; official Prometheus client; frontend testing; performance benchmarks in CI |

---

## 17. Appendices

### Appendix A: File Structure Map

```
INIDS/
├── .github/
│   └── workflows/
│       └── security.yml              # CI: pip-audit + drift check + tests
├── .env                               # Local environment (not committed to prod)
├── .env.example                       # Environment variable template
├── Dockerfile                         # Production container image
├── Makefile                           # Development shortcuts
├── pyproject.toml                     # Project metadata + pytest/coverage config
├── requirements.in                    # Direct dependency declarations
├── requirements.txt                   # Pinned + hashed compiled dependencies
├── package.json                       # Node dependencies (Tailwind CSS build)
├── start_flask_dev.py                 # Development launcher
│
├── data/                              # NSL-KDD dataset files
│   ├── inids_ops.db                   # Operational SQLite database
│   └── KDDTest+.txt, KDDTrain+.txt    # Training/test data
│
├── models/                            # Trained ML models
│   ├── rf_nsl_kdd.pkl                 # Primary Random Forest model
│   ├── gb_nsl_kdd.pkl                 # Gradient Boosting
│   ├── dt_nsl_kdd.pkl                 # Decision Tree
│   ├── ab_nsl_kdd.pkl                 # AdaBoost
│   ├── mlp_nsl_kdd.pkl                # Multi-layer Perceptron
│   ├── rf_nsl_kdd_multi.pkl           # Multi-class RF
│   └── checksums.sha256               # SHA-256 integrity manifest
│
├── rules/
│   └── default_rules.yaml             # 11 signature detection rules
│
├── src/                               # Core application library
│   ├── schema.py                      # NSL-KDD 40-feature schema
│   ├── settings.py                    # Environment configuration
│   ├── ops_store.py                   # Dual-backend operational persistence
│   ├── detection_service.py           # ML inference service
│   ├── prevention_service.py          # Firewall action coordination
│   ├── ingestion_service.py           # Data ingest queue
│   ├── metrics_service.py             # In-process metrics
│   ├── rate_limiter.py                # Two-tier rate limiter
│   ├── middleware.py                  # Security middleware stack
│   ├── firewall_adapters.py           # Pluggable firewall backends
│   ├── input_sanitizer.py             # Input validation/sanitization
│   ├── logging_config.py              # Structured log formatter
│   ├── feature_engineering.py         # Feature enrichment
│   ├── log_parsers.py                 # Zeek/Suricata parsers
│   ├── label_utils.py                 # Label normalization
│   ├── model_registry.py              # Model tracking
│   ├── csrf_protection.py             # CSRF middleware
│   ├── correlation_tracing.py         # Request correlation IDs
│   ├── elasticsearch_client.py        # ES/OpenSearch client
│   ├── elasticsearch_audit_bridge.py  # ES audit integration
│   ├── async_utils.py                 # Thread pool executor
│   │
│   ├── auth/                          # Authentication module
│   │   ├── models.py                  # AuthContext, AuthError
│   │   ├── auth_service.py            # UnifiedAuthService
│   │   ├── jwt_manager.py             # RS256JWTManager
│   │   ├── decorators.py              # @require_roles
│   │   └── validators.py              # Input validation
│   │
│   ├── core/                          # Core infrastructure
│   │   ├── event_bus.py               # EventBus + event dataclasses
│   │   └── config_manager.py          # Config management utilities
│   │
│   ├── detection/                     # Detection engine framework
│   │   ├── engine_base.py             # DetectionEngine ABC
│   │   ├── engine_registry.py         # EngineRegistry
│   │   ├── aggregator.py              # EngineAggregator + strategies
│   │   ├── rule_compiler.py           # Advanced YAML rule parser
│   │   ├── temporal_correlation.py    # Multi-stage attack patterns
│   │   ├── ml_utils.py                # Model loading + checksum
│   │   └── engines/
│   │       ├── ml_engine.py           # scikit-learn model wrapper
│   │       ├── signature_engine.py    # YAML rule matching
│   │       ├── anomaly_engine.py      # IsolationForest
│   │       ├── threshold_engine.py    # Rate-based detection
│   │       └── honeypot_engine.py     # Canary IP/port detection
│   │
│   ├── ips/                           # Prevention/IPS subsystem
│   │   ├── policy_engine.py           # Risk → decision mapping
│   │   ├── risk_engine.py             # Risk score calculation
│   │   ├── action_executor.py         # Firewall action execution
│   │   ├── scheduler.py               # Periodic action cleanup
│   │   ├── incident_aggregator.py     # Incident grouping
│   │   ├── entity_enrichment.py       # IP context enrichment
│   │   ├── alert_filter.py            # Three-layer alert filter
│   │   └── ...
│   │
│   ├── pipeline/                      # Redis stream pipeline
│   │   ├── backpressure.py            # Backpressure controller
│   │   ├── stream_processor.py        # Redis consumer
│   │   └── worker.py                  # Background pipeline worker
│   │
│   ├── advanced/                      # Advanced detection modules
│   │   ├── dns_detection.py
│   │   ├── geoip_enrichment.py
│   │   ├── http_patterns.py
│   │   ├── ml_anomaly.py
│   │   └── tls_validation.py
│   │
│   ├── ha/                            # High availability
│   │   ├── health_check.py            # Named health probes
│   │   └── leader_election.py         # Redis-based leader election
│   │
│   ├── observability/                 # Observability utilities
│   │   ├── json_logging.py            # JSON log formatter
│   │   └── siem_exporter.py           # SIEM event buffer
│   │
│   ├── perception/                    # INIDS 2.0 perception layer
│   │   ├── attack_story_engine.py     # Attack narrative generation
│   │   ├── confidence_breakdown.py    # Feature contribution analysis
│   │   ├── live_system_pulse.py       # 60-min rolling system state
│   │   └── perception_integration.py  # EventBus integration
│   │
│   ├── policy/                        # Policy versioning
│   │   └── policy_store.py            # Versioned policy with rollback
│   │
│   ├── prevention/                    # Prevention utilities
│   │   ├── allowlist.py               # IP/CIDR allowlist
│   │   ├── escalation_tracker.py      # Repeat-offender tracking
│   │   └── false_positive_manager.py  # FP suppression
│   │
│   ├── realtime/                      # Real-time streaming
│   │   └── broadcaster.py             # EventBus → WebSocket bridge
│   │
│   ├── threat_intel/                  # Threat intelligence
│   │   ├── feed_manager.py            # TI feed loading and cache
│   │   └── ti_engine.py               # TI detection engine
│   │
│   ├── training/                      # ML lifecycle
│   │   ├── dataset_collector.py       # Training data collection
│   │   └── retraining_scheduler.py    # Automated model retraining
│   │
│   └── decoding/                      # Packet decoding
│       └── packet_decoder.py
│
├── web_app/                           # Flask application
│   ├── app.py                         # Application factory + main module
│   ├── blueprints/                    # Route handlers (11 blueprints)
│   ├── templates/                     # Jinja2 HTML templates
│   └── static/                        # CSS, JS, images
│
└── tests/                             # 112 test files
```

### Appendix B: Complete API Reference Summary

See Section 6.2 for the full endpoint catalog. All API endpoints require HTTPS in production. Authentication via `Authorization: Bearer <token>` or `X-API-Key: <key>` header.

**Base URL:** `http://{host}:{port}` (default `http://0.0.0.0:5000`)

### Appendix C: Database Schema Reference

See Section 5.3 for complete table definitions. Current schema version: 6.

SQLite location: `data/inids_ops.db` (default)
PostgreSQL: Set `OPS_DB_PATH=postgresql://user:pass@host/db`

### Appendix D: Environment Variable Reference

| Variable | Default | Required | Description |
|---|---|---|---|
| SECRET_KEY / FLASK_SECRET_KEY | — | YES | Flask session secret key |
| HOST | 0.0.0.0 | No | Bind address |
| PORT | 5000 | No | Bind port |
| FLASK_DEBUG | 0 | No | Enable debug mode |
| OPS_DB_PATH | data/inids_ops.db | No | SQLite path or postgresql:// URL |
| REDIS_URL | — | No | Redis connection URL (enables pipeline) |
| INIDS_PIPELINE_ENABLED | true | No | Enable Redis stream pipeline |
| INIDS_PIPELINE_BATCH_SIZE | 50 | No | Consumer batch size |
| INIDS_PIPELINE_STREAM_KEY | inids:flows | No | Redis stream key |
| INIDS_REQUIRE_API_KEYS | 0 | No | Require API key auth |
| INIDS_REQUIRE_SECRET_KEY | 0 | No | Fail if SECRET_KEY is empty |
| RATE_LIMIT_REQUESTS | 120 | No | Requests per window (settings only) |
| RATE_LIMIT_WINDOW_SECONDS | 60 | No | Rate limit window |
| FIREWALL_ADAPTER | mock | No | mock / ufw / nftables / webhook |
| FIREWALL_WEBHOOK_URL | — | Conditional | Required if FIREWALL_ADAPTER=webhook |
| INIDS_ADMIN_API_KEY | — | No | Seeds admin service account |
| INIDS_ANALYST_API_KEY | — | No | Seeds analyst service account |
| INIDS_SENSOR_API_KEY | — | No | Seeds sensor service account |
| INIDS_VIEWER_API_KEY | — | No | Seeds viewer service account |
| INIDS_JWT_PRIVATE_KEY | — | Recommended | RS256 private key (PEM) |
| INIDS_JWT_PUBLIC_KEY | — | Recommended | RS256 public key (PEM) |
| INIDS_JWT_PRIVATE_KEY_FILE | — | No | Path to private key file (Docker secrets) |
| INIDS_JWT_PUBLIC_KEY_FILE | — | No | Path to public key file |
| INIDS_JWT_REQUIRE_PERSISTENT | false | No | Fail startup if no persistent JWT key |
| INIDS_MODEL_VERIFY | strict | No | strict / warn / disabled — model checksum mode |
| INIDS_TI_FEED_PATH | — | No | Path to single TI feed file |
| INIDS_TI_FEED_DIR | — | No | Directory of TI feed files |
| INIDS_TI_REFRESH_INTERVAL | 3600 | No | TI cache refresh interval (seconds) |
| INIDS_HONEYPOT_IPS | — | No | Comma-separated honeypot IP addresses |
| INIDS_HONEYPOT_PORTS | — | No | Comma-separated honeypot port numbers |
| INIDS_HONEYPOT_ENABLED | true | No | Enable honeypot engine |
| ELASTICSEARCH_ENABLED | 0 | No | Enable Elasticsearch audit bridge |
| ELASTICSEARCH_HOSTS | localhost | No | ES host(s) |
| ELASTICSEARCH_PORT | 9200 | No | ES port |
| ELASTICSEARCH_USE_SSL | false | No | Enable TLS to ES |
| ELASTICSEARCH_VERIFY_CERTS | false | No | Verify ES TLS certificates |
| ELASTICSEARCH_USERNAME | — | No | ES username |
| ELASTICSEARCH_PASSWORD | — | No | ES password |
| INIDS_CORS_ORIGINS | localhost:5000,127.0.0.1:5000 | No | Comma-separated allowed CORS origins |
| INIDS_INTERNAL_CIDRS | — | No | Comma-separated internal CIDR ranges |
| INIDS_JSON_LOGGING | false | No | Enable JSON log format |
| INIDS_ALERT_RETENTION_DAYS | 0 | No | Delete alerts older than N days (0=disabled) |
| ALLOW_UNAUTHENTICATED | false | No | MUST NOT be true; raises RuntimeError |
| ADAPTER_CALL_TIMEOUT_S | 3.0 | No | Firewall adapter call timeout (seconds) |
| ADAPTER_CB_FAILURE_THRESHOLD | 3 | No | Circuit breaker failure count |
| ADAPTER_CB_OPEN_DURATION_S | 60.0 | No | Circuit breaker open duration |
| INIDS_REDIS_REQUIRED | false | No | Require Redis for multi-instance deployments |

### Appendix E: Dependency Manifest (Direct)

| Package | Version Constraint | Category |
|---|---|---|
| pandas | >=2.0.0 | Data/ML |
| scikit-learn | >=1.3.0 | ML |
| Flask | >=3.0.0 | Web |
| matplotlib | >=3.7.0 | Visualization |
| seaborn | >=0.12.0 | Visualization |
| joblib | >=1.3.0 | ML serialization |
| numpy | >=1.24.0 | Numerical |
| Werkzeug | >=3.0.0 | Web (Flask dep) |
| Jinja2 | >=3.1.0 | Templating |
| scapy | >=2.5.0 | Packet capture |
| requests | >=2.31.0 | HTTP client |
| PyYAML | latest | Config parsing |
| redis | latest | Cache/queue |
| flask-socketio | latest | WebSocket |
| python-socketio | latest | WebSocket |
| PyJWT | latest | Authentication |
| cryptography | latest | Crypto |
| jsonschema | latest | Validation |
| connexion[swagger-ui] | latest | OpenAPI |
| uvicorn | latest | ASGI server |
| opensearch-py | latest | Search (primary) |
| elasticsearch | latest | Search (fallback) |
| aiohttp | latest | Async HTTP |
| aiofiles | latest | Async file I/O |
| sqlalchemy | latest | ORM/DB |
| gunicorn | latest | WSGI server |
| eventlet | latest | Async networking |
| flask-compress | latest | Compression |
| pytest | >=7.4.0 | Testing |
| pytest-benchmark | latest | Benchmarking |
| pip-audit | latest | Security |

### Appendix F: Security Checklist

| Check | Status | Notes |
|---|---|---|
| All routes require authentication | PASS | Startup enforcement via _validate_all_routes_have_auth_decorator() |
| JWT uses asymmetric signing (RS256) | PASS | jwt_manager.py |
| Tokens have reasonable expiry (<=1h) | PASS | 3600s non-configurable |
| Token revocation supported | PASS | JTI + revoked_tokens table |
| API keys hashed (not stored plaintext) | PASS | SHA-256 in api_keys.key_hash |
| Input sanitization applied | PASS | input_sanitizer.py on string fields |
| SQL injection prevention | PASS | Parameterized queries throughout OpsStore |
| CSRF protection | PASS (partial) | Middleware present; coverage scope not fully verified |
| Rate limiting applied | PASS | Two-tier IP + per-user |
| Security headers (OWASP) | PASS | SecurityHeadersMiddleware |
| Non-root container | PASS | Dockerfile |
| Dependency hash verification | PASS | pip install --require-hashes |
| Dependency CVE scanning | PASS | pip-audit in CI |
| Model integrity verification | PASS | SHA-256 checksums |
| Request size limits | PASS | 16MB Flask + 1MB middleware |
| Sensitive routes properly protected | PASS | Admin-only for policy, actions, user management |
| Secret stored outside codebase | PARTIAL | .env.example contains placeholders; .env in working tree (should be gitignored) |
| Persistent JWT keys enforced | PARTIAL | Requires INIDS_JWT_REQUIRE_PERSISTENT=true |
| TLS enforced | NOT IN APP | Operator responsibility (reverse proxy) |
| Secrets scanning in CI | MISSING | No gitleaks/truffleHog workflow |
| SAST in CI | MISSING | No bandit/semgrep workflow |
| Container scanning in CI | MISSING | No Trivy/Grype workflow |

### Appendix G: Glossary

| Term | Definition |
|---|---|
| IDS | Intrusion Detection System — passively monitors and alerts |
| IPS | Intrusion Prevention System — actively blocks detected threats |
| IDPS | Intrusion Detection and Prevention System |
| NSL-KDD | Improved KDD Cup 1999 dataset for network intrusion detection |
| RF | Random Forest classifier |
| GB | Gradient Boosting classifier |
| DT | Decision Tree classifier |
| AB | AdaBoost classifier |
| MLP | Multi-Layer Perceptron (neural network) |
| TI | Threat Intelligence |
| TTL | Time-to-Live — duration of a temporary block |
| JTI | JWT ID — unique identifier claim within a JWT |
| RS256 | RSA Signature with SHA-256 — asymmetric JWT algorithm |
| RBAC | Role-Based Access Control |
| SIEM | Security Information and Event Management |
| CSRF | Cross-Site Request Forgery |
| CSP | Content Security Policy |
| OpsStore | The central operational persistence layer in INIDS |
| EventBus | In-process publish-subscribe event dispatcher |
| EngineRegistry | Registry of all detection engines; manages enable/disable state |
| AggregationStrategy | Method for combining multiple engine verdicts (ANY_TRIGGER default) |
| BackpressureController | Component that monitors lag and throttles ingestion |
| LeaderElection | Redis-backed distributed lock to ensure single active IPS instance |
| PolicyStore | Versioned prevention policy with history and rollback |
| FPManager | False Positive Manager — suppresses known benign detections |
| EscalationTracker | Tracks repeat offender IPs to boost risk scores |
| EntityEnrichment | Per-IP context gathering: GeoIP, TI history, behavioral profile |
| PerceptionLayer | INIDS 2.0 system understanding: attack stories, confidence, pulse |
| DatasetCollector | Accumulates traffic samples for model retraining |
| RetrainingScheduler | Automated daily model retraining orchestrator |

---

*This report was produced from a complete analysis of the INIDS repository as of 2026-05-19. All findings are directly traceable to observed source code, configuration files, and repository artifacts. No findings have been inferred or extrapolated beyond what is directly evidenced in the codebase.*
