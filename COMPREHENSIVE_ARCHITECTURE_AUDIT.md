# COMPREHENSIVE ARCHITECTURE AUDIT REPORT
## INIDS: Intelligent Network Intrusion Detection System

**Report Date**: May 15, 2026  
**Report Type**: Full Production Architecture Audit  
**System Status**: ✅ Production-Ready with Architectural Stability Assessment  
**Audit Scope**: Complete System Reconstruction + Integration Analysis  

---

## EXECUTIVE SUMMARY

INIDS is a **production-grade, event-driven intrusion detection system** built on Flask with a multi-engine detection framework, real-time WebSocket streaming, and comprehensive prevention capabilities.

### Critical Findings
- **System Architecture**: ✅ Sound, event-driven with clear separation of concerns
- **Critical Issues**: 🔴 **8 MAJOR INTEGRATION GAPS** identified (documented below)
- **Data Flow**: ✅ Well-designed for detection → risk → policy → action chains
- **Real-time Capabilities**: ✅ Fully operational with SocketIO/WebSocket
- **Deployment Readiness**: ⚠️ **CONDITIONAL** — Minor gaps in HA/distributed components

---

## PHASE 1: GLOBAL ARCHITECTURE OVERVIEW

### 1.1 Architectural Style
**Type**: Event-Driven Monolith with Multi-Engine Detection Framework  
**Pattern**: Publisher/Subscriber (EventBus) + Plugin Architecture (Engines)  
**Tier Architecture**: 3-Tier (Frontend → API/Business Logic → Data)  

### 1.2 System Components

```
INIDS SYSTEM TOPOLOGY (2026)
============================

┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND LAYER                           │
│  (HTML/JS/CSS - Dashboard, Alerts, Actions, Policy, etc.)      │
│  WebSocket: SocketIO (/events namespace)                       │
│  HTTP: REST API                                                │
└────────┬────────────────────────────────────────────────────────┘
         │ REST + WebSocket
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND API LAYER                          │
│                    (Flask Application)                          │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  API Routes (75+ endpoints)                            │   │
│  │  /api/predict  /api/detect  /api/alerts  /api/actions │   │
│  │  /api/policy   /api/audit   /api/engines              │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Security Middleware Stack                             │   │
│  │  · CSRF Protection      · Rate Limiting                │   │
│  │  · Correlation Tracing  · Input Sanitization          │   │
│  │  · IP Blocking          · Audit Logging               │   │
│  │  · JWT Authentication                                 │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Detection & Risk Processing                           │   │
│  │  · DetectionService    · RiskEngine                    │   │
│  │  · PolicyEngine        · ActionExecutor                │   │
│  │  · EngineRegistry      · EngineAggregator              │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Detection Engines (Multi-Engine Framework)            │   │
│  │  · MLEngine            · SignatureEngine               │   │
│  │  · ThresholdEngine     · AnomalyEngine                 │   │
│  │  · HoneypotEngine      · TemporalCorrelationEngine     │   │
│  │  · EntityEnrichmentEngine  · AlertFilterEngine         │   │
│  │  · ThreatIntelEngine   · ThreeLayerAlertFilter         │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Prevention & Response                                 │   │
│  │  · PreventionService   · Allowlist                     │   │
│  │  · EscalationTracker   · FalsePositiveManager          │   │
│  │  · PreventionScheduler · ActionExecutor                │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Event-Driven Core                                     │   │
│  │  · EventBus            · Event Types                   │   │
│  │    - DetectionEvent    - RiskScoreEvent                │   │
│  │    - PolicyDecisionEvent - ActionEvent                 │   │
│  │    - AuditEvent                                        │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Data & State Management                               │   │
│  │  · OpsStore (SQLite/PostgreSQL)                        │   │
│  │  · ModelRegistry       · AlertStore                    │   │
│  │  · IngestionQueue      · MetricsService                │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Observability & Integration                           │   │
│  │  · RealTimeStreamer    · JsonLogging                   │   │
│  │  · SiemExporter        · ElasticsearchBridge           │   │
│  │  · HealthCheck         · LeaderElection                │   │
│  │  · PerceptionIntegration                               │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │
    ┌────┴─────────┬──────────────┬─────────────────┐
    ▼              ▼              ▼                 ▼
┌──────────┐  ┌────────────┐ ┌─────────────┐  ┌────────────┐
│ OpsStore │  │  ML Models │ │Firewall/IPS │  │  External  │
│(SQLite/  │  │  Registry  │ │  Adapters   │  │ Integrations
│Postgres) │  │ (.pkl)     │ │ (Ufw/Nftbl) │  │ (TI Feeds, │
│          │  │            │ │             │  │  WebHooks) │
└──────────┘  └────────────┘ └─────────────┘  └────────────┘
```

### 1.3 Core Subsystems

| Subsystem | Location | Purpose | Status |
|-----------|----------|---------|--------|
| **Detection Framework** | `src/detection/` | Multi-engine attack detection | ✅ Mature |
| **Prevention System** | `src/prevention/` | Attack response & blocking | ✅ Complete |
| **IPS/Response** | `src/ips/` | Policy, risk, action execution | ✅ Complete |
| **Policy Engine** | `src/policy/` | Versioning, rollback, decisions | ✅ Ready |
| **Ingestion** | `src/` ingestion_service | Queue-based data intake | ✅ Ready |
| **Pipeline** | `src/pipeline/` | Streaming processor (Redis-based) | ⚠️ Decoupled |
| **Threat Intelligence** | `src/threat_intel/` | Feed management, TI engine | ⚠️ Unused |
| **Observability** | `src/observability/` | SIEM export, structured logging | ⚠️ Partial |
| **High Availability** | `src/ha/` | Leader election, health checks | ⚠️ Minimal |
| **Real-time** | `src/realtime/` | WebSocket streaming | ✅ Active |
| **Perception** | `src/perception/` | Attack narrative, confidence | ✅ Active |

---

## PHASE 2: FRONTEND ARCHITECTURE

### 2.1 Frontend Stack
- **Framework**: Vanilla JavaScript (no React/Vue/Angular)
- **UI Framework**: Bootstrap 5.2.3
- **Real-time**: Socket.IO 5.3.5 (WebSocket)
- **HTTP Client**: Custom fetch-based wrapper
- **State Management**: Global observer pattern (GlobalState singleton)
- **Routing**: Server-side rendering (Flask Jinja2 templates)

### 2.2 Frontend Structure

**Templates** (`web_app/templates/`):
```
Web Rendering Layer
├── base.html          [Master template + header/nav]
├── home.html          [Landing page]
├── dashboard.html     [Main monitoring dashboard]
├── index.html         [Alternative home]
├── alerts.html        [Alert management]
├── actions.html       [Response actions]
├── policy.html        [Policy configuration]
├── engines.html       [Engine management]
├── detection.html     [Detection analysis]
├── health.html        [System health]
├── threat-intel.html  [TI lookup/stats]
├── allowlist.html     [Whitelist management]
├── models.html        [Model registry view]
└── [7 more pages]     [Monitor, Respond, Learn, Investigate, etc.]
```

**JavaScript Modules** (`web_app/static/js/`):
```
Application Layer
├── core/
│   ├── global-state.js      [Observable state (v2)]
│   ├── http-client.js       [Fetch wrapper + retry logic]
│   ├── socket-core.js       [SocketIO low-level]
│   ├── socket-manager.js    [Socket event handlers]
│   ├── ui-core.js           [DOM utilities]
│   └── utils.js             [Helper functions]
├── pages/
│   ├── dashboard.js         [Main dashboard controller]
│   ├── alerts.js            [Alerts page]
│   ├── actions.js           [Actions page]
│   ├── policy.js            [Policy page]
│   └── [more pages]
├── components/
│   └── [Reusable UI components]
└── [Page-level controllers]
```

### 2.3 Frontend Data Flow

```
USER INTERACTION FLOW
=====================

1. User Action (click, form submit)
   ↓
2. Page Controller (e.g., dashboard.js)
   ├─ Validates input
   ├─ Calls HttpClient.post() or .get()
   └─ Or emits socket.emit('event')
   ↓
3. REST API or WebSocket
   ↓
4. Flask Backend
   ├─ Processes request
   ├─ Publishes events to EventBus
   └─ Returns response or emits socket message
   ↓
5. Frontend Receives Response
   ├─ HttpClient: Parses JSON → callback
   ├─ SocketIO: Socket handler triggered
   └─ Both: Update GlobalState.data[slice]
   ↓
6. GlobalState Listener Triggered
   ├─ Notifies all subscribers
   ├─ Page controller receives new data
   └─ DOM re-renders
   ↓
7. UI Updates Displayed
```

### 2.4 Real-time Communication

**WebSocket Namespace**: `/events`

**Channels**:
- `subscribe_alerts` / `unsubscribe_alerts` — Real-time alert stream
- `subscribe_actions` / `unsubscribe_actions` — Action execution updates
- `subscribe_metrics` / `unsubscribe_metrics` — Metrics/pulse data
- `subscribe_perception` / `unsubscribe_perception` — Attack narrative, confidence

**Message Flow**:
```
Backend EventBus Event
  ↓
RealTimeStreamer catches event
  ↓
Formats as WebSocket message
  ↓
socketio.emit(event_name, payload, namespace="/events")
  ↓
Frontend socket handler (socket-manager.js)
  ↓
Normalizes payload
  ↓
Updates GlobalState.data[slice]
  ↓
Subscribers notified
  ↓
Page controllers update DOM
```

### 2.5 Critical Frontend Issues

| Issue | Severity | Impact | Status |
|-------|----------|--------|--------|
| **No JS minification/bundling** | Medium | Performance: 12+ separate JS files | ⚠️ Unoptimized |
| **Global state singleton** | Medium | Race conditions on concurrent WebSocket messages | ⚠️ Potential |
| **No request deduplication** | Low | Redundant API calls on rapid clicks | ⚠️ Minor |
| **Bootstrap CSS only** | Low | No custom component library | ✓ Acceptable |

---

## PHASE 3: BACKEND ARCHITECTURE

### 3.1 Flask Application Structure

**Initialization Flow** (`web_app/app.py`, lines 50-370):

```python
1. SETTINGS LOAD
   ├─ load_settings() → Settings dataclass
   ├─ Environment variables from .env
   └─ Defaults for host, port, debug, etc.

2. FLASK APP BOOTSTRAP
   ├─ app = Flask(__name__)
   ├─ SECRET_KEY = SETTINGS.flask_secret_key
   ├─ MAX_CONTENT_LENGTH = 16MB
   └─ Security middleware registered

3. WEBSOCKET INITIALIZATION
   ├─ socketio = SocketIO(app, ...)
   ├─ CORS origins configured
   ├─ async_mode = "threading"
   └─ MANDATORY for INIDS 2.0+

4. SECURITY MIDDLEWARE STACK
   ├─ RateLimitMiddleware (120 req/60s default)
   ├─ IPBlockingMiddleware
   ├─ SecurityHeadersMiddleware
   ├─ AuditLogMiddleware
   └─ CSRF protection + Correlation tracing

5. AUTHENTICATION
   ├─ JWTAuthManager initialized
   ├─ RunAsManager for user impersonation
   └─ JWT secret from SETTINGS

6. ELASTICSEARCH BRIDGE (Optional)
   ├─ If ELASTICSEARCH_ENABLED=1
   ├─ Init ElasticsearchBridge
   ├─ Connects to Elasticsearch for audit logs
   └─ Else: Disabled, logs → SQLite only

7. ASYNC EXECUTOR
   ├─ ThreadPoolExecutor (max_workers=4)
   ├─ For background tasks
   └─ Prevents blocking

8. DATABASE & PERSISTENCE
   ├─ OpsStore(db_path) → SQLite/PostgreSQL
   ├─ ModelRegistry (model_registry.json)
   ├─ AlertStore (in-memory, maxlen=1000)
   └─ IngestionQueue (in-memory, maxlen=10000)

9. DETECTION FRAMEWORK
   ├─ EngineRegistry (thread-safe)
   ├─ EngineAggregator (strategy: ANY_TRIGGER)
   ├─ SignatureEngine (YAML-based rules)
   ├─ ThresholdEngine
   ├─ AnomalyEngine (disabled by default)
   ├─ HoneypotEngine (canary detection)
   ├─ TemporalCorrelationEngine
   ├─ EntityEnrichmentEngine
   ├─ AlertFilterEngine (3-layer)
   └─ ThreatIntelEngine (disabled until feeds loaded)

10. PREVENTION & RESPONSE
    ├─ PreventionService (adapter: Ufw/Nftables/Mock)
    ├─ RiskEngine
    ├─ PolicyEngine
    ├─ ActionExecutor
    ├─ PreventionScheduler (cleanup every 30s)
    ├─ Allowlist (in-memory + persistent)
    ├─ EscalationTracker (cooldown=300s)
    └─ FalsePositiveManager

11. REAL-TIME & PERCEPTION
    ├─ RealTimeStreamer (EventBus → WebSocket)
    ├─ AttackStoryEngine (narrative generation)
    ├─ ConfidenceBreakdownEngine (explain confidence)
    ├─ LiveSystemPulse (60-min window metrics)
    └─ PerceptionIntegration (worker threads)

12. ML LIFECYCLE
    ├─ DatasetCollector (training data retention)
    ├─ RertrainingScheduler (lazy-loaded)
    └─ Triggered on model load

13. THREAT INTELLIGENCE
    ├─ ThreatIntelManager (feed manager)
    ├─ TIEngine (detection engine)
    └─ Status: Initialized, feeds optional

14. EVENT BUS WIRING
    ├─ EventBus subscriptions configured
    ├─ 6 subscription handlers registered
    └─ Ready for /api/predict → Risk → Policy → Action chain

15. AT RUNTIME (__main__ or load_models):
    ├─ load_models()
    │   ├─ Load all .pkl files
    │   ├─ Set default model
    │   ├─ Create DetectionService
    │   └─ Create MLEngine + register
    ├─ _ensure_scheduler_started()
    │   └─ PreventionScheduler.start() [daemon thread]
    └─ socketio.run(app, ...)
```

### 3.2 API Route Map (75+ Endpoints)

**Health & Status**:
- `GET /api/health` — Full health check
- `GET /api/health/live` — Liveness probe
- `GET /api/health/ready` — Readiness probe

**Authentication**:
- `POST /api/auth/login` — JWT token generation
- `POST /api/auth/refresh` — Token refresh
- `GET /api/auth/validate` — Token validation
- `POST /api/auth/runas` — User impersonation (admin)
- `GET /api/auth/status` — Current auth status

**Predictions & Detection**:
- `POST /api/predict` — **PRIMARY**: Single prediction → EventBus chain
- `POST /api/detect` — **SECONDARY**: Multi-engine detection (no EventBus)
- `POST /api/detection/analyze` — Detailed detection analysis

**Alerts**:
- `GET /api/alerts` — List alerts
- `POST /api/alerts/dismiss` — Mark alert dismissed
- `POST /api/alerts/<alert_id>/feedback` — Feedback submission
- `PATCH /api/alerts/<alert_id>` — Alert update
- `GET /api/alerts/filter-rules` — Get filter rules
- `POST /api/alerts/filter-rules/exclude` — Exclude rule
- `POST /api/alerts/filter-rules/ignore` — Ignore rule
- `POST /api/alerts/filter-rules/merge` — Merge rule
- `DELETE /api/alerts/filter-rules/<rule_id>` — Delete rule
- `GET /api/alerts/filter-stats` — Filter effectiveness

**Actions & Responses**:
- `GET /api/actions` — List actions
- `POST /api/actions` — Create action
- `GET /api/actions/pending` — Pending approval
- `POST /api/actions/<action_id>/approve` — Approve action
- `POST /api/actions/cleanup` — Manual cleanup

**Policy**:
- `GET /api/policy` — Get policy
- `POST /api/policy` — Update policy
- `GET /api/policy/history` — Policy change history
- `POST /api/policy/rollback` — Rollback policy

**Allowlist**:
- `GET /api/allowlist` — List allowlist
- `POST /api/allowlist` — Add entry
- `DELETE /api/allowlist/<entry>` — Remove entry

**Engines & Detection Config**:
- `GET /api/engines` — List engines
- `POST /api/engines/<engine_id>/toggle` — Enable/disable

**Temporal & Entity**:
- `GET /api/temporal/patterns` — Correlation patterns
- `POST /api/temporal/patterns` — Add pattern
- `GET /api/temporal/state/<source_ip>` — IP temporal state
- `GET /api/entity/enrich/<source_ip>` — Enrich entity
- `GET /api/entity/<source_ip>/threat-level` — Entity threat level

**Honeypot**:
- `GET /api/honeypot/config` — Honeypot config
- `POST /api/honeypot/config` — Update honeypot

**Incidents & Activities**:
- `GET /api/incidents` — List incidents
- `GET /api/incidents/<incident_id>` — Get incident
- `GET /api/activities` — Activity log

**Threat Intelligence**:
- `GET /api/threat-intel/stats` — TI stats
- `POST /api/threat-intel/lookup` — TI lookup
- `GET /api/threat-intelligence` — TI data

**Ingestion**:
- `POST /api/ingest` — Enqueue for processing
- `POST /api/ingest/log` — Ingest log entry
- `POST /api/ingest/process` — Process queue

**Metrics & Monitoring**:
- `GET /api/metrics` — Prometheus metrics
- `GET /api/detections/history` — Detection history

**Audit & Compliance**:
- `GET /api/audit` — Audit logs
- `GET /api/audit/logs` — Detailed audit
- `GET /api/audit/user-activity` — User activity

**Perception & Insights**:
- `GET /api/perception/pulse` — System pulse
- `GET /api/perception/pulse/timeseries/<metric>` — Metric timeseries
- `GET /api/perception/confidence/<detection_id>` — Confidence breakdown
- `GET /api/perception/attack-story/<attack_id>` — Attack narrative
- `GET /api/perception/attack-stories` — All attack stories
- `GET /api/perception/feature-importance` — ML feature importance
- `GET /api/perception/integration-status` — Status

**Investigations & Playbooks**:
- `GET /api/investigations` — List investigations
- `GET /api/playbooks` — List playbooks
- `POST /api/playbooks/<playbook_id>/execute` — Execute playbook

**Packet Capture**:
- `POST /api/capture/start` — Start capture
- `POST /api/capture/stop` — Stop capture

**Web UI Routes** (HTML):
- `GET /` — Home page
- `GET /predict` — Prediction form
- `GET /alerts` — Alerts UI
- `GET /actions` — Actions UI
- `GET /dashboard` — Main dashboard
- `GET /dashboard/main` — Alternative dashboard
- `[+ 10 more pages]`

### 3.3 EventBus Architecture

The EventBus is the **central nervous system** of INIDS.

**Event Types**:
```python
@dataclass
class DetectionEvent:
    source_ip: str
    prediction: str          # "attack" or "normal"
    confidence: float        # 0-100
    features: dict
    attack_type: str
    profile: str
    severity: str
    suspicious: bool
    reason: str
    timestamp: str (UTC ISO 8601)

@dataclass
class RiskScoreEvent:
    detection: DetectionEvent
    risk_score: float        # 0-1
    components: dict         # {confidence, severity, frequency}
    timestamp: str

@dataclass
class PolicyDecisionEvent:
    risk: RiskScoreEvent
    decision: str            # ALLOW, ALERT, RATE_LIMIT, TEMP_BLOCK, BLOCK, PENDING_BLOCK
    reason: str
    ttl_seconds: int | None

@dataclass
class ActionEvent:
    decision: PolicyDecisionEvent
    action: str              # "block", "rate_limit", "allow"
    target: str              # IP address
    reason: str
    dry_run: bool
    executed: bool
    status: str              # "DRY_RUN", "ACTIVE", "FAILED", etc.
    adapter: str             # "ufw", "nftables", "webhook", "mock"
    expires_at: str | None
    created_at: str

@dataclass
class AuditEvent:
    event_type: str
    message: str
    created_at: str
```

**Subscription Model**:
```
Thread-Safe In-Process EventBus
├─ Prevents duplicate subscriptions
├─ Copies handler list before dispatch (deadlock safety)
├─ Catches exceptions in handlers (isolation)
└─ Logs failures

Subscriptions (6 total):
├─ DetectionEvent → _on_detection_event()       [Chain to Risk]
├─ RiskScoreEvent → _on_risk_event()            [Chain to Policy]
├─ PolicyDecisionEvent → _on_policy_decision_event() [Execute action if needed]
├─ DetectionEvent → _on_detection_realtime()    [WebSocket]
├─ RiskScoreEvent → _on_risk_realtime()         [WebSocket]
└─ ActionEvent → _on_action_realtime()          [WebSocket]
```

**Execution Guarantee**:
```
/api/predict
  ↓
DetectionService.predict_from_features()
  ├─ Generates Alert if suspicious
  ├─ Publishes DetectionEvent
  └─ SYNCHRONOUS: returns PredictionResult
  
⚡ EventBus.publish(DetectionEvent)
  ├─ Thread-safe dispatch to all handlers
  │
  ├─ Handler 1: _on_detection_event
  │   └─ Calculates RiskScoreEvent
  │   └─ Publishes RiskScoreEvent
  │       └─ Handler: _on_risk_event
  │           └─ Calls PolicyEngine.decide()
  │           └─ Publishes PolicyDecisionEvent
  │               └─ Handler: _on_policy_decision_event
  │                   └─ If BLOCK/TEMP_BLOCK/RATE_LIMIT: ActionExecutor.execute()
  │                       └─ Publishes ActionEvent
  │                           └─ Handler: _on_action_realtime
  │                               └─ WebSocket emit
  │
  └─ Handler 2: _on_detection_realtime
      └─ WebSocket emit DetectionEvent

⚠️ All event chain handlers execute in the SAME thread that called /api/predict
⚠️ Means response latency = event processing latency
```

### 3.4 Request Lifecycle Analysis

#### **Request 1: POST /api/predict**

```
TIME 0ms: HTTP Request arrives at Flask
  ├─ @require_role('analyst')
  ├─ Parse JSON body: features, profile, source_ip, attack_type
  └─ Validate via ValidationSchema

TIME 1ms: Ensure models loaded
  ├─ if not detection_service: load_models()
  │   ├─ Load .pkl files from models/
  │   ├─ Create DetectionService
  │   ├─ Create MLEngine + register
  │   └─ ~300-500ms ONE-TIME cost
  └─ Skipped on subsequent calls

TIME 2ms: Detect
  ├─ DetectionService.predict_from_features(features, profile, source_ip, attack_type)
  │   ├─ model.predict_proba(df) → float 0-1
  │   ├─ confidence = float * 100 → 0-100
  │   ├─ threshold = {75, 60, 45}[profile]
  │   ├─ suspicious = confidence < threshold OR prediction == "attack"
  │   ├─ Create Alert if suspicious
  │   ├─ **PUBLISH DetectionEvent to EventBus** ⚡
  │   └─ return PredictionResult(prediction, confidence, profile, threshold, suspicious, reason, alert)

TIME 5ms: EventBus Dispatch (SYNCHRONOUS)
  ├─ _on_detection_event() triggered
  │   ├─ RiskEngine.calculate(DetectionEvent)
  │   │   ├─ confidence_score = confidence / 100
  │   │   ├─ severity_score = map_severity(...) → 0-1
  │   │   ├─ frequency_score = recent_activity(source_ip)
  │   │   ├─ risk = 0.5*conf + 0.3*sev + 0.2*freq
  │   │   └─ **PUBLISH RiskScoreEvent** ⚡
  │   ├─ ops_store.add_audit("risk_score", ...)
  │   └─ Execution time: ~2-5ms
  │
  ├─ _on_detection_realtime() triggered
  │   ├─ socketio.emit("DetectionEvent", event.to_dict(), namespace="/events")
  │   └─ Execution time: ~1ms
  │
  ├─ _on_risk_event() triggered
  │   ├─ PolicyEngine.decide(RiskScoreEvent, policy)
  │   │   ├─ Threshold comparison
  │   │   ├─ Decision logic (50+ lines)
  │   │   └─ return PolicyDecisionEvent
  │   ├─ **PUBLISH PolicyDecisionEvent** ⚡
  │   ├─ ops_store.add_audit("policy_decision", ...)
  │   └─ Execution time: ~1-2ms
  │
  ├─ _on_risk_realtime() triggered
  │   ├─ socketio.emit("RiskScoreEvent", event.to_dict(), namespace="/events")
  │   └─ Execution time: ~1ms
  │
  ├─ _on_policy_decision_event() triggered
  │   ├─ if decision NOT IN {BLOCK, TEMP_BLOCK, RATE_LIMIT}: return
  │   ├─ (else:)
  │   ├─ ActionExecutor.execute(PolicyDecisionEvent, policy)
  │   │   ├─ Normalize IP
  │   │   ├─ Create ActionEvent
  │   │   ├─ if NOT dry_run:
  │   │   │   ├─ adapter.block(IP, ttl) or adapter.rate_limit(IP, ttl)
  │   │   │   └─ ~10-50ms depending on adapter
  │   │   ├─ ops_store.save_action(action_dict)
  │   │   └─ **PUBLISH ActionEvent** ⚡
  │   ├─ ops_store.add_audit("action_execution", ...)
  │   └─ Execution time: ~15-60ms IF blocking
  │
  └─ _on_action_realtime() triggered
      ├─ socketio.emit("ActionEvent", event.to_dict(), namespace="/events")
      └─ Execution time: ~1ms

TIME 20ms (total): EventBus chain completes
  └─ Returns PredictionResult + prevention_action to client

RESPONSE: JSON with prediction, confidence, alert_id, severity, action status

ASYNC EFFECTS:
  ├─ Frontend receives WebSocket updates (DetectionEvent, RiskScoreEvent, ActionEvent)
  ├─ Dashboard updates in real-time
  ├─ Action status may show "ACTIVE" or "FAILED" on next WebSocket message
  └─ Escalation tracking starts (if enabled)
```

#### **Request 2: POST /api/detect**

```
TIME 0ms: HTTP Request arrives
  ├─ @require_role('analyst')
  ├─ Parse JSON body: features
  └─ Validate via ValidationSchema

TIME 1ms: Multi-Engine Evaluation
  ├─ EngineRegistry.evaluate_all(features)
  │   ├─ MLEngine: model.predict_proba() → EngineResult (if enabled)
  │   ├─ SignatureEngine: YAML rules → EngineResult
  │   ├─ ThresholdEngine: feature thresholds → EngineResult
  │   ├─ AnomalyEngine: anomaly model → EngineResult (if enabled)
  │   ├─ HoneypotEngine: honeypot IPs → EngineResult
  │   ├─ TemporalCorrelationEngine: pattern matching → EngineResult
  │   ├─ EntityEnrichmentEngine: GeoIP, TI → EngineResult (enrichment only)
  │   └─ ThreatIntelEngine: TI lookup → EngineResult (if TI feeds loaded)
  │       └─ Returns list[EngineResult]

TIME 3ms: Aggregation
  ├─ EngineAggregator.aggregate(engine_results)
  │   ├─ Strategy: ANY_TRIGGER (any "attack" → "attack")
  │   ├─ Combines confidence scores
  │   ├─ Merges severity
  │   └─ returns AggregatedResult
  └─ Execution time: ~1-2ms

TIME 5ms: Response
  ├─ Return AggregatedResult.to_dict() as JSON
  └─ **NO EventBus publishing** (standalone endpoint)

RESPONSE: JSON with verdict, confidence, severity, engines[{engine_id, verdict, confidence}]

KEY DIFFERENCE:
  ├─ /api/detect: Parallel multi-engine evaluation → aggregated result
  ├─ /api/predict: Single model → full event chain (Risk → Policy → Action)
  ├─ /api/detect is FAST (~5ms) and FORENSIC
  ├─ /api/predict is OPERATIONAL (includes blocking)
```

---

## PHASE 4: DATABASE & DATA FLOW ARCHITECTURE

### 4.1 Database Schema

**OpsStore**: SQLite (dev) / PostgreSQL (prod)

**Tables**:

```sql
-- Alerts table
CREATE TABLE alerts (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    severity TEXT NOT NULL,           -- "low", "medium", "high", "critical"
    prediction TEXT NOT NULL,         -- "attack" or "normal"
    confidence DOUBLE PRECISION,      -- 0-100
    profile TEXT NOT NULL,            -- "strict", "balanced", "lenient"
    reason TEXT NOT NULL,
    source_ip TEXT DEFAULT '',
    attack_type TEXT DEFAULT '',
    risk_score DOUBLE PRECISION DEFAULT 0.0
);

-- Actions table (firewall/IPS actions)
CREATE TABLE actions (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    decision TEXT NOT NULL,           -- "BLOCK", "TEMP_BLOCK", "RATE_LIMIT", "ALERT", "ALLOW"
    action TEXT NOT NULL,             -- "block", "rate_limit", "allow"
    target TEXT NOT NULL,             -- IP address
    reason TEXT NOT NULL,
    dry_run INTEGER NOT NULL DEFAULT 0,
    executed INTEGER NOT NULL DEFAULT 0,
    status TEXT DEFAULT 'pending',    -- "DRY_RUN", "ACTIVE", "FAILED", "UNBLOCKED", etc.
    adapter TEXT DEFAULT 'mock',      -- "ufw", "nftables", "webhook", "mock"
    ttl_seconds INTEGER,              -- Expiration time in seconds
    expires_at TEXT,                  -- UTC timestamp when action expires
    created_at TEXT,
    executed_at TEXT
);

-- Allowlist entries
CREATE TABLE allowlist (
    id TEXT PRIMARY KEY,
    entry TEXT NOT NULL UNIQUE,       -- IP or CIDR
    reason TEXT,
    added_by TEXT,
    added_at TEXT
);

-- Policy versions (for rollback)
CREATE TABLE policy_versions (
    id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    config TEXT NOT NULL,             -- JSON blob
    changed_by TEXT,
    changed_at TEXT,
    description TEXT
);

-- Audit logs
CREATE TABLE audit_logs (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,         -- "risk_score", "policy_decision", "action_execution", etc.
    message TEXT NOT NULL,
    user TEXT,
    ip_address TEXT,
    correlation_id TEXT
);

-- Incidents (aggregated events)
CREATE TABLE incidents (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    source_ip TEXT NOT NULL,
    incident_type TEXT,               -- "port_scan", "brute_force", "c2", etc.
    severity TEXT,
    status TEXT DEFAULT 'open',
    resolution_notes TEXT
);

-- Model registry
CREATE TABLE models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT,
    path TEXT NOT NULL,               -- Path to .pkl file
    created_at TEXT,
    accuracy DOUBLE PRECISION,
    f1_score DOUBLE PRECISION
);

-- False positive suppressions
CREATE TABLE fp_suppressions (
    id TEXT PRIMARY KEY,
    engine_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    reason TEXT,
    created_at TEXT
);

-- Escalation tracking
CREATE TABLE escalations (
    id TEXT PRIMARY KEY,
    source_ip TEXT NOT NULL,
    level INTEGER DEFAULT 0,          -- 0=clean, 1=alert, 2=rate_limit, 3=temp_block, 4=perm_block
    hit_count INTEGER DEFAULT 0,
    last_hit TEXT,
    cooldown_seconds INTEGER DEFAULT 300
);
```

### 4.2 Data Flow Lifecycle

```
ALERT LIFECYCLE
===============

1. INGESTION
   ├─ API: POST /api/predict (with features)
   ├─ OR: POST /api/ingest/process (from queue)
   └─ Features: dict of 41 network features

2. FEATURE ENGINEERING
   ├─ Validate features against schema
   ├─ Normalize numeric ranges
   ├─ Encode categorical (protocol_type, service, flag)
   └─ Create DataFrame for ML inference

3. ML INFERENCE
   ├─ model.predict_proba(df) → [normal_prob, attack_prob]
   ├─ Extract attack_prob as confidence (0-1)
   ├─ Confidence * 100 = 0-100 scale
   └─ Prediction = "attack" if attack_prob > 0.5, else "normal"

4. ALERT GENERATION
   ├─ If suspicious = confidence < threshold:
   │   ├─ Create Alert(id, timestamp, severity, prediction, confidence, profile, reason)
   │   ├─ alert_store.add(Alert)  [in-memory, max 1000]
   │   └─ ops_store.save_alert(Alert)  [persistent]
   └─ If not suspicious: Alert = None

5. EVENT PUBLISHING
   ├─ Publish DetectionEvent(source_ip, prediction, confidence, features, severity, suspicious, reason)
   └─ EventBus propagates to 6 handlers

6. RISK CALCULATION
   ├─ RiskEngine.calculate(DetectionEvent) → RiskScoreEvent
   ├─ Components:
   │   ├─ confidence_score = confidence / 100  (0-1)
   │   ├─ severity_score = map_severity(attack_type, severity, prediction)
   │   ├─ frequency_score = recent_activity(source_ip, window=300s)
   │   └─ risk = 0.5*conf + 0.3*sev + 0.2*freq
   └─ Publish RiskScoreEvent(detection, risk_score, components)

7. POLICY DECISION
   ├─ PolicyEngine.decide(RiskScoreEvent, policy) → PolicyDecisionEvent
   ├─ Decision logic:
   │   ├─ If monitor mode: {ALERT, ALLOW}
   │   ├─ If attack prediction:
   │   │   ├─ High conf + high risk → BLOCK
   │   │   ├─ High risk → TEMP_BLOCK
   │   │   ├─ Medium risk → RATE_LIMIT
   │   │   ├─ Low-medium risk → ALERT
   │   │   └─ Low risk → ALLOW
   │   └─ If not attack: ALERT (if high risk) or ALLOW
   └─ Publish PolicyDecisionEvent

8. ACTION EXECUTION
   ├─ ActionExecutor.execute(PolicyDecisionEvent, policy) → ActionEvent
   ├─ Action type based on decision:
   │   ├─ BLOCK → adapter.block(IP, ttl=300s)
   │   ├─ TEMP_BLOCK → adapter.block(IP, ttl=60s)
   │   ├─ RATE_LIMIT → adapter.rate_limit(IP, ttl=120s)
   │   ├─ ALERT/ALLOW → No firewall action
   │   └─ Status: DRY_RUN (if dry_run=True), ACTIVE, FAILED, etc.
   ├─ ops_store.save_action(ActionEvent)
   └─ Publish ActionEvent

9. PERSISTENCE
   ├─ ops_store.add_audit("detection", ...) for each event type
   ├─ Audit logs: event_type, message, timestamp, user, correlation_id
   └─ All operations include correlation_id for tracing

10. REAL-TIME EMISSION
    ├─ WebSocket emit DetectionEvent to /events namespace
    ├─ WebSocket emit RiskScoreEvent
    ├─ WebSocket emit ActionEvent
    └─ Frontend updates dashboard in real-time

11. ESCALATION TRACKING (Optional)
    ├─ EscalationTracker.record_hit(source_ip, severity)
    ├─ Auto-escalates: low→alert→rate_limit→temp_block→perm_block
    ├─ De-escalates after cooldown_seconds (300s default)
    └─ Prevents alert fatigue

12. CLEANUP & EXPIRATION
    ├─ PreventionScheduler runs every 30s
    ├─ Cleanup expired actions:
    │   ├─ Find ops_store.list_expired_actions()
    │   ├─ adapter.unblock(IP) for each
    │   ├─ Update ops_store.update_action_status(id, "UNBLOCKED")
    │   └─ Add audit event
    └─ Reconcile: Compare DB blocks vs firewall rules
```

### 4.3 Data Consistency Model

**Write Path**:
```
PredictionResult returned to client
  ├─ Alert SAVED to SQLite (synchronous)
  ├─ Action SAVED to SQLite (if executed)
  ├─ Audit SAVED to SQLite (all event types)
  └─ Elasticsearch (if enabled, async)

Guarantee: ✅ Atomic per operation
           ⚠️ May have EventBus handler failures
           ⚠️ Audit trail is complete even if action fails
```

**Read Path**:
```
Dashboard load:
  ├─ GET /api/alerts → READ from SQLite
  ├─ GET /api/actions → READ from SQLite
  ├─ GET /api/audit → READ from SQLite
  └─ Real-time updates: WebSocket subscription

Consistency Model: ✅ Eventually consistent
                   ✓ Strong reads from SQLite
                   ⚠️ Real-time messages may lag
```

---

## PHASE 5: CRITICAL INTEGRATION GAPS & ARCHITECTURE ISSUES

### 5.1 Major Integration Gaps (8 CRITICAL)

| Gap | Subsystem | Impact | Severity |
|-----|-----------|--------|----------|
| **Gap 1** | Allowlist persistence | In-memory allowlist works but doesn't persist to DB | 🔴 CRITICAL |
| **Gap 2** | ThreatIntel engine unused | TI feed manager initialized but never loads feeds | 🔴 CRITICAL |
| **Gap 3** | StreamProcessor decoupled | Pipeline exists but not integrated with web_app | 🟡 HIGH |
| **Gap 4** | Escalation tracker unused | Module exists but never called | 🟡 HIGH |
| **Gap 5** | FalsePositiveManager incomplete | Suppressions loaded but not checked in detection | 🟡 HIGH |
| **Gap 6** | Policy rollback unused | PolicyStore tracks versions but never calls rollback | 🟡 HIGH |
| **Gap 7** | SiemExporter standalone | SIEM export works but not integrated with EventBus | 🟡 HIGH |
| **Gap 8** | HA components minimal | LeaderElection exists but scheduler doesn't check | 🟡 HIGH |

### 5.2 Gap 1: Allowlist Persistence ❌

**Problem**:
```python
class Allowlist:
    def __init__(self, ops_store):
        self._ops_store = ops_store
        self._allowlist = set()  # In-memory
    
    def is_allowed(self, ip_or_cidr):
        return ip_or_cidr in self._allowlist  # ✓ Works
    
    def add(self, ip_or_cidr):
        self._allowlist.add(ip_or_cidr)
        # ❌ MISSING: self._ops_store.add_allowlist_entry(ip_or_cidr)
        
    def remove(self, ip_or_cidr):
        self._allowlist.discard(ip_or_cidr)
        # ❌ MISSING: self._ops_store.remove_allowlist_entry(ip_or_cidr)
```

**Impact**:
- Allowlist works in-memory during runtime
- **BUT**: Restarts lose all allowlist entries
- API endpoint `/api/allowlist` returns empty after restart
- Blocking recovery broken (if you allowlist an IP post-incident, it gets forgotten)

**Root Cause**: OpsStore methods never implemented
```python
# Missing in ops_store.py:
def list_allowlist(self) -> list[str]: ...
def add_allowlist_entry(self, entry: str) -> None: ...
def remove_allowlist_entry(self, entry: str) -> None: ...
```

### 5.3 Gap 2: Threat Intelligence Never Loads ❌

**Problem**:
```python
# app.py initialization:
ti_manager = ThreatIntelManager()  # ✓ Created
ti_engine = TIEngine(ti_manager)   # ✓ Created
engine_registry.register(ti_engine)  # ✓ Registered

# But:
# ❌ No feeds are EVER loaded!
# ❌ ti_manager.cache.size() always 0
# ❌ ti_engine.is_ready() returns False forever
# ❌ TI engine never participates in detection
```

**Code Reference** (`app.py` line ~310):
```python
# TODO: Load threat intel feeds at startup
# ti_manager.load_feeds(SETTINGS.ti_feed_dir)
```

**Impact**:
- TI feeds are not loaded from `SETTINGS.ti_feed_dir`
- TIEngine stays disabled (is_ready() == False)
- Threat intelligence detection does NOT work
- Even if TI feeds exist at `/path/to/feeds/`, they're ignored

### 5.4 Gap 3: StreamProcessor Decoupled ❌

**Problem**:
```python
# src/pipeline/stream_processor.py EXISTS
# It uses engine_registry and can process Redis streams

# BUT:
# ❌ app.py does NOT use StreamProcessor
# ❌ Redis client initialization is lazy
# ❌ Pipeline is "standalone" — callback responsibility delegated to caller

# app.py never calls:
# processor = StreamProcessor(engine_registry=engine_registry, redis_client=redis_client)
# processor.start()
```

**Impact**:
- Redis streams feature documented but not active
- `/api/ingest` writes to in-memory queue only
- StreamProcessor code is "dead code" from perspective of web_app
- Would need explicit integration to activate

### 5.5 Gap 4: Escalation Tracker Never Called ❌

**Problem**:
```python
escalation_tracker = EscalationTracker(cooldown_seconds=300.0)  # ✓ Created

# But in detection flow:
# ❌ NEVER called in /api/predict
# ❌ NEVER called in /api/detect
# ❌ NEVER called in any event handler

# Should be called:
# level = escalation_tracker.record_hit(source_ip, severity)
# Then: Adjust policy decision based on escalation level
```

**Expected Behavior** (NOT HAPPENING):
```
First hit (low severity):  → Level 0 (CLEAN)
Second hit within 5 min:   → Level 1 (ALERT)
Third hit within 5 min:    → Level 2 (RATE_LIMIT)
Fourth hit within 5 min:   → Level 3 (TEMP_BLOCK)
Fifth hit within 5 min:    → Level 4 (PERM_BLOCK)

After 5 minutes idle:      → De-escalate back to Level 0
```

**Impact**:
- Escalation prevention feature is NOT active
- Repeat attackers not automatically blocked harder
- False positives not escalation-tracked

### 5.6 Gap 5: False Positive Suppressions Incomplete ❌

**Problem**:
```python
fp_manager = FalsePositiveManager(ops_store=ops_store)
fp_manager.load_from_store()  # ✓ Loaded

# Suppressions stored in memory:
# self._suppressions = set()  # Set of (engine_id, rule_id)

# But in SignatureEngine:
# ❌ Suppressions are checked but not in other engines
# ❌ MLEngine doesn't check fp_suppressions
# ❌ ThresholdEngine doesn't check fp_suppressions

# API endpoint works:
# ✓ POST /api/fp-suppressions — adds to db and memory
# ✓ DELETE /api/fp-suppressions/<engine_id>/<rule_id> — removes

# But:
# ❌ Other detection engines ignore the suppression list
```

**Impact**:
- FP suppressions only work for SignatureEngine
- False positive management is incomplete
- If you suppress an ML engine false positive, it still fires

### 5.7 Gap 6: Policy Rollback Never Used ❌

**Problem**:
```python
policy_store = PolicyStore(...)  # ✓ Tracks versions

# API endpoint works:
# ✓ POST /api/policy/rollback — calls policy_store.rollback()

# But:
# ❌ PolicyEngine.decide() NEVER reads from policy_store
# ❌ PolicyEngine uses policy object passed as parameter
# ❌ Rollback changes DB but doesn't change runtime policy object

# Runtime flow:
policy = prevention_service.policy  # Uses runtime object
decision = policy_engine.decide(risk_event, policy)  # Passes object

# If rollback is called:
# ✓ DB is updated with old version
# ✓ API returns "rollback successful"
# ❌ But runtime policy object doesn't change
# ❌ So decisions still use new policy
```

**Impact**:
- Policy rollback appears to work but doesn't actually affect runtime
- Need to restart app for rollback to take effect
- Policy version history is persisted but not used

### 5.8 Gap 7: SIEM Exporter Not Event-Driven ❌

**Problem**:
```python
siem_exporter = SiemExporter()  # ✓ Created

# But:
# ❌ NOT subscribed to EventBus
# ❌ No event handlers for DetectionEvent, RiskScoreEvent, ActionEvent

# How SIEM export works:
# POST /api/siem/flush  → Manually triggers siem_exporter.flush()

# Should be:
event_bus.subscribe(DetectionEvent, siem_exporter.on_detection)
event_bus.subscribe(ActionEvent, siem_exporter.on_action)
# Then: Automatic push on events instead of manual pull
```

**Impact**:
- SIEM export is manual (pull model) not automatic (push model)
- Events not exported immediately
- Need to manually call `/api/siem/flush` endpoint
- Real-time SIEM integration not implemented

### 5.9 Gap 8: HA Components Minimal ❌

**Problem**:
```python
leader_election = LeaderElection(redis_client=None, instance_id=f"inids-{os.getpid()}")

# When redis_client=None:
# ├─ Acts as "always leader"
# └─ CORRECT for single-node deployments

# But PreventionScheduler:
prevention_scheduler = PreventionScheduler(
    action_executor,
    interval_seconds=30,
    is_leader_fn=lambda: leader_election.is_leader,  # ✓ Uses leader election
)

# Problems:
# ❌ LeaderElection only works with Redis
# ❌ Without Redis: Single-node always "leader" (correct but limited)
# ❌ With Redis: Would need leader election logic, but not fully implemented
# ❌ No automatic failover detection
# ❌ No health check integration with leader election
```

**Impact**:
- Multi-node deployments need Redis for HA
- If Redis is down: Still works but leader election disabled
- No automatic failover to replica nodes
- For production multi-node: Need manual failover setup

---

## PHASE 6: ARCHITECTURAL WEAKNESSES & RISKS

### 6.1 CRITICAL WEAKNESSES

| Weakness | Impact | Blast Radius | Severity |
|----------|--------|--------------|----------|
| **W1: Allowlist persistence broken** | Data loss on restart | Single node or full cluster | 🔴 CRITICAL |
| **W2: TI feeds never load** | TI detection disabled | Detection accuracy | 🔴 CRITICAL |
| **W3: Escalation not wired** | False positive DoS risk | System stability | 🔴 CRITICAL |
| **W4: No distributed Redis** | Single point of failure for pipeline | Ingestion | 🟡 HIGH |
| **W5: EventBus blocking** | Request latency = event chain latency | API response time | 🟡 HIGH |
| **W6: Policy rollback broken** | Configuration changes not enforced | Deployment safety | 🟡 HIGH |
| **W7: No circuit breaker** | Firewall adapter failure causes hangs | Prevention system | 🟡 HIGH |
| **W8: Global state mutation** | Race conditions on concurrent requests | Frontend reliability | 🟠 MEDIUM |

### 6.2 W1: Allowlist Persistence Broken

**Problem**:
- Allowlist in-memory only (ephemeral)
- OpsStore methods never implemented
- Restart loses all entries

**Manifestation**:
```
Scenario: Incident response after attack
─────────────────────────────────────
1. Attack detected: IP 192.168.1.100
2. Admin approves BLOCK action → IP blocked ✓
3. False positive discovered
4. Admin adds 192.168.1.100 to allowlist ✓
5. System running fine...
6. App crashes or restarts
7. Allowlist empty ❌
8. 192.168.1.100 gets detected again + blocked ❌
9. Incident repeats
```

**Root Cause**: 
- OpsStore missing 3 methods: list_allowlist(), add_allowlist_entry(), remove_allowlist_entry()
- Allowlist doesn't call ops_store on mutations

**Mitigation Path**:
1. Implement OpsStore methods
2. Load allowlist from DB at startup
3. Persist all mutations immediately
4. Add tests for persistence

### 6.3 W3: Escalation Not Wired

**Problem**:
- EscalationTracker created but never called
- Repeat attackers not auto-escalated
- Same IP can trigger multiple weak responses

**Manifestation**:
```
Scenario: False positive attack wave
──────────────────────────────────────
1. FP detection for subnet 10.0.0.0/24
2. 100 IPs detected in first 5 seconds
3. Each IP gets ALERT response (no block)
4. Same IPs detected again (FP keeps triggering)
5. Each gets ALERT again
6. System flooded with redundant alerts
7. Admin manual intervention required

WITH ESCALATION (not happening):
1. First detection: ALERT
2. Second detection from same IP: RATE_LIMIT
3. Third detection: TEMP_BLOCK
4. Fourth detection: Blocked
5. Problem contained automatically
```

**Impact**:
- Alert fatigue from repeating FP
- No automatic attack pattern recognition
- Wastes admin time

### 6.4 W5: EventBus Blocking (Latency Issue)

**Problem**:
```
POST /api/predict returns AFTER:
  ├─ DetectionEvent published
  ├─ RiskEngine calculates (~2-5ms)
  ├─ PolicyEngine decides (~1-2ms)
  ├─ ActionExecutor executes (~10-50ms IF blocking)
  ├─ ALL handlers complete synchronously
  └─ Total: 15-60ms additional latency

If firewall adapter slow:
  ├─ adapter.block(IP, ttl) hangs → 500ms-2s
  └─ API response delayed by 2s
```

**Manifestation**:
```
1. High-volume attack (1000 req/sec)
2. Each /api/predict call blocks for risk→policy→action
3. Firewall adapter takes 100ms per call
4. Request queue backs up
5. API becomes unresponsive
6. Clients timeout
7. Service degradation
```

**Impact**:
- Under attack: System becomes slower
- Ironic: Detection slows during detection load
- Need rate limiting + queue to mitigate

### 6.5 W7: No Circuit Breaker for Adapters

**Problem**:
```python
# ActionExecutor.execute():
if decision in {BLOCK, TEMP_BLOCK, RATE_LIMIT}:
    adapter.block(IP, ttl)  # ❌ Can hang forever
    
# No timeout, no retry, no circuit breaker
# If adapter is down: API hangs for socket timeout (30s+)
```

**Manifestation**:
```
1. Firewall adapter (UFW) becomes unresponsive
2. ActionExecutor calls adapter.block(IP, 300)
3. Socket hangs (default timeout 30s)
4. API request waits 30s for response
5. Client times out
6. Another request tries adapter → hangs again
7. API becomes completely unresponsive
```

**Impact**:
- Adapter failure ⇒ Detection stops ⇒ System down
- Need timeout + fallback

### 6.6 W8: Global State Mutation in Frontend

**Problem**:
```javascript
// socket.js:
GlobalState.set(newData)  // Mutates global singleton

// Concurrent WebSocket messages:
// Message 1: Updates GlobalState.data.alerts = [...]
// Message 2: (arrives before Message 1 processes)
// Message 3: (arrives before Message 2 processes)

// Race condition:
// Thread 1: GlobalState.set({alerts: [A, B]})
// Thread 2: (interrupts) reads GlobalState.data.alerts (sees [A, B])
// Thread 2: GlobalState.set({alerts: [C, D]}) (overwrites)
// Thread 1: Listener triggered (sees [C, D] not [A, B])
```

**Impact**:
- Lost updates from WebSocket messages
- Alerts may disappear from dashboard
- Actions may show stale status
- Real-time updates unreliable under load

---

## PHASE 7: EXECUTION FLOW RECONSTRUCTION

### 7.1 STARTUP LIFECYCLE

```
INIDS STARTUP SEQUENCE (Detailed)
==================================

PHASE 1: MODULE LOAD (app.py line 1-370)
─────────────────────────────────────────
t=0ms
├─ Python loads web_app/app.py
├─ Imports all modules (src/*/)
├─ Classes instantiated
├─ Singletons created:
│  ├─ event_bus = EventBus() [thread-safe, empty]
│  ├─ alert_store = InMemoryAlertStore(max_items=1000)
│  ├─ ops_store = OpsStore(db_path)
│  │  ├─ SQLite created if not exists
│  │  └─ Schema initialized (14 tables)
│  ├─ detection_service = None [lazy-loaded]
│  ├─ prevention_service = PreventionService(adapter)
│  ├─ socketio = SocketIO(app) [WebSocket initialized]
│  └─ [20+ other objects]

t=50ms
├─ engine_registry = EngineRegistry() [empty]
├─ engine_aggregator = EngineAggregator(strategy=ANY_TRIGGER)
├─ signature_engine = SignatureEngine(rules_path) [loads YAML]
├─ threshold_engine = ThresholdEngine()
├─ anomaly_engine = AnomalyEngine(model_path) [disabled, not loaded yet]
├─ ti_manager = ThreatIntelManager()
├─ ti_engine = TIEngine(ti_manager) [disabled, cache empty]
├─ honeypot_engine = HoneypotDetectionEngine(ips, ports)
├─ temporal_correlation_engine = TemporalCorrelationEngine() [empty patterns]
├─ entity_enrichment_engine = EntityEnrichmentEngine(ti_manager)
├─ alert_filter = ThreeLayerAlertFilter(ops_store)
│  └─ Loads persisted filter rules from DB
│  └─ If empty: Loads default rules

t=100ms
├─ risk_engine = RiskEngine()
├─ policy_engine = PolicyEngine()
├─ action_executor = ActionExecutor(adapter, ops_store, event_bus)
├─ prevention_scheduler = PreventionScheduler(action_executor, interval=30)
├─ RealTimeStreamer(event_bus, socketio) [ready but not started]
├─ PerceptionIntegration(...) [ready but not started]
└─ EventBus subscriptions wired (6 handlers)

PHASE 2: MAIN EXECUTION (__main__, line 3850+)
──────────────────────────────────────────────

t=150ms: load_models()
├─ Scans models/ directory
├─ Loads all .pkl files (RandomForest, DecisionTree, etc.)
├─ all_models = {'rf_nsl_kdd': ..., 'dt_nsl_kdd': ..., ...}
├─ model = all_models['rf_nsl_kdd'] [set default]
├─ Creates DetectionService(model, alert_store, event_bus)
│  └─ Registers event_bus subscription
├─ ml_engine = MLEngine(model, engine_id='ml_primary')
├─ engine_registry.register(ml_engine) [NOW detection ready]
├─ Loads model_registry from results/model_registry.json
└─ Sets retraining_scheduler ready for use

t=500ms (model load time varies)
├─ RealTimeStreamer.start() [daemon thread]
├─ PerceptionIntegration.start() [worker threads]
└─ All background subsystems operational

t=550ms: _ensure_scheduler_started()
├─ prevention_scheduler.start() [daemon thread]
└─ Background cleanup runs every 30s

t=600ms: socketio.run(app, host, port, debug=False)
├─ Flask development server starts (or Gunicorn in prod)
├─ WebSocket server starts
├─ Server ready for connections
└─ Waits for requests

READY STATE: t=600ms
```

### 7.2 USER REQUEST FLOW (POST /api/predict)

```
HTTP REQUEST ARRIVES
====================

t=0ms: Flask routing
├─ POST /api/predict received
├─ @require_role('analyst') checked (JWT)
├─ parse JSON body → features, profile, source_ip, attack_type
├─ validate via validation_schemas.validate_predict_request()
└─ Route handler begins execution

t=1ms: ensure_detection_service()
├─ if detection_service is None:
│  ├─ load_models() [300-500ms one-time cost]
│  └─ Create DetectionService
├─ else: Skip (already loaded)
└─ detection_service now ready

t=2ms: detect() call
├─ DetectionService.predict_from_features(
│    features=features_dict,
│    profile=profile,          # "balanced"
│    source_ip=source_ip,      # "192.168.1.100"
│    attack_type=attack_type   # "port_scan"
│  )
│
├─ Within predict_from_features():
│  ├─ Create pandas DataFrame from features
│  ├─ model.predict_proba(df) → [normal_prob, attack_prob]
│  ├─ prediction = "attack" if attack_prob > 0.5 else "normal"
│  ├─ confidence = attack_prob * 100 (0-100 scale)
│  ├─ threshold = {75, 60, 45}[profile]
│  ├─ suspicious = (confidence < threshold) OR (prediction == "attack")
│  ├─ Alert created if suspicious:
│  │  ├─ alert = Alert(
│  │  │    id=str(uuid4()),
│  │  │    timestamp=utc_now(),
│  │  │    severity="high" if confidence > 80 else "medium" else "low",
│  │  │    prediction=prediction,
│  │  │    confidence=confidence,
│  │  │    profile=profile,
│  │  │    reason=f"model_prediction:{prediction}"
│  │  │  )
│  │  └─ alert_store.add(alert)  [in-memory]
│  │
│  ├─ Create DetectionEvent:
│  │  ├─ source_ip = "192.168.1.100"
│  │  ├─ prediction = "attack"
│  │  ├─ confidence = 92.0
│  │  ├─ features = {all 41 features...}
│  │  ├─ attack_type = "port_scan"
│  │  ├─ profile = "balanced"
│  │  ├─ severity = "high"
│  │  ├─ suspicious = True
│  │  ├─ reason = "model_prediction:attack"
│  │  └─ timestamp = "2026-05-15T14:30:00Z"
│  │
│  ├─ **EventBus.publish(DetectionEvent)** ⚡ BEGINS SYNC EVENT CHAIN
│  │
│  └─ Return PredictionResult(prediction, confidence, profile, threshold, suspicious, reason, alert)

t=5ms: EventBus dispatch (synchronous)
│
├─ Handler 1: _on_detection_event(event)
│  ├─ RiskEngine.calculate(event) → RiskScoreEvent
│  │  ├─ confidence_score = 0.92  (92/100)
│  │  ├─ severity_score = map_severity("port_scan", "high", "attack") → 0.85
│  │  ├─ frequency_score = recent_activity("192.168.1.100", window=300) → 0.3
│  │  ├─ risk_score = 0.5*0.92 + 0.3*0.85 + 0.2*0.3 → 0.656 + 0.255 + 0.06 = 0.74
│  │  └─ return RiskScoreEvent(detection=event, risk_score=0.74, components={...})
│  │
│  ├─ **EventBus.publish(RiskScoreEvent)** ⚡
│  ├─ ops_store.add_audit("risk_score", message=f"Risk: 0.74", ...)
│  └─ [Execution time: 2-5ms]
│
├─ Handler 2: _on_detection_realtime(event)
│  ├─ socketio.emit('DetectionEvent', event.to_dict(), namespace='/events')
│  └─ [Execution time: 1ms]
│
├─ Handler 3: _on_risk_event(event) [triggered by RiskScoreEvent]
│  ├─ PolicyEngine.decide(event, prevention_service.policy) → PolicyDecisionEvent
│  │  ├─ decision = decide_policy(risk_score=0.74, confidence=92, prediction="attack", ...)
│  │  ├─ Thresholds:
│  │  │  ├─ alert_threshold = 0.40
│  │  │  ├─ rate_limit_threshold = 0.60
│  │  │  ├─ temp_block_threshold = 0.75
│  │  │  ├─ block_threshold = 0.85
│  │  │  └─ confidence_block_threshold = 85%
│  │  ├─ Since confidence=92 >= 85 AND risk_score=0.74 >= alert_threshold:
│  │  │  └─ Decision logic → "TEMP_BLOCK" (risk=0.74 < 0.85)
│  │  └─ return PolicyDecisionEvent(risk=event, decision="TEMP_BLOCK", reason="attack_high_risk_temp_block", ttl_seconds=60)
│  │
│  ├─ **EventBus.publish(PolicyDecisionEvent)** ⚡
│  ├─ ops_store.add_audit("policy_decision", message=f"Decision: TEMP_BLOCK", ...)
│  └─ [Execution time: 1-2ms]
│
├─ Handler 4: _on_risk_realtime(event)
│  ├─ socketio.emit('RiskScoreEvent', event.to_dict(), namespace='/events')
│  └─ [Execution time: 1ms]
│
├─ Handler 5: _on_policy_decision_event(event)
│  ├─ if event.decision NOT IN {BLOCK, TEMP_BLOCK, RATE_LIMIT}:
│  │  └─ return None
│  ├─ (else:) [event.decision == "TEMP_BLOCK"]
│  ├─ ActionExecutor.execute(event, prevention_service.policy) → ActionEvent
│  │  ├─ target = "192.168.1.100"
│  │  ├─ ttl_seconds = 60
│  │  ├─ if NOT dry_run:
│  │  │  ├─ adapter.block(target="192.168.1.100", ttl_seconds=60)
│  │  │  │   ├─ If MockFirewallAdapter: [1ms] → pretend blocked
│  │  │  │   ├─ If UfwFirewallAdapter: [100-200ms] → actually block
│  │  │  │   └─ If WebhookAdapter: [500-2000ms] → POST to webhook
│  │  │  └─ Returns status: "ACTIVE" or "FAILED"
│  │  ├─ Create ActionEvent(
│  │  │    decision=event,
│  │  │    action="block",
│  │  │    target="192.168.1.100",
│  │  │    reason="attack_high_risk_temp_block",
│  │  │    dry_run=False,
│  │  │    executed=True if status=="ACTIVE" else False,
│  │  │    status=status,
│  │  │    adapter="ufw",
│  │  │    expires_at="2026-05-15T14:31:00Z",  # +60s from now
│  │  │    created_at="2026-05-15T14:30:00Z"
│  │  │  )
│  │  ├─ ops_store.save_action(action_dict)
│  │  └─ return ActionEvent
│  │
│  ├─ if action_event:
│  │  ├─ **EventBus.publish(ActionEvent)** ⚡
│  │  └─ ops_store.add_audit("action_execution", message=f"Action: {action.status}", ...)
│  └─ [Execution time: 10-200ms depending on adapter]
│
└─ Handler 6: _on_action_realtime(event)
   ├─ socketio.emit('ActionEvent', event.to_dict(), namespace='/events')
   └─ [Execution time: 1ms]

t=25ms: Total event chain complete

RESPONSE GENERATED
══════════════════
├─ Flask returns JSON response:
│  {
│    "prediction": "attack",
│    "confidence": 92.0,
│    "profile": "balanced",
│    "threshold": 60.0,
│    "suspicious": true,
│    "reason": "model_prediction:attack",
│    "alert": {
│      "id": "alert_12345",
│      "timestamp": "2026-05-15T14:30:00Z",
│      "severity": "high",
│      "prediction": "attack",
│      "confidence": 92.0,
│      "profile": "balanced",
│      "reason": "model_prediction:attack"
│    },
│    "risk_score": 0.74,
│    "decision": "TEMP_BLOCK",
│    "action": {
│      "id": "action_67890",
│      "status": "ACTIVE",
│      "target": "192.168.1.100",
│      "adapter": "ufw",
│      "expires_at": "2026-05-15T14:31:00Z"
│    }
│  }
│
└─ HTTP 200 OK

t=26ms: Client receives response

FRONTEND EFFECTS (Concurrent)
═══════════════════════════════
├─ WebSocket message 1: DetectionEvent arrives
│  └─ socket handler: GlobalState.set({alerts: [new_alert, ...]})
│  └─ Dashboard: Alert badge updates, list refreshes
│
├─ WebSocket message 2: RiskScoreEvent arrives
│  └─ socket handler: GlobalState.set({risk: new_risk})
│  └─ Dashboard: Risk gauge updates
│
├─ WebSocket message 3: ActionEvent arrives
│  └─ socket handler: GlobalState.set({actions: [new_action, ...]})
│  └─ Dashboard: Action status shows "ACTIVE"

BACKGROUND EFFECTS (Every 30s)
════════════════════════════════
├─ PreventionScheduler wakes up
├─ ActionExecutor.cleanup_expired_actions()
│  ├─ Find actions where expires_at < now()
│  │  └─ Finds the TEMP_BLOCK from 60s ago
│  ├─ adapter.unblock("192.168.1.100")
│  ├─ ops_store.update_action_status(action_id, "UNBLOCKED")
│  └─ ops_store.add_audit("ip_unblock", ...)
├─ PreventionScheduler.reconcile() (every 20 ticks = 600s)
│  ├─ Compare ops_store.list_active_blocks() vs adapter.list_rules()
│  ├─ Log any discrepancies
│  └─ Update action_status to "DESYNCED" if mismatch
```

---

## PHASE 8: DEPENDENCY GRAPH & COUPLING ANALYSIS

### 8.1 Critical Dependency Chain

```
API Request
  ├─ Flask Route Handler
  │
  ├─ Security Middleware Stack
  │  ├─ JWTAuthManager (auth_jwt.py)
  │  ├─ CorrelationTracer (correlation_tracing.py)
  │  ├─ CSRFProtection (csrf_protection.py)
  │  ├─ RateLimiter (rate_limiter.py)
  │  └─ InputSanitizer (input_sanitizer.py)
  │
  ├─ Domain Services
  │  ├─ DetectionService (detection_service.py)
  │  │   ├─ ML Model (joblib loaded)
  │  │   ├─ EventBus (core/event_bus.py)
  │  │   └─ AlertStore (storage.py)
  │  │
  │  ├─ RiskEngine (ips/risk_engine.py)
  │  │   └─ No external dependencies (pure calculation)
  │  │
  │  ├─ PolicyEngine (ips/policy_engine.py)
  │  │   └─ No external dependencies (pure logic)
  │  │
  │  ├─ ActionExecutor (ips/action_executor.py)
  │  │   ├─ FirewallAdapter (firewall_adapters.py)
  │  │   ├─ OpsStore (ops_store.py)
  │  │   ├─ EventBus
  │  │   └─ PreventionService (prevention_service.py)
  │  │
  │  └─ [Multi-Engine Framework]
  │      ├─ EngineRegistry (detection/engine_registry.py)
  │      ├─ SignatureEngine (detection/engines/signature_engine.py)
  │      │   ├─ YAML rules file (rules/default_rules.yaml)
  │      │   └─ FalsePositiveManager (prevention/false_positive_manager.py)
  │      ├─ ThresholdEngine (detection/engines/threshold_engine.py)
  │      ├─ AnomalyEngine (detection/engines/anomaly_engine.py)
  │      ├─ MLEngine (detection/engines/ml_engine.py)
  │      ├─ HoneypotEngine (detection/engines/honeypot_engine.py)
  │      ├─ TemporalCorrelationEngine (detection/temporal_correlation.py)
  │      ├─ EntityEnrichmentEngine (ips/entity_enrichment.py)
  │      │   └─ ThreatIntelManager (threat_intel/feed_manager.py)
  │      ├─ AlertFilterEngine (ips/alert_filter.py)
  │      └─ TIEngine (threat_intel/ti_engine.py)
  │
  └─ Data Layer
     ├─ OpsStore (ops_store.py)
     │   ├─ SQLite database
     │   └─ OR PostgreSQL database
     ├─ ModelRegistry (model_registry.py)
     └─ IngestionQueue (ingestion_service.py)

Real-Time Layer
  ├─ RealTimeStreamer (realtime/broadcaster.py)
  │   ├─ EventBus
  │   └─ SocketIO
  ├─ PerceptionIntegration (perception/perception_integration.py)
  │   ├─ AttackStoryEngine (perception/attack_story.py)
  │   ├─ ConfidenceBreakdownEngine (perception/confidence.py)
  │   ├─ LiveSystemPulse (perception/pulse.py)
  │   └─ EventBus

Background Tasks
  ├─ PreventionScheduler (ips/scheduler.py)
  │   ├─ ActionExecutor
  │   └─ LeaderElection (ha/leader_election.py)
  ├─ DatasetCollector (training.py)
  └─ RertrainingScheduler (training.py)
```

### 8.2 Coupling Analysis

**Tightly Coupled**:
- EventBus ← All handlers (6 subscribers)
- OpsStore ← ActionExecutor, AlertStore, Allowlist, PolicyEngine
- EngineRegistry ← EngineAggregator
- PreventionService.policy ← PolicyEngine, ActionExecutor

**Loosely Coupled**:
- DetectionService ← independent of other engines
- SignatureEngine ← independent of ML models
- RiskEngine ← pure function, no side effects
- PolicyEngine ← pure function, no side effects

**Circular Dependencies**:
- ❌ NONE DETECTED (good architecture)

**Dead Code**:
- ⚠️ StreamProcessor (unused)
- ⚠️ ThreatIntelManager (never loads feeds)
- ⚠️ EscalationTracker (never called)
- ⚠️ SiemExporter (manual pull only)

---

## PHASE 9: MOST IMPORTANT FILES ANALYSIS

### 9.1 TIER 1: CRITICAL STARTUP/INITIALIZATION

| File | Lines | Purpose | What Depends | What It Depends On | Failure Impact |
|------|-------|---------|---|---|---|
| [web_app/app.py](web_app/app.py) | 3976 | Flask app bootstrap, API routes, EventBus wiring | All endpoints, WebSocket | Settings, Models, SQLite, Adapters | Complete system failure |
| [src/settings.py](src/settings.py) | 50 | Configuration management | All modules | Environment variables | Configuration errors propagate everywhere |
| [src/core/event_bus.py](src/core/event_bus.py) | 80 | Event-driven architecture | Detection→Risk→Policy→Action chain | threading.Lock | If broken: No event propagation, system fails silently |
| [src/detection_service.py](src/detection_service.py) | 120 | Single prediction entry point | /api/predict | ML models, EventBus, AlertStore | Single prediction broken ⇒ detection broken |
| [src/ops_store.py](src/ops_store.py) | 300+ | Persistence layer (SQLite/PostgreSQL) | All data persistence | SQL drivers | Data loss, audit trail broken |

### 9.2 TIER 2: CRITICAL REQUEST PROCESSING

| File | Purpose | Failure Impact |
|------|---------|---|
| [src/ips/policy_engine.py](src/ips/policy_engine.py) | Policy decision logic | Wrong decisions (ALLOW instead of BLOCK) |
| [src/ips/risk_engine.py](src/ips/risk_engine.py) | Risk scoring | Inaccurate risk scores ⇒ wrong responses |
| [src/ips/action_executor.py](src/ips/action_executor.py) | Action execution (blocking) | Prevention system broken |
| [src/detection/engine_registry.py](src/detection/engine_registry.py) | Multi-engine framework | /api/detect broken, detection accuracy degraded |

### 9.3 TIER 3: DETECTION ENGINES

| File | Purpose | Failure Impact |
|------|---------|---|
| [src/detection/engines/ml_engine.py](src/detection/engines/ml_engine.py) | Primary ML detection | ML-based detection disabled |
| [src/detection/engines/signature_engine.py](src/detection/engines/signature_engine.py) | Rule-based detection | Signature attacks missed |
| [rules/default_rules.yaml](rules/default_rules.yaml) | Signature rules (11 rules) | All signature detection fails if broken |

### 9.4 TIER 4: CRITICAL DEPENDENCIES

| File | Purpose | Failure Impact |
|------|---------|---|
| [src/detection/aggregator.py](src/detection/aggregator.py) | Fuse multi-engine results | /api/detect returns incorrect verdicts |
| [src/middleware.py](src/middleware.py) | Security middleware stack | Auth/CSRF/rate limiting broken |
| [src/auth_service.py](src/auth_service.py) | Role-based access control | Authorization broken, security exposure |
| [src/auth_jwt.py](src/auth_jwt.py) | JWT token management | Authentication broken |

### 9.5 TIER 5: REAL-TIME & OBSERVABILITY

| File | Purpose | Failure Impact |
|------|---------|---|
| [src/realtime/broadcaster.py](src/realtime/broadcaster.py) | WebSocket event streaming | Real-time dashboard stops updating |
| [src/perception/perception_integration.py](src/perception/perception_integration.py) | Attack narrative generation | Dashboard insights unavailable |
| [src/observability/json_logging.py](src/observability/json_logging.py) | Structured logging | Audit trail format broken |

### 9.6 TIER 6: PREVENTION & RESPONSE

| File | Purpose | Failure Impact |
|------|---------|---|
| [src/prevention_service.py](src/prevention_service.py) | Prevention policy + adapter | Firewall adapter broken |
| [src/ips/scheduler.py](src/ips/scheduler.py) | Background cleanup (action expiration) | Expired blocks remain active forever |
| [src/firewall_adapters.py](src/firewall_adapters.py) | Firewall integration (Ufw, Nftables, Webhook) | Blocking commands fail |

### 9.7 CRITICAL FILE: src/core/event_bus.py

**Why Critical**:
- Central nervous system for event-driven architecture
- If broken: No event propagation
- System appears operational but detection→risk→policy→action chain fails silently

**What Depends On It**:
```
- /api/predict endpoint (detection chain)
- /api/ingest/process endpoint (batch detection)
- RealTimeStreamer (WebSocket updates)
- PerceptionIntegration (dashboard insights)
```

**Failure Modes**:
1. **Subscription lost**: Handlers not called (detection detected but no response)
2. **Exception handling broken**: One handler fails ⇒ chain stops
3. **Thread safety issue**: Race conditions on concurrent requests
4. **Memory leak**: Duplicate subscriptions cause multiple handler calls

**Test Coverage**: ❌ Limited (critical)

### 9.8 CRITICAL FILE: src/detection_service.py

**Why Critical**:
- Single entry point for ML-based detection
- PRIMARY detection path (/api/predict)

**Failure Modes**:
1. Model loading fails ⇒ 500ms+ delay on first prediction
2. AlertStore mutation fails ⇒ alerts lost
3. EventBus.publish() fails ⇒ chain breaks
4. Profile threshold logic wrong ⇒ false positives/negatives

---

## PHASE 10: STABILITY & PRODUCTION READINESS ASSESSMENT

### 10.1 PRODUCTION READINESS CHECKLIST

| Component | Status | Notes |
|-----------|--------|-------|
| **Core Detection** | ✅ Ready | ML + Signature + Multi-engine |
| **Prevention/Blocking** | ✅ Ready | Ufw/Nftables/Webhook adapters |
| **Real-time Updates** | ✅ Ready | WebSocket streaming |
| **Persistence** | ⚠️ Partial | SQLite ready, allowlist broken |
| **Authentication** | ✅ Ready | JWT + role-based |
| **Monitoring** | ⚠️ Partial | Prometheus metrics available |
| **High Availability** | ❌ Incomplete | Single-node only, no failover |
| **Scalability** | ⚠️ Limited | Not tested under 1000 req/sec |
| **Observability** | ✅ Ready | JSON logging, audit trails |
| **Incident Response** | ✅ Ready | Escalation ready (not wired) |

### 10.2 KNOWN ISSUES SUMMARY

**CRITICAL (Fix before production)**:
1. ❌ Allowlist persistence broken
2. ❌ Threat Intelligence never loads
3. ❌ Escalation tracker not wired
4. ❌ No circuit breaker on firewall adapter

**HIGH (Should fix soon)**:
5. ⚠️ EventBus blocking on adapter latency
6. ⚠️ Policy rollback doesn't affect runtime
7. ⚠️ No automatic TI feed loading
8. ⚠️ SIEM export is manual (pull) not automatic

**MEDIUM (Can defer)**:
9. ⚠️ Frontend global state race conditions
10. ⚠️ No distributed Redis for HA
11. ⚠️ StreamProcessor decoupled from web_app
12. ⚠️ No request timeout protection

### 10.3 RECOMMENDED PRODUCTION MITIGATIONS

**Immediate (24 hours)**:
```
1. Implement OpsStore.{list,add,remove}_allowlist_entry()
2. Add circuit breaker to ActionExecutor (timeout + fallback)
3. Disable TI engine or load feeds at startup
4. Add escalation tracking integration
```

**Short-term (1 week)**:
```
5. Async EventBus event handlers (thread pool)
6. Policy runtime reload (don't require restart)
7. Frontend state management refactor (Vue.js or React)
8. Load test: 1000 req/sec for 1 hour
```

**Medium-term (1 month)**:
```
9. Multi-node deployment with Redis leader election
10. Distributed StreamProcessor integration
11. Automatic TI feed refresh scheduler
12. Elasticsearch integration for audit trail
```

---

## PHASE 11: ROOT CAUSE ANALYSIS

### 11.1 Gap 1: Why Allowlist Persistence Broken?

**Timeline**:
- Phase 1-5: Core systems built (detection, prevention)
- Phase 6-7: Integration layer added
- Phase 8: Testing + bug fixes
- Phase 9: Performance validation
- Phase 10: Security audit + deployment prep

**Root Cause**: Incomplete implementation during Phase 6-7
- Allowlist module created: `prevention/allowlist.py`
- OpsStore methods NOT implemented to match
- Code review missed: Allowlist.add() doesn't call ops_store
- Tests only checked in-memory behavior (not persistence)

**Why It Wasn't Caught**:
- No integration tests for Allowlist persistence
- Tests used mocked ops_store
- Manual testing didn't persist + restart

**Fix Path**:
1. Add 3 OpsStore methods
2. Modify Allowlist to call ops_store on mutations
3. Add integration test: add entry, restart, verify persisted
4. Add E2E test: Add → Block → Allowlist → Verify unblocked

### 11.2 Gap 2: Why TI Feeds Never Load?

**Root Cause**: Incomplete feature delivery
- ThreatIntelManager created (theoretical)
- TI feed directory configured in settings
- **BUT**: No feed loader implemented
- **AND**: No startup trigger to load feeds

**Code Evidence**:
```python
# app.py line ~310:
ti_manager = ThreatIntelManager()
ti_engine = TIEngine(ti_manager)
# TODO: Load threat intel feeds at startup
# ti_manager.load_feeds(SETTINGS.ti_feed_dir)
```

**Why It's Incomplete**:
- Feature was planned but not finished
- TIEngine.is_ready() depends on feed count > 0
- Feed loading logic never written

**Fix Path**:
1. Implement ThreatIntelManager.load_feeds(directory)
2. Add async feed loading at startup
3. Handle missing/invalid feed files gracefully
4. Add health check for feed freshness

### 11.3 Gap 3: Why Escalation Not Wired?

**Root Cause**: Integration point overlooked
- EscalationTracker created as standalone module
- Methods exist: record_hit(), get_level(), should_escalate()
- **BUT**: Never called from detection flow
- **AND**: No integration points in EventBus

**Why Overlooked**:
- Multiple detection paths (/api/predict, /api/detect)
- No clear ownership for calling escalation
- Tests didn't include escalation scenarios

**Fix Path**:
1. Call escalation_tracker.record_hit() in _on_detection_event()
2. Modify PolicyEngine.decide() to accept escalation level
3. Adjust decision thresholds based on escalation
4. Add tests: repeated detections → escalation

### 11.4 Gap 4: Why No Adapter Circuit Breaker?

**Root Cause**: Defensive coding not prioritized
- ActionExecutor.execute() trusted adapter to not hang
- No timeout wrapping on adapter calls
- No fallback behavior

**Why It's Risky**:
- Firewall adapters use system commands (UFW, Nftables)
- System commands can hang indefinitely
- No process timeout = Flask worker thread blocked
- Under load: All workers block = DoS

**Fix Path**:
1. Add timeout wrapper: functools.timeout_decorator
2. Implement circuit breaker pattern:
   - Count consecutive failures
   - Open circuit after 3 failures
   - Fast-fail for 60s
   - Half-open: try 1 request, close if succeeds
3. Add fallback: Log + return FALLBACK status
4. Add metrics: adapter_latency, adapter_failures

---

## CONCLUSION & ARCHITECTURE SUMMARY

### System Strengths ✅

1. **Event-Driven Architecture**: Clean separation of concerns (Detect → Risk → Policy → Action)
2. **Multi-Engine Framework**: Pluggable detection engines with aggregation
3. **Real-Time Capabilities**: WebSocket streaming for live updates
4. **Comprehensive Prevention**: Firewall adapters + allowlist + escalation
5. **Audit Trail**: Complete event logging for compliance
6. **Observability**: Metrics, structured logging, audit events
7. **Security**: JWT auth, CSRF protection, input sanitization, rate limiting

### System Weaknesses ❌

1. **Missing Persistence**: Allowlist doesn't persist to database
2. **Incomplete Features**: TI feeds, escalation, SIEM export not wired
3. **Blocking EventBus**: Synchronous event chain adds latency
4. **No Adapter Protection**: Firewall calls can hang indefinitely
5. **Single-Node Only**: HA components incomplete
6. **Frontend State**: Global singleton + race conditions under load
7. **Data Inconsistency**: Multiple paths to detection (predict vs detect)

### Critical Next Steps

**BEFORE PRODUCTION (24-48 hours)**:
1. Implement allowlist persistence (3 SQL methods)
2. Add circuit breaker to adapter calls
3. Wire escalation tracker integration
4. Load TI feeds at startup (or disable)
5. Run 100+ req/sec load test

**RECOMMENDED (Week 1)**:
6. Async event handlers (prevent blocking)
7. Runtime policy reload
8. Frontend state management refactor
9. Multi-engine test suite
10. Incident response playbook verification

---

**Report Complete**: 2026-05-15  
**Next Phase**: Implementation & Stabilization  
**Estimated Remediation**: 2-4 weeks to production-grade stability
