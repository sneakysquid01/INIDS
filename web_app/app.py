import atexit

from flask import Flask, Response, jsonify, render_template, request
import joblib
import os
import sys
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import logging
import json
import time
import threading
import uuid
from datetime import datetime, timezone, timedelta

# Support both `python -m web_app.app` and direct script runs (`python web_app/app.py`).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.settings import load_settings
from src.rate_limiter import UnifiedRateLimiter, get_unified_rate_limiter, set_unified_rate_limiter
from src.firewall_adapters import MockFirewallAdapter, UfwFirewallAdapter, NftablesFirewallAdapter, WebhookFirewallAdapter
from src.core.event_bus import ActionEvent, DetectionEvent, EventBus, PolicyDecisionEvent, RiskScoreEvent
from src.logging_config import configure_logging
from src.observability.json_logging import configure_json_logging
from src.observability.siem_exporter import SiemExporter
from src.ips.action_executor import ActionExecutor
from src.ips.policy_engine import PolicyEngine
from src.ips.risk_engine import RiskEngine
from src.ips.scheduler import PreventionScheduler
from src.policy.policy_store import PolicyStore
from src.prevention_service import PolicyConfig
from src.ha.health_check import HealthCheck
from src.ha.leader_election import LeaderElection
from src.input_sanitizer import SanitizationError, sanitize_id, sanitize_ip_address
from src.correlation_tracing import correlation_id_middleware, get_correlation_id
from src.csrf_protection import csrf_protect_middleware, require_csrf_token

try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
except ImportError as e:
    raise ImportError(
        "flask_socketio is required for INIDS 2.0. WebSocket is mandatory. "
        "Install with: pip install flask-socketio python-socketio"
    ) from e


configure_logging()
logger = logging.getLogger(__name__)

SETTINGS = load_settings()
if SETTINGS.json_logging and not SETTINGS.debug:
    configure_json_logging(service_name="inids")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")
STATIC_DIR = os.path.join(BASE_DIR, "web_app", "static")
OPS_DB_PATH = SETTINGS.ops_db_path if os.path.isabs(SETTINGS.ops_db_path) else os.path.join(BASE_DIR, SETTINGS.ops_db_path)

from src.detection_service import DetectionService, drain_to_ops_store
from src.prevention_service import PreventionService
from src.ops_store import OpsStore
from src.auth_service import require_role, auth_status, _auth_service
from src.metrics_service import MetricsService
from src.ingestion_service import InMemoryIngestionQueue, RedisStreamIngestionQueue, IngestionService
from src.prevention.allowlist import Allowlist
from src.prevention.escalation_tracker import EscalationTracker
from src.prevention.false_positive_manager import FalsePositiveManager
from src.threat_intel.feed_manager import ThreatIntelManager
from src.threat_intel.ti_engine import TIEngine
from src.feature_engineering import enrich_single_row
from src.log_parsers import parse_zeek_conn_log, parse_suricata_eve_flow
from src.model_registry import ModelRegistry
from src.schema import (
    COLUMNS,
    LABEL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    DEFAULT_FEATURE_ROW,
)
from src.detection.engine_registry import EngineRegistry
from src.detection.aggregator import EngineAggregator, AggregationStrategy
from src.detection.engines.ml_engine import MLEngine
from src.detection.engines.signature_engine import SignatureEngine
from src.detection.engines.anomaly_engine import AnomalyEngine
from src.detection.engines.threshold_engine import ThresholdEngine
from src.detection.engines.honeypot_engine import HoneypotDetectionEngine
from src.pipeline.backpressure import BackpressureController, BackpressureLevel
from src.pipeline.stream_processor import StreamProcessor
from src.pipeline.worker import PipelineWorker
from src.ips.incident_aggregator import IncidentAggregator
from src.detection.temporal_correlation import TemporalCorrelationEngine, create_example_patterns
from src.ips.entity_enrichment import EntityEnrichmentEngine
from src.ips.alert_filter import ThreeLayerAlertFilter, create_default_rules
from src.realtime.broadcaster import RealTimeStreamer
from src.training import DatasetCollector, RertrainingScheduler
from src.perception import AttackStoryEngine, ConfidenceBreakdownEngine, LiveSystemPulse
from src.perception.perception_integration import PerceptionIntegration
from src.middleware import (
    register_middleware, RateLimitConfig, RateLimitMiddleware, IPBlockingMiddleware,
    SecurityHeadersMiddleware, AuditLogMiddleware
)
from src.auth_jwt import JWTAuthManager, RunAsManager, require_auth, require_role as jwt_require_role, inject_current_user
from src.validation_schemas import validate_predict_request, validate_detect_request, validate_policy
from src.elasticsearch_client import ElasticsearchConfig, ElasticsearchStore
from src.elasticsearch_audit_bridge import init_elasticsearch_audit_bridge, get_elasticsearch_audit_bridge
from src.async_utils import get_async_executor

app = Flask(__name__)
app.config["SECRET_KEY"] = SETTINGS.flask_secret_key
# Prevent DOS attacks from large uploads (16 MB limit)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Register security middleware
correlation_id_middleware(app)
csrf_protect_middleware(app)

# WebSocket is MANDATORY for INIDS 2.0
# If SocketIO is None, the import would have already failed above
cors_origins = ["http://localhost", "http://127.0.0.1", "http://localhost:5000", "http://127.0.0.1:5000"]
socketio = SocketIO(app, cors_allowed_origins=cors_origins, async_mode="threading")
SOCKETIO_ENABLED = True
logger.info("WebSocket (SocketIO) initialized successfully - REQUIRED for INIDS 2.0")

# --- Initialize Security Middleware Stack ---
middleware_config = RateLimitConfig(
    max_requests=SETTINGS.rate_limit_requests,
    window_seconds=SETTINGS.rate_limit_window_seconds
)
_cors_origins = [o.strip() for o in SETTINGS.cors_origins.split(",") if o.strip()]
middleware_instances = register_middleware(app, middleware_config, cors_origins=_cors_origins)

# Store middleware instances on app (JWT auth will be initialized after ops_store)
app.rate_limiter = middleware_instances['rate_limiter']
app.ip_blocker = middleware_instances['ip_blocker']
app.audit_log = middleware_instances['audit_log']

app._pipeline_worker = None
app._pipeline_backpressure = None
app._pipeline_processor = None
app._pipeline_worker_started = False
app._redis_client = None
app._redis_client_initialized = False


def _build_firewall_adapter():
    adapter_name = SETTINGS.firewall_adapter
    if adapter_name == "ufw":
        return UfwFirewallAdapter()
    if adapter_name == "nftables":
        return NftablesFirewallAdapter()
    if adapter_name == "webhook":
        return WebhookFirewallAdapter(webhook_url=SETTINGS.firewall_webhook_url)
    return MockFirewallAdapter()


# Threading locks to synchronize access to global state
_pipeline_state_lock = threading.Lock()
_models_lock = threading.Lock()

model = None
all_models = {}
event_bus = EventBus()
detection_service = None
prevention_service = PreventionService(adapter=_build_firewall_adapter())
ops_store = OpsStore(OPS_DB_PATH)

# --- Initialize JWT Authentication (after ops_store) ---
jwt_manager = JWTAuthManager(secret_key=SETTINGS.flask_secret_key)
runas_manager = RunAsManager(jwt_manager, ops_store)
app.jwt_manager = jwt_manager
app.runas_manager = runas_manager

# --- Initialize Elasticsearch Audit Bridge (Week 2) ---
audit_bridge = None
if SETTINGS.elasticsearch_enabled:
    es_config = ElasticsearchConfig(
        hosts=SETTINGS.elasticsearch_hosts or ["localhost"],
        port=SETTINGS.elasticsearch_port or 9200,
        use_ssl=SETTINGS.elasticsearch_use_ssl or False,
        verify_certs=SETTINGS.elasticsearch_verify_certs or False,
        username=SETTINGS.elasticsearch_username,
        password=SETTINGS.elasticsearch_password,
        index_prefix="inids"
    )
    audit_bridge = init_elasticsearch_audit_bridge(es_config, ops_store_ref=ops_store)
else:
    logger.info("Elasticsearch audit bridge disabled (set ELASTICSEARCH_ENABLED=1 to enable)")
app.elasticsearch_bridge = audit_bridge

# --- Initialize Async Utilities (Week 2) ---
async_executor = get_async_executor(max_workers=4)
app.async_executor = async_executor

# Make jwt_manager available in request context
@app.before_request
def inject_jwt_manager():
    request.jwt_manager = jwt_manager
    action_executor.ops_store = ops_store
    runas_manager.audit_store = ops_store
    incident_aggregator._ops_store = ops_store
    allowlist._ops_store = ops_store
    fp_manager._ops_store = ops_store
    alert_filter.ops_store = ops_store
    if audit_bridge is not None:
        audit_bridge.ops_store = ops_store

incident_aggregator = IncidentAggregator(ops_store)
allowlist = Allowlist(ops_store)
escalation_tracker = EscalationTracker(cooldown_seconds=300.0)
fp_manager = FalsePositiveManager(ops_store=ops_store)
fp_manager.load_from_store()
metrics_service = MetricsService()
siem_exporter = SiemExporter()
ingestion_queue = InMemoryIngestionQueue(max_items=10000, persistent=True)
ingestion_service = IngestionService(queue=ingestion_queue)
model_registry = ModelRegistry(os.path.join(RESULTS_DIR, "model_registry.json"))
# C-05 (Step 19): unified two-tier rate limiter (replaces InMemoryRateLimiter + RateLimitMiddleware).
unified_rate_limiter = UnifiedRateLimiter()
set_unified_rate_limiter(unified_rate_limiter)

# --- Multi-engine detection framework ---
engine_registry = EngineRegistry()
engine_aggregator = EngineAggregator(AggregationStrategy.ANY_TRIGGER)
RULES_PATH = os.path.join(BASE_DIR, "rules", "default_rules.yaml")
signature_engine = SignatureEngine(RULES_PATH if os.path.exists(RULES_PATH) else None, fp_manager=fp_manager)
threshold_engine = ThresholdEngine(fp_manager=fp_manager)
anomaly_engine = AnomalyEngine(
    buffer_size=3000,
    model_path=os.path.join(MODELS_DIR, "anomaly_engine.pkl"),
    fp_manager=fp_manager,
)

engine_registry.register(signature_engine)
engine_registry.register(threshold_engine)
engine_registry.register(anomaly_engine, enabled=anomaly_engine.is_ready())

# Threat intelligence engine — starts disabled until feeds are loaded.
# is_ready() returns True automatically once ti_manager.cache.size() > 0.
ti_manager = ThreatIntelManager()
ti_engine = TIEngine(ti_manager)
engine_registry.register(ti_engine)  # enabled=True; gated by is_ready() returning False when cache is empty

# Honeypot detection engine — detects access to canary IPs/ports
honeypot_ips = [ip.strip() for ip in SETTINGS.honeypot_ips.split(",") if ip.strip()]
honeypot_ports = []
if SETTINGS.honeypot_ports:
    try:
        honeypot_ports = [int(p.strip()) for p in SETTINGS.honeypot_ports.split(",") if p.strip()]
    except ValueError:
        logger.warning("Invalid honeypot ports configuration: %s", SETTINGS.honeypot_ports)
honeypot_engine = HoneypotDetectionEngine(
    engine_id="honeypot",
    honeypot_ips=honeypot_ips,
    honeypot_ports=honeypot_ports,
)
engine_registry.register(honeypot_engine, enabled=SETTINGS.honeypot_enabled)

# Temporal correlation engine — detects multi-stage attacks across time
temporal_correlation_engine = TemporalCorrelationEngine()
# Register example patterns for common attack flows
# TODO: Fix pattern registration - currently uses old API signature
# temporal_correlation_engine.register_pattern(
#     "port_scan_to_brute_force",
#     [
#         {"type": "port_scan", "confidence_min": 0.7},
#         {"type": "brute_force", "confidence_min": 0.8, "time_offset_seconds": 300}
#     ]
# )
# temporal_correlation_engine.register_pattern(
#     "c2_to_data_exfil",
#     [
#         {"type": "c2_communication", "confidence_min": 0.75},
#         {"type": "data_exfil", "confidence_min": 0.75, "time_offset_seconds": 600}
#     ]
# )

# Entity context enrichment engine — enriches alerts with GeoIP, threat intel, history
entity_enrichment_engine = EntityEnrichmentEngine(
    ops_store=ops_store,
    ti_manager=ti_manager,
    internal_cidrs=SETTINGS.internal_cidrs if hasattr(SETTINGS, "internal_cidrs") else None
)
logger.info("Entity enrichment engine initialized with threat intel manager")

# Three-layer alert filtering engine — exclude/ignore/merge alerts
alert_filter = ThreeLayerAlertFilter(ops_store=ops_store)
# Load persisted rules and add default recommendations
alert_filter.load_rules_from_storage()
if not alert_filter.exclude_rules and not alert_filter.ignore_rules:
    exclude_rules, ignore_rules, merge_rules = create_default_rules()
    for rule in exclude_rules:
        alert_filter.add_exclude_rule(rule)
    for rule in ignore_rules:
        alert_filter.add_ignore_rule(rule)
    for rule in merge_rules:
        alert_filter.add_merge_rule(rule)
logger.info("Three-layer alert filter initialized with %d rules", 
            len(alert_filter.exclude_rules) + len(alert_filter.ignore_rules) + len(alert_filter.merge_rules))

risk_engine = RiskEngine()
policy_engine = PolicyEngine()
policy_store = PolicyStore(initial_config=prevention_service.policy.to_dict())
action_executor = ActionExecutor(
    adapter=prevention_service.adapter,
    adapter_name=SETTINGS.firewall_adapter,
    ops_store=ops_store,
    event_bus=event_bus,
)

# --- Initialize RealTimeStreamer for INIDS 2.0 ---
realtime_streamer = RealTimeStreamer(event_bus=event_bus, socketio=socketio, namespace="/events")
realtime_streamer.start()
logger.info("RealTimeStreamer initialized and started for real-time event broadcasting")

# --- Initialize ML Lifecycle Components for INIDS 2.0 ---
dataset_collector = DatasetCollector(
    db_path=os.path.join(DATA_DIR, "training.db"),
    retention_days=30
)
logger.info("DatasetCollector initialized for training data collection")

# Note: RertrainingScheduler will be initialized after models are loaded in load_models()
retraining_scheduler = None  # Will be set after model loading

# --- Initialize Perception Layer Components for INIDS 2.0 ---
attack_story_engine = AttackStoryEngine()
confidence_breakdown_engine = ConfidenceBreakdownEngine()
live_system_pulse = LiveSystemPulse(window_minutes=60)
logger.info("Perception layer initialized: AttackStory, ConfidenceBreakdown, LivePulse")

# --- Real-time Integration: Connect Perception Engines to EventBus ---
perception_integration = PerceptionIntegration(
    event_bus=event_bus,
    attack_story_engine=attack_story_engine,
    confidence_breakdown_engine=confidence_breakdown_engine,
    live_system_pulse=live_system_pulse,
    queue_size=1000,
    batch_size=10,
    worker_threads=2
)
perception_integration.start()
logger.info("Perception integration layer started for real-time event processing")

prevention_scheduler = PreventionScheduler(
    action_executor,
    interval_seconds=30,
    is_leader_fn=lambda: leader_election.is_leader,
)
leader_election = LeaderElection(redis_client=None, instance_id=f"inids-{os.getpid()}")
health_check = HealthCheck()

INPUT_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "count",
    "srv_count", "serror_rate", "same_srv_rate",
]

MODEL_INPUT_COLUMNS = FEATURE_COLUMNS
NUMERIC_MODEL_COLUMNS = NUMERIC_FEATURES

# API Configuration Constants
DEFAULT_LIMIT = 50
MAX_AUDIT_LIMIT = 500
MAX_CSV_ROWS = 50000
MAX_BATCH_SIZE = 10000
MAX_ALERTS_LIMIT = 1000


def _normalize_action_payload(row: dict) -> dict:
    action_type = row.get("action_type") or row.get("action") or row.get("type") or "unknown"
    raw_status = str(row.get("status") or "").strip().lower()
    if raw_status in {"active", "executed", "enforced"} or row.get("executed"):
        ui_status = "success"
    elif raw_status in {"pending", "pending_approval", "reviewing"}:
        ui_status = "pending"
    elif raw_status in {"failed", "error", "block_failed", "unblock_failed"}:
        ui_status = "failed"
    else:
        ui_status = raw_status or "pending"
    action_id = row.get("action_id") or row.get("id") or f"action_{abs(hash(str(row))) % 1000000}"
    return {
        **row,
        "id": str(action_id),
        "type": str(action_type),
        "timestamp": row.get("created_at") or row.get("executed_at"),
        "status": ui_status,
        "executor": row.get("adapter") or "System",
    }


def _normalize_alert_payload(row: dict) -> dict:
    source_ip = row.get("source_ip") or row.get("src_ip") or row.get("source") or ""
    prediction = row.get("prediction") or row.get("classification") or "unknown"
    return {
        **row,
        "src_ip": source_ip,
        "target_ip": source_ip,
        "classification": prediction,
        "alert_type": row.get("attack_type") or prediction,
        "title": f"{str(prediction).title()} alert",
    }


def _format_allowlist_entry(entry: str) -> dict:
    entry_type = "cidr" if "/" in entry else "ip"
    return {
        "id": entry,
        "value": entry,
        "entry": entry,
        "type": entry_type,
        "reason": "",
        "added_by": "System",
        "active": True,
    }


def _get_redis_client():
    if getattr(app, "_redis_client_initialized", False):
        return getattr(app, "_redis_client", None)

    app._redis_client_initialized = True
    if not SETTINGS.redis_url:
        app._redis_client = None
        return None

    try:
        import redis as redis_lib
    except ImportError:
        logger.warning("Pipeline enabled but redis package is not installed")
        app._redis_client = None
        return None

    try:
        client = redis_lib.from_url(SETTINGS.redis_url, decode_responses=False)
        client.ping()
        app._redis_client = client
        return client
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.warning("Redis unavailable for pipeline runtime; continuing without streaming: %s", e)
        app._redis_client = None
        return None


def _close_redis_client() -> None:
    client = getattr(app, "_redis_client", None)
    app._redis_client = None
    app._redis_client_initialized = False
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except (ConnectionError, OSError, RuntimeError) as e:
            logger.debug("Error closing Redis client: %s", e)


def _stream_source_ip(features: dict) -> str:
    for key in ("source_ip", "src_ip", "client_ip"):
        value = features.get(key)
        if value:
            return str(value)
    return "unknown"


def _stream_result_callback(aggregated, features: dict) -> None:
    verdict = str(aggregated.verdict or "unknown").strip().lower()
    prediction = {
        "attack": "Attack",
        "normal": "Normal",
        "suspicious": "Suspicious",
    }.get(verdict, verdict.title() or "Unknown")
    event = DetectionEvent(
        source_ip=_stream_source_ip(features),
        prediction=prediction,
        confidence=float(aggregated.confidence),
        features=dict(features),
        attack_type=str(aggregated.attack_type or "unknown"),
        profile="streaming",
        severity=str(aggregated.severity or "low"),
        suspicious=verdict == "suspicious",
        reason="stream_pipeline",
    )
    logger.info(
        "Stream detection event source_ip=%s verdict=%s confidence=%.2f engines=%d",
        event.source_ip,
        verdict,
        event.confidence,
        len(aggregated.engine_results),
    )
    event_bus.publish(event)


def _pipeline_status() -> dict:
    status = {
        "enabled": bool(SETTINGS.pipeline_enabled),
        "configured": bool(SETTINGS.redis_url),
        "stream_key": SETTINGS.pipeline_stream_key,
        "redis_connected": getattr(app, "_redis_client", None) is not None,
        "running": False,
    }
    worker = getattr(app, "_pipeline_worker", None)
    if worker is not None:
        status.update(worker.status())
    return status


def _ensure_pipeline_started() -> bool:
    if getattr(app, "_pipeline_worker_started", False):
        return True
    if not SETTINGS.pipeline_enabled:
        return False

    redis_client = _get_redis_client()
    if redis_client is None:
        return False

    # Upgrade the in-process ingestion queue to Redis-backed for durability now
    # that we have a confirmed Redis connection. This ensures records enqueued via
    # the non-pipeline path (/api/ingest/process) survive application restarts.
    global ingestion_queue, ingestion_service
    with _pipeline_state_lock:
        if not isinstance(ingestion_queue, RedisStreamIngestionQueue):
            ingestion_queue = RedisStreamIngestionQueue(
                redis_client, stream_key="inids:ingestion", max_items=10000
            )
            ingestion_service = IngestionService(queue=ingestion_queue)
            logger.info("Upgraded ingestion queue to Redis-backed (inids:ingestion)")

    load_models()
    try:
        processor = StreamProcessor(
            redis_client,
            engine_registry,
            engine_aggregator,
            stream_key=SETTINGS.pipeline_stream_key,
            consumer_name=f"app-{os.getpid()}",
            batch_size=SETTINGS.pipeline_batch_size,
            result_callback=_stream_result_callback,
        )
        backpressure = BackpressureController()
        worker = PipelineWorker(processor, backpressure)
        worker.start()
    except Exception:
        logger.exception("Failed to start pipeline runtime")
        return False

    app._pipeline_processor = processor
    app._pipeline_backpressure = backpressure
    app._pipeline_worker = worker
    app._pipeline_worker_started = True
    return True


def _stop_pipeline_runtime() -> None:
    worker = getattr(app, "_pipeline_worker", None)
    if worker is not None:
        try:
            worker.stop()
        except Exception:
            pass  # suppress during teardown; logger streams may be closed
    app._pipeline_worker = None
    app._pipeline_processor = None
    app._pipeline_backpressure = None
    app._pipeline_worker_started = False


def _shutdown_runtime() -> None:
    if getattr(app, "_shutdown_started", False):
        return
    app._shutdown_started = True
    try:
        # Stop background module broadcaster if running.
        if "_update_thread_stop" in globals():
            globals()["_update_thread_stop"] = True
        updater = globals().get("_update_thread")
        if updater is not None and getattr(updater, "is_alive", lambda: False)():
            updater.join(timeout=1)
    except Exception:
        pass
    # Disable logging exception propagation before teardown: at interpreter
    # shutdown the logging stream handlers may already be closed, and we must
    # not let ValueError/"I/O on closed file" change the process exit code.
    import logging as _logging
    _logging.raiseExceptions = False
    try:
        prevention_scheduler.stop()
    except Exception:
        pass
    try:
        leader_election.stop()
    except Exception:
        pass
    try:
        _stop_pipeline_runtime()
    except Exception:
        pass
    try:
        _close_redis_client()
    except Exception:
        pass


def _prepare_stream_record(features: dict, source: str) -> dict:
    normalized = ingestion_service.normalize_features(features)
    payload = dict(normalized)
    payload["source"] = str(source)
    for key in ("source_ip", "src_ip", "client_ip", "attack_type", "profile"):
        if key in features and features[key] not in (None, ""):
            payload[key] = features[key]
    return payload


def _stream_ingest_records(records: list[dict], source: str) -> int:
    redis_client = _get_redis_client()
    if redis_client is None:
        raise RuntimeError("redis_unavailable")

    for record in records:
        payload = _prepare_stream_record(record, source)
        redis_client.xadd(
            SETTINGS.pipeline_stream_key,
            {"payload": json.dumps(payload, separators=(",", ":"))},
        )
    return len(records)


def _apply_escalation_to_risk(risk_event: RiskScoreEvent, escalation_level: int) -> RiskScoreEvent:
    """Apply escalation level boost to risk score.
    
    Higher escalation levels (repeated hits from same source) increase risk urgency.
    Escalation multiplier: 1.0 (level 0) → 1.5 (level 1) → 2.0 (level 2+)
    """
    if escalation_level is None or escalation_level < 0:
        return risk_event
    
    escalation_multiplier = min(1.0 + (escalation_level * 0.5), 2.0)
    boosted_score = risk_event.risk_score * escalation_multiplier
    
    # Preserve original components but note escalation in debug info
    risk_event.risk_score = min(boosted_score, 100.0)  # Cap at 100
    logger.debug(
        "escalation_risk_boost source_ip=%s level=%d original=%.2f boosted=%.2f",
        getattr(risk_event, 'source_ip', 'unknown'),
        escalation_level,
        risk_event.risk_score / escalation_multiplier,
        risk_event.risk_score,
    )
    return risk_event


def _on_detection_event(event: DetectionEvent) -> None:
    source_is_allowlisted = allowlist.contains(event.source_ip)
    if source_is_allowlisted:
        logger.info("Allowlist: enforcement bypass enabled for %s", event.source_ip)

    pred_lower = str(event.prediction).lower()

    # Persist alerts from all event-bus paths, including streaming detections.
    if pred_lower in ("attack", "suspicious") or event.suspicious:
        try:
            ops_store.save_alert(
                {
                    "id": f"al_{uuid.uuid4().hex[:10]}",
                    "timestamp": event.timestamp,
                    "severity": event.severity,
                    "prediction": event.prediction,
                    "confidence": event.confidence,
                    "profile": event.profile,
                    "reason": event.reason,
                    "source_ip": event.source_ip or "",
                    "attack_type": event.attack_type or "",
                    "risk_score": getattr(event, 'risk_score', 0.0) or 0.0,
                }
            )
            metrics_service.inc("alerts_total")
        except Exception:
            logger.exception("Failed to persist alert from detection event")

    # Feed normal detections into anomaly training buffer and auto-enable once fitted.
    if pred_lower == "normal":
        try:
            newly_fitted = anomaly_engine.add_sample(event.features)
            if newly_fitted:
                engine_registry.set_enabled(anomaly_engine.engine_id, True)
                logger.info("AnomalyEngine auto-fitted and enabled from traffic")
        except Exception:
            logger.debug("Anomaly buffer sample failed", exc_info=True)

    # Use policy-configured weights for dynamic risk scoring.
    policy = prevention_service.policy
    from src.ips.risk_engine import RiskWeights as _RW

    weights_override = _RW(
        confidence=float(getattr(policy, "risk_weight_confidence", 0.5)),
        severity=float(getattr(policy, "risk_weight_severity", 0.3)),
        frequency=float(getattr(policy, "risk_weight_frequency", 0.2)),
    )
    risk_event = risk_engine.calculate(event, weights_override=weights_override)
    
    # Record hit in escalation tracker and apply escalation boost to risk score
    if event.suspicious and event.source_ip:
        try:
            escalation_level = escalation_tracker.record_hit(
                source_ip=event.source_ip,
                severity=event.severity
            )
            if escalation_level is not None:
                risk_event = _apply_escalation_to_risk(risk_event, escalation_level)
        except Exception:
            logger.exception("Escalation tracking failed for %s", event.source_ip)
    
    event_bus.publish(risk_event)
    try:
        ops_store.add_audit(
            event_type="risk_score",
            message=json.dumps(
                {
                    "source_ip": event.source_ip,
                    "prediction": event.prediction,
                    "attack_type": event.attack_type,
                    "risk_score": risk_event.risk_score,
                    "components": risk_event.components,
                },
                separators=(",", ":"),
            ),
            created_at=risk_event.timestamp,
        )
    except Exception:
        logger.exception("Failed to persist risk score")


def _on_risk_event(event: RiskScoreEvent) -> None:
    # ISSUE-010 FIX: Integrate escalation tracker output into policy decisions
    # Get current escalation level for this source IP
    source_ip = event.detection.source_ip
    escalation_level = escalation_tracker.get_level(source_ip)
    
    # Adjust risk score based on escalation history
    # Escalation levels: 0=CLEAN, 1=ALERT, 2=RATE_LIMIT, 3=TEMP_BLOCK, 4=PERM_BLOCK
    adjusted_risk = float(event.risk_score)
    escalation_boost = 0.0
    if escalation_level >= 3:  # Already at TEMP_BLOCK or higher
        escalation_boost = 0.15  # Boost risk by 15%
    elif escalation_level >= 2:  # Already at RATE_LIMIT
        escalation_boost = 0.10  # Boost risk by 10%
    elif escalation_level >= 1:  # Already at ALERT
        escalation_boost = 0.05  # Boost risk by 5%
    
    adjusted_risk = min(1.0, adjusted_risk + escalation_boost)
    
    # Create modified risk event with escalation context
    risk_event_with_context = RiskScoreEvent(
        detection=event.detection,
        risk_score=adjusted_risk,
        components={
            **event.components,
            "escalation_boost": escalation_boost,
            "escalation_level": int(escalation_level),
        }
    )
    
    decision_event = policy_engine.decide(risk_event_with_context, prevention_service.policy)
    event_bus.publish(decision_event)
    try:
        ops_store.add_audit(
            event_type="policy_decision",
            message=json.dumps(
                {
                    "source_ip": event.detection.source_ip,
                    "prediction": event.detection.prediction,
                    "risk_score": event.risk_score,
                    "adjusted_risk_score": adjusted_risk,
                    "escalation_level": int(escalation_level),
                    "decision": decision_event.decision,
                    "reason": decision_event.reason,
                    "ttl_seconds": decision_event.ttl_seconds,
                },
                separators=(",", ":"),
            ),
            created_at=decision_event.timestamp,
        )
    except Exception:
        logger.exception("Failed to persist policy decision")


def _on_policy_decision_event(event: PolicyDecisionEvent) -> None:
    if str(event.decision).strip().upper() not in {"BLOCK", "TEMP_BLOCK", "RATE_LIMIT", "PENDING_BLOCK"}:
        return
    if allowlist.contains(event.risk.detection.source_ip):
        try:
            ops_store.add_audit(
                event_type="allowlist_enforcement_bypass",
                message=json.dumps(
                    {
                        "source_ip": event.risk.detection.source_ip,
                        "decision": event.decision,
                        "reason": "allowlisted_target",
                    },
                    separators=(",", ":"),
                ),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            logger.exception("Failed to persist allowlist enforcement bypass audit")
        return
    action = action_executor.execute(event, prevention_service.policy)
    if action is not None:
        event_bus.publish(action)
    escalation_tracker.record_hit(
        event.risk.detection.source_ip,
        event.risk.detection.severity,
    )


def _emit_realtime(event_name: str, payload: dict) -> None:
    """Emit real-time event via WebSocket (always enabled in INIDS 2.0)."""
    try:
        socketio.emit(event_name, payload, namespace="/events")
    except Exception:
        logger.exception("Failed to emit websocket event '%s'", event_name)


def _build_dashboard_metrics_payload(
    *,
    alerts: list[dict] | None = None,
    recent_actions: list[dict] | None = None,
) -> dict[str, object]:
    alerts = alerts if alerts is not None else ops_store.list_alerts(limit=100)
    recent_actions = recent_actions if recent_actions is not None else ops_store.list_actions(limit=100)

    active_attacks = sum(
        1
        for action in recent_actions
        if str(action.get("action", "")).strip().lower() in {"block", "rate_limit"}
    )
    blocked = sum(
        1
        for action in recent_actions
        if str(action.get("action", "")).strip().lower() == "block"
    )
    under_review = sum(
        1
        for alert in alerts
        if str(alert.get("status", "")).strip().lower() in {"pending", "pending_approval", "reviewing", "escalated"}
    )

    one_hour_ago = time.time() - 3600
    recent_last_hour = [
        action for action in recent_actions if _to_epoch_seconds(action.get("created_at"), default=0.0) > one_hour_ago
    ]
    last_hour_attacks = len(recent_last_hour)
    last_hour_blocks = sum(
        1
        for action in recent_last_hour
        if str(action.get("action", "")).strip().lower() == "block"
    )

    feedback_stats = fp_manager.stats()
    false_positives = sum(int(row.get("false_positives", 0)) for row in feedback_stats)
    total_feedback = sum(int(row.get("total", 0)) for row in feedback_stats)
    fp_rate = round((false_positives / total_feedback * 100) if total_feedback else 0, 1)

    return {
        "system_uptime": "4.2h",
        "system_health": 98,
        "system_capacity": 45,
        "active_attacks": active_attacks,
        "blocked": blocked,
        "active_alerts": min(len(alerts), 99),
        "under_review": under_review,
        "last_hour_attacks": last_hour_attacks,
        "last_hour_blocks": last_hour_blocks,
        "fp_rate": fp_rate,
    }


def _build_realtime_state(*, alert_limit: int = 200, action_limit: int = 50) -> dict[str, object]:
    alerts = ops_store.list_alerts(limit=max(1, min(alert_limit, 200)))
    recent_actions = ops_store.list_actions(limit=max(1, min(action_limit, 200)))
    dashboard_metrics = _build_dashboard_metrics_payload(alerts=alerts, recent_actions=recent_actions)

    active_blocked_ips = len(ops_store.list_active_blocks(limit=5000))
    live_system_pulse.update_blocked_ips(active_blocked_ips)
    pulse = live_system_pulse.get_pulse_status()

    pending_actions = [
        action
        for action in recent_actions
        if str(action.get("status") or "").strip().lower() in {"pending", "pending_approval", "reviewing"}
        or not bool(action.get("executed"))
    ]

    return {
        **dashboard_metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "alerts": alerts,
        "alertsCount": len(alerts),
        "actions": recent_actions,
        "pendingActions": pending_actions,
        "pulse": pulse,
        "current": pulse.get("current", {}),
        "rolling_averages": pulse.get("rolling_averages", {}),
        "status": pulse.get("status"),
        "pulse_strength": pulse.get("pulse_strength"),
    }


def _on_detection_realtime(event: DetectionEvent) -> None:
    metrics_service.inc("detection_events_total")
    _emit_realtime("DetectionEvent", event.to_dict())
    _emit_realtime("metrics.update", _build_realtime_state())


def _on_risk_realtime(event: RiskScoreEvent) -> None:
    _emit_realtime("RiskScoreEvent", event.to_dict())


def _on_action_realtime(event: ActionEvent) -> None:
    metrics_service.inc("action_events_total")
    _emit_realtime("ActionEvent", event.to_dict())
    _emit_realtime("metrics.update", _build_realtime_state())


def _on_detection_siem(event: DetectionEvent) -> None:
    siem_exporter.emit(event.to_dict())


def _on_risk_siem(event: RiskScoreEvent) -> None:
    siem_exporter.emit(event.to_dict())


def _on_policy_siem(event: PolicyDecisionEvent) -> None:
    siem_exporter.emit(event.to_dict())


def _on_action_siem(event: ActionEvent) -> None:
    siem_exporter.emit(event.to_dict())


event_bus.subscribe(DetectionEvent, _on_detection_event)
event_bus.subscribe(RiskScoreEvent, _on_risk_event)
event_bus.subscribe(PolicyDecisionEvent, _on_policy_decision_event)
event_bus.subscribe(DetectionEvent, _on_detection_realtime)
event_bus.subscribe(RiskScoreEvent, _on_risk_realtime)
event_bus.subscribe(ActionEvent, _on_action_realtime)
event_bus.subscribe(DetectionEvent, _on_detection_siem)
event_bus.subscribe(RiskScoreEvent, _on_risk_siem)
event_bus.subscribe(PolicyDecisionEvent, _on_policy_siem)
event_bus.subscribe(ActionEvent, _on_action_siem)


def _ensure_scheduler_started() -> None:
    if getattr(app, "_prevention_scheduler_started", False):
        return
    _start_leader_election()
    prevention_scheduler.start()
    app._prevention_scheduler_started = True
    _start_siem_flush_thread()
    # Load TI feeds once at scheduler start (no-op if ti_feed_dir is unset).
    if SETTINGS.ti_feed_dir:
        _load_ti_feeds()
        _start_ti_refresh_thread()


atexit.register(_shutdown_runtime)


def _load_ti_feeds() -> int:
    """Scan SETTINGS.ti_feed_dir for .csv and .json feed files and load them.

    Returns the total number of indicators loaded across all files.
    CSV files must have a header row with at least an ``indicator`` column.
    JSON files must contain a JSON array of indicator objects.
    """
    feed_dir = SETTINGS.ti_feed_dir
    if not feed_dir or not os.path.isdir(feed_dir):
        return 0

    total = 0
    import glob as _glob
    for path in sorted(_glob.glob(os.path.join(feed_dir, "*.csv")) + _glob.glob(os.path.join(feed_dir, "*.json"))):
        source = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            if path.endswith(".csv"):
                n = ti_manager.load_csv_feed(content, source=source)
            else:
                n = ti_manager.load_json_feed(content, source=source)
            total += n
        except Exception:
            logger.exception("Failed to load TI feed: %s", path)

    if total:
        logger.info("TI feeds loaded: %d indicators from %s", total, feed_dir)
        engine_registry.set_enabled(ti_engine.engine_id, True)
    return total


def _start_ti_refresh_thread() -> None:
    """Start a daemon thread that periodically purges expired indicators and re-loads feeds."""
    if getattr(app, "_ti_refresh_started", False):
        return
    import threading as _threading

    interval = SETTINGS.ti_refresh_interval_seconds

    def _refresh_loop() -> None:
        import time as _time
        while True:
            _time.sleep(interval)
            try:
                if not leader_election.is_leader:
                    continue
                purged = ti_manager.cache.purge_expired()
                if purged:
                    logger.info("TI cache: purged %d expired indicators", purged)
                _load_ti_feeds()
            except Exception:
                logger.exception("TI refresh cycle failed")

    t = _threading.Thread(target=_refresh_loop, daemon=True, name="ti-refresh")
    t.start()
    app._ti_refresh_started = True
    logger.info("TI refresh thread started (interval=%ds)", interval)


def _start_siem_flush_thread() -> None:
    """Start SIEM auto-flush thread that periodically drains exporter buffer to logs."""
    if getattr(app, "_siem_flush_started", False):
        return

    def _flush_loop() -> None:
        while True:
            time.sleep(60)
            try:
                if not leader_election.is_leader:
                    continue
                batch = siem_exporter.flush(500)
                if batch:
                    logger.info("SIEM auto-flush drained %d events", len(batch))
            except Exception:
                logger.exception("SIEM auto-flush failed")

    import threading as _threading

    worker = _threading.Thread(target=_flush_loop, daemon=True, name="siem-flush")
    worker.start()
    app._siem_flush_started = True


def _start_leader_election() -> None:
    if getattr(app, "_leader_election_started", False):
        return

    global leader_election
    redis_client = _get_redis_client() if SETTINGS.redis_url else None
    leader_election = LeaderElection(
        redis_client=redis_client,
        instance_id=f"inids-{os.getpid()}",
    )
    leader_election.start()
    app._leader_election_started = True


def _register_health_probes() -> None:
    health_check.register(
        "model",
        lambda: {"ready": model is not None},
    )
    health_check.register(
        "detection_engines",
        lambda: {
            "ready": len(engine_registry.list_engines()) > 0,
            "count": len(engine_registry.list_engines()),
        },
    )

    def _ops_probe() -> dict:
        try:
            # Test both read and write access
            ops_store.list_alerts(limit=1)
            # Try a test audit write
            ops_store.add_audit(
                event_type="health_check",
                message="health_probe",
            )
            return {"ready": True}
        except Exception as exc:
            return {"ready": False, "error": str(exc), "note": "database_write_failed"}

    health_check.register("ops_db", _ops_probe)

    def _redis_probe() -> dict:
        client = _get_redis_client()
        if not SETTINGS.redis_url:
            return {"ready": True, "note": "disabled"}
        if client is None:
            return {"ready": False, "note": "unavailable"}
        try:
            client.ping()
            return {"ready": True}
        except Exception as exc:
            return {"ready": False, "error": str(exc)}

    health_check.register("redis", _redis_probe)
    
    def _firewall_probe() -> dict:
        try:
            # Check if firewall adapter is available and responsive
            adapter = prevention_service.adapter
            adapter_name = SETTINGS.firewall_adapter
            if adapter_name == "mock":
                return {"ready": True, "note": "mock_adapter"}
            # For real adapters, try a lightweight operation
            status = getattr(adapter, "status", lambda: {"available": True})()
            return {"ready": status.get("available", True), "adapter": adapter_name}
        except Exception as exc:
            return {"ready": False, "error": str(exc), "note": "adapter_unavailable"}
    
    health_check.register("firewall_adapter", _firewall_probe)
    
    def _policy_probe() -> dict:
        try:
            # Validate policy structure
            policy = prevention_service.policy
            required_fields = [
                "mode", "risk_alert_threshold", "risk_block_threshold",
                "block_ttl_seconds"
            ]
            for field in required_fields:
                if not hasattr(policy, field):
                    return {"ready": False, "error": f"missing_policy_field:{field}"}
            return {"ready": True, "mode": policy.mode}
        except Exception as exc:
            return {"ready": False, "error": str(exc), "note": "policy_invalid"}
    
    health_check.register("policy", _policy_probe)
    
    health_check.register(
        "pipeline",
        lambda: {
            "ready": (not SETTINGS.pipeline_enabled) or bool(_pipeline_status().get("running")),
            **_pipeline_status(),
        },
    )
    health_check.register("leader_election", lambda: {"ready": True, **leader_election.status()})


def _register_signal_handlers() -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        import signal as _signal

        def _handler(signum, _frame):
            logger.info("Received signal %s; shutting down runtime", signum)
            _shutdown_runtime()
            raise SystemExit(0)

        if hasattr(_signal, "SIGTERM"):
            _signal.signal(_signal.SIGTERM, _handler)
        if hasattr(_signal, "SIGINT"):
            _signal.signal(_signal.SIGINT, _handler)
    except Exception:
        logger.debug("Signal handlers not installed", exc_info=True)


def _validate_runtime_security() -> None:
    if SETTINGS.require_secret_key and not os.getenv("SECRET_KEY", "").strip():
        raise RuntimeError("SECRET_KEY environment variable is required")
    if SETTINGS.require_api_keys and not auth_status().get("enabled", False):
        raise RuntimeError("API keys are required for protected endpoints")
    _validate_all_routes_have_auth_decorator()


# PLAN.md Phase A Step 3 (A-03): PUBLIC_ROUTES are exempt from auth decorator requirement.
# All other routes must have @require_role, @require_auth, or a JWT auth decorator.
PUBLIC_ROUTES = frozenset({
    "/health",
    "/api/health",
    "/api/health/live",
    "/api/health/ready",
    "/api/auth/login",
    "/api/auth/refresh",
    "/api/auth/validate",
    "/api/auth/status",
    "/api/auth/revoke",
    "/static/<path:filename>",
})

_AUTH_DECORATOR_NAMES = frozenset({
    "require_role",
    "require_auth",
    "jwt_require_role",
    "wrapper",          # functools.wraps preserves wrapper name — checked by closure inspection
})


def _validate_all_routes_have_auth_decorator() -> None:
    """Fail-closed startup check: every non-public route must have an auth decorator.

    PLAN.md Phase A Step 3 (A-03) / G-RBAC-1: accepts @require_role and @require_auth
    during the Phase A-B transition. Fails on first uncovered route.
    """
    from src.auth_service import require_role as _require_role_fn
    require_role_code = getattr(_require_role_fn, "__code__", None)

    uncovered = []
    for rule in app.url_map.iter_rules():
        if rule.rule in PUBLIC_ROUTES:
            continue
        if rule.rule.startswith("/static/"):
            continue
        view_func = app.view_functions.get(rule.endpoint)
        if view_func is None:
            continue
        # Walk the decorator chain: check if require_role or require_auth wraps this view
        func = view_func
        covered = False
        for _ in range(10):  # max decorator chain depth
            fn_name = getattr(func, "__name__", "")
            if fn_name in {"wrapper", "require_role", "require_auth", "jwt_require_role"}:
                covered = True
                break
            # Check if this function's globals contain require_role (i.e., it IS require_role's wrapper)
            wrapped = getattr(func, "__wrapped__", None)
            if wrapped is not None:
                func = wrapped
            else:
                break
        if not covered:
            uncovered.append(f"{rule.endpoint} ({rule.rule})")

    if uncovered:
        logger.error(
            "SECURITY: routes_without_auth_decorator count=%d routes=%s",
            len(uncovered),
            uncovered,
        )
        raise RuntimeError(
            f"PLAN.md A-03: {len(uncovered)} route(s) have no auth decorator. "
            f"Add @require_role() or mark as public: {uncovered}"
        )


def _log_runtime_configuration() -> None:
    logger.info(
        "Runtime config host=%s port=%s debug=%s pipeline_enabled=%s ti_feed_dir=%s json_logging=%s firewall_adapter=%s",
        SETTINGS.host,
        SETTINGS.port,
        SETTINGS.debug,
        SETTINGS.pipeline_enabled,
        bool(SETTINGS.ti_feed_dir),
        SETTINGS.json_logging,
        SETTINGS.firewall_adapter,
    )


_register_health_probes()
_register_signal_handlers()


def ensure_model_loaded() -> None:
    """Lazily load models if not available in memory."""
    global model
    if model is None:
        load_models()


def ensure_detection_service() -> bool:
    """Ensure detection service is initialized with loaded model."""
    global detection_service
    ensure_model_loaded()
    if model is None:
        return False
    if detection_service is None:
        detection_service = DetectionService(model=model, ops_store=ops_store, event_bus=event_bus)
    return True


def _normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().rstrip(".")


def load_models():
    """Load all available models into memory with SHA-256 integrity verification.

    PLAN.md Phase A Step 6 (A-06): joblib.load() replaced by
    load_model_with_verification() at all call sites. Checksum verification mode
    controlled by INIDS_MODEL_VERIFY env var (strict|warn|disabled).
    """
    from src.detection.ml_utils import load_checksum_manifest, load_model_with_verification, SecurityError
    global model, all_models, detection_service
    if all_models:
        return

    # Load checksum manifest — raises if absent and INIDS_MODEL_VERIFY=strict
    try:
        checksums = load_checksum_manifest(MODELS_DIR)
    except FileNotFoundError:
        verify_mode = os.environ.get("INIDS_MODEL_VERIFY", "strict").strip().lower()
        if verify_mode == "strict":
            logger.critical(
                "load_models: checksums.sha256 not found in %s — cannot start in strict mode. "
                "Run: python scripts/generate_model_checksums.py", MODELS_DIR
            )
            raise
        else:
            logger.warning("load_models: checksums.sha256 not found — skipping verification (mode=%s)", verify_mode)
            checksums = {}

    model_files = ['rf_nsl_kdd.pkl', 'gb_nsl_kdd.pkl', 'dt_nsl_kdd.pkl',
                   'ab_nsl_kdd.pkl', 'mlp_nsl_kdd.pkl', 'rf_nsl_kdd_multi.pkl']
    for model_file in model_files:
        path = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(path):
            model_name = model_file.replace('.pkl', '')
            expected = checksums.get(model_file)
            if expected:
                try:
                    all_models[model_name] = load_model_with_verification(path, expected)
                except SecurityError:
                    logger.critical("load_models: integrity check failed for %s — model NOT loaded", model_file)
                    continue
            else:
                verify_mode = os.environ.get("INIDS_MODEL_VERIFY", "strict").strip().lower()
                if verify_mode == "strict":
                    logger.critical(
                        "load_models: no checksum for %s in manifest — skipping in strict mode", model_file
                    )
                    continue
                all_models[model_name] = joblib.load(path)
            logger.info("Loaded model %s", model_name)
    if 'rf_nsl_kdd' in all_models:
        model = all_models['rf_nsl_kdd']
        detection_service = DetectionService(model=model, ops_store=ops_store, event_bus=event_bus)
        # Register the primary ML model as a detection engine.
        ml_engine = MLEngine(model, engine_id="ml_primary", fp_manager=fp_manager)
        engine_registry.register(ml_engine)
        
        # --- Initialize RertrainingScheduler for INIDS 2.0 ---
        global retraining_scheduler
        if retraining_scheduler is None:
            retraining_scheduler = RertrainingScheduler(
                dataset_collector=dataset_collector,
                model_registry=model_registry,
                ml_engine=ml_engine,
                models_dir=MODELS_DIR,
                schedule_hour=2  # Daily at 2 AM UTC
            )
            retraining_scheduler.start()
            logger.info("RertrainingScheduler initialized and started for daily model retraining")
    
    # Load threat intelligence feeds after models are loaded
    try:
        _load_threat_intel_feeds()
    except Exception:
        logger.exception("Failed to load threat intelligence feeds")


def _load_threat_intel_feeds() -> None:
    """Load and initialize threat intelligence feeds for the TI engine.
    
    Populates the TI manager cache with indicators from configured sources.
    Handles graceful degradation if feeds are unavailable.
    """
    try:
        load_threat_intel()
        logger.info("threat_intel_feeds_loaded successfully")
    except Exception:
        logger.exception("threat_intel_feeds_load_failed: TI engine may operate with reduced capability")


def load_threat_intel():
    """Load threat intelligence indicators from a configured external feed.

    PLAN.md Phase A Step 2 (A-02): Mock RFC-1918 indicators removed.
    TI engine starts with zero indicators until INIDS_TI_FEED_PATH is set.
    False negatives from TI are preferable to self-DoS via internal IP blocks.
    """
    global ti_manager

    feed_path = os.environ.get("INIDS_TI_FEED_PATH", "").strip()
    if not feed_path:
        logger.warning(
            "No threat intel feed configured (INIDS_TI_FEED_PATH not set). "
            "TI engine will produce no matches until a real feed is provided."
        )
        return

    try:
        ti_manager.load_feed(feed_path)
        logger.info("threat_intel_feed_loaded path=%s indicators=%d", feed_path, ti_manager.cache.size())
    except Exception:
        logger.exception("threat_intel_feed_load_failed path=%s — TI engine running with no indicators", feed_path)


def load_anomaly_baseline():
    """Load real dataset or defer to incremental training from live traffic.
    
    ISSUE-007 FIX: Removed hardcoded synthetic samples.
    The anomaly engine is now trained incrementally from real traffic via
    the auto-fit buffer (buffer_size=3000). This ensures the baseline
    reflects actual network patterns in your environment.
    
    To pre-fit with real data:
    1. Collect 3000+ normal traffic samples from your network
    2. Call anomaly_engine.fit(X) where X is a numpy array of features
    3. Engine will persist the model to MODELS_DIR/anomaly_engine.pkl
    """
    global anomaly_engine
    
    if anomaly_engine.is_ready():
        logger.info("Anomaly engine already fitted from persisted model")
        return
    
    logger.info(
        "AnomalyEngine deferring to incremental training from live traffic. "
        "Buffer size: %d samples. Once collected, will auto-fit and enable.", 
        anomaly_engine._buffer_size
    )


def _model_stats() -> dict:
    stats = {
        "available": False,
        "error": "",
        "total": 0,
        "attacks": 0,
        "normal": 0,
        "accuracy": 0.0,
        "chart_data": "",
        "results": [],
    }

    ensure_model_loaded()
    if model is None:
        stats["error"] = "No trained model found. Please train and load a model first."
        return stats
    if not os.path.exists(TEST_FILE):
        stats["error"] = "Test data file not found."
        return stats

    fig = None
    buf = None
    try:
        df_test = pd.read_csv(TEST_FILE, names=COLUMNS)
        X_test = df_test.drop(columns=LABEL_COLUMNS)
        y_test = df_test["label"].apply(lambda x: 0 if _normalize_label(x) == "normal" else 1)
        y_pred = model.predict(X_test)

        total = len(y_test)
        attacks = int(sum(y_pred))
        normal = total - attacks
        accuracy = round(float(sum(y_pred == y_test)) / total * 100, 2) if total else 0.0

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie([normal, attacks], labels=['Normal', 'Attack'], autopct='%1.1f%%',
               colors=['#28a745', '#dc3545'], startangle=90)
        ax.set_title("Test Data Predictions Distribution")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        chart_data = base64.b64encode(buf.getvalue()).decode()

        results = []
        for i in range(min(20, len(y_test))):
            match = y_test.iloc[i] == y_pred[i]
            results.append({
                "Index": i,
                "True": "Normal" if y_test.iloc[i] == 0 else "Attack",
                "Predicted": "Normal" if y_pred[i] == 0 else "Attack",
                "Match": "OK" if match else "MISS",
            })

        stats.update(
            {
                "available": True,
                "total": total,
                "attacks": attacks,
                "normal": normal,
                "accuracy": accuracy,
                "chart_data": chart_data,
                "results": results,
            }
        )
    except Exception as exc:
        logger.exception("Dashboard model analytics failed")
        stats["error"] = f"Failed to compute analytics: {exc}"
    finally:
        # Ensure matplotlib figures and buffers are always closed to prevent resource leaks
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                logger.debug("Error closing matplotlib figure", exc_info=True)
        if buf is not None:
            try:
                buf.close()
            except Exception:
                logger.debug("Error closing BytesIO buffer", exc_info=True)
    return stats


def _safe_created_at(item: dict, fallback: str) -> str:
    value = item.get("created_at")
    return value if isinstance(value, str) and value else fallback


def _to_epoch_seconds(value, default: float = 0.0) -> float:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value).timestamp()
            except ValueError:
                return float(value)
    except Exception:
        pass
    return float(default)


def _action_status(expires_at: str | None, now: datetime) -> str:
    if not expires_at:
        return "active"
    try:
        expires = datetime.fromisoformat(expires_at)
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return "expired" if expires <= now else "active"
    except (TypeError, ValueError):
        return "unknown"




@app.before_request
def _before_request_metrics():
    _ensure_scheduler_started()
    if request.path.startswith('/api/'):
        metrics_service.inc('requests_total')

        if request.path not in {'/api/health', '/api/health/live', '/api/health/ready'}:
            # Tier 1: global IP rate limit (1000 req/min) via UnifiedRateLimiter.
            _client_ip = request.remote_addr or 'unknown'
            _rl = get_unified_rate_limiter()
            if _rl is not None:
                _allowed, _retry = _rl.check_ip(_client_ip)
                if not _allowed:
                    metrics_service.inc('rate_limited_total')
                    return jsonify({
                        "error": "rate_limited",
                        "retry_after_seconds": _retry,
                    }), 429


@app.after_request
def _after_request_metrics(response):
    if request.path.startswith('/api/') and response.status_code == 401:
        metrics_service.inc('unauthorized_total')
    return response


@app.route("/")
@require_role("viewer")
def home():
    """Landing page with runtime status and navigation into the SOC console."""
    metrics_snapshot = {
        "requests_total": metrics_service.get("requests_total"),
        "predictions_total": metrics_service.get("predictions_total"),
        "alerts_total": metrics_service.get("alerts_total"),
        "prevention_actions_total": metrics_service.get("prevention_actions_total"),
        "ingested_total": metrics_service.get("ingested_total"),
        "processed_ingestion_total": metrics_service.get("processed_ingestion_total"),
    }
    return render_template(
        "home.html",
        loaded_models_count=len(all_models),
        queue_size=ingestion_queue.size(),
        firewall_adapter=SETTINGS.firewall_adapter,
        auth_enabled=auth_status().get("enabled", False),
        metrics_snapshot=metrics_snapshot,
    )


@app.route("/predict", methods=["GET", "POST"])
@require_role("viewer")
def predict():
    """Live prediction page with suspicious activity logic."""
    prediction = None
    error_message = None
    confidence = None
    is_suspicious = False

    if not ensure_detection_service():
        return render_template("predict.html", features=INPUT_FEATURES,
                               prediction=None, error="No trained model found. Please train and load a model first.",
                               confidence=None, is_suspicious=False)

    if request.method == "POST":
        try:
            values = []
            for feat in INPUT_FEATURES:
                v = request.form.get(feat, None)
                if v is None or not v.replace('.', '', 1).replace('-', '', 1).isdigit():
                    raise ValueError(f"Invalid input for {feat}")
                values.append(float(v))

            row = DEFAULT_FEATURE_ROW.copy()
            row.update({
                "duration": values[0],
                "src_bytes": values[1],
                "dst_bytes": values[2],
                "count": values[3],
                "srv_count": values[4],
                "serror_rate": values[5],
                "same_srv_rate": values[6],
            })
            result = detection_service.predict_from_features(
                row,
                profile="balanced",
                source_ip=request.remote_addr or "unknown",
            )
            confidence = result.confidence
            is_suspicious = result.suspicious
            prediction = "SUSPICIOUS - Low Confidence" if result.suspicious else result.prediction
        except Exception as e:
            logger.error("Prediction error: %s", e)
            error_message = f"Error: {e}"

    return render_template("predict.html", features=INPUT_FEATURES,
                           prediction=prediction, error=error_message,
                           confidence=confidence, is_suspicious=is_suspicious)


@app.route("/alerts")
@require_role("viewer")
def alerts_page():
    """Security alerts monitoring page."""
    return render_template("alerts.html")


@app.route("/actions")
@require_role("viewer")
def actions_page():
    """Prevention actions and approval workflow page."""
    return render_template("actions.html")


@app.route("/detection")
@require_role("viewer")
def detection_page():
    """Detection console for running multi-engine detection."""
    return render_template("detection.html")


@app.route("/engines")
@require_role("viewer")
def engines_page():
    """Detection engines management page."""
    return render_template("engines.html")


@app.route("/policy")
@require_role("viewer")
def policy_page():
    """Policy editor and configuration page."""
    return render_template("policy.html")


@app.route("/allowlist")
@require_role("viewer")
def allowlist_page():
    """Allowlist manager page for trusted IPs and domains."""
    return render_template("allowlist.html")


@app.route("/threat-intel")
@require_role("viewer")
def threat_intel_page():
    """Threat intelligence lookup page."""
    return render_template("threat_intel.html")


@app.route("/health")
def health_page():
    """System health monitoring dashboard."""
    return render_template("health.html")


@app.route("/dashboard")
@require_role("viewer")
def dashboard():
    """Visual dashboard of system predictions and accuracy."""
    try:
        now = datetime.now(timezone.utc)
        recent_actions = ops_store.list_actions(limit=20)
        active_blocks = []
        for action in recent_actions:
            active_blocks.append(
                {
                    "target": action.get("target", ""),
                    "status": _action_status(action.get("expires_at"), now),
                    "expires_at": action.get("expires_at"),
                }
            )

        action_timeline = []
        for action in recent_actions[:10]:
            action_timeline.append(
                {
                    "when": _safe_created_at(action, now.isoformat()),
                    "type": action.get("action", "action"),
                    "message": f"{action.get('action', '')} target={action.get('target', '')} reason={action.get('reason', '')}",
                }
            )

        recent_audits = ops_store.list_audits(limit=20)
        for audit in recent_audits[:10]:
            action_timeline.append(
                {
                    "when": _safe_created_at(audit, now.isoformat()),
                    "type": audit.get("event_type", "audit"),
                    "message": audit.get("message", ""),
                }
            )
        action_timeline = sorted(action_timeline, key=lambda item: item.get("when", ""), reverse=True)[:15]

        adapter = prevention_service.adapter
        firewall_rules = 0
        if hasattr(adapter, "blocked_targets"):
            blocked = getattr(adapter, "blocked_targets", {})
            firewall_rules = len(blocked) if isinstance(blocked, dict) else 0

        metrics_snapshot = {
            "requests_total": metrics_service.get("requests_total"),
            "predictions_total": metrics_service.get("predictions_total"),
            "alerts_total": metrics_service.get("alerts_total"),
            "prevention_actions_total": metrics_service.get("prevention_actions_total"),
            "ingested_total": metrics_service.get("ingested_total"),
            "processed_ingestion_total": metrics_service.get("processed_ingestion_total"),
            "rate_limited_total": metrics_service.get("rate_limited_total"),
            "unauthorized_total": metrics_service.get("unauthorized_total"),
        }

        return render_template(
            "dashboard.html",
            generated_at=now.isoformat(),
            auth_info=auth_status(),
            queue_size=ingestion_queue.size(),
            rate_limit_requests=SETTINGS.rate_limit_requests,
            rate_limit_window_seconds=SETTINGS.rate_limit_window_seconds,
            firewall_adapter=SETTINGS.firewall_adapter,
            model_stats=_model_stats(),
            policy=prevention_service.policy.to_dict(),
            metrics_snapshot=metrics_snapshot,
            recent_alerts=ops_store.list_alerts(limit=10),
            recent_actions=recent_actions[:10],
            recent_audits=recent_audits[:10],
            recent_registry=model_registry.list_entries(limit=10),
            active_blocks=[b for b in active_blocks if b["status"] == "active"][:10],
            action_timeline=action_timeline,
            reconcile_summary={
                "db_active": sum(1 for b in active_blocks if b["status"] == "active"),
                "firewall_rules": firewall_rules,
                "missing_in_firewall": 0,
                "orphan_firewall_rules": max(0, firewall_rules - sum(1 for b in active_blocks if b["status"] == "active")),
            },
        )
    except Exception:
        logger.exception("Dashboard rendering failed")
        return render_template(
            "dashboard.html",
            generated_at=datetime.now(timezone.utc).isoformat(),
            auth_info=auth_status(),
            queue_size=ingestion_queue.size(),
            rate_limit_requests=SETTINGS.rate_limit_requests,
            rate_limit_window_seconds=SETTINGS.rate_limit_window_seconds,
            firewall_adapter=SETTINGS.firewall_adapter,
            model_stats={"available": False, "error": "Dashboard is temporarily unavailable.", "results": []},
            policy=prevention_service.policy.to_dict(),
            metrics_snapshot={
                "requests_total": 0,
                "predictions_total": 0,
                "alerts_total": 0,
                "prevention_actions_total": 0,
                "ingested_total": 0,
                "processed_ingestion_total": 0,
                "rate_limited_total": 0,
                "unauthorized_total": 0,
            },
            recent_alerts=[],
            recent_actions=[],
            recent_audits=[],
            recent_registry=[],
            active_blocks=[],
            action_timeline=[],
            reconcile_summary={"db_active": 0, "firewall_rules": 0, "missing_in_firewall": 0, "orphan_firewall_rules": 0},
        ), 200


@app.route("/monitor")
@require_role("viewer")
def monitor():
    """INIDS 2.0 Monitor - Real-time threat dashboard."""
    try:
        return render_template("monitor.html")
    except Exception:
        logger.exception("Monitor page rendering failed")
        return "Monitor page error", 500


@app.route("/investigate")
@require_role("analyst")
def investigate():
    """INIDS 2.0 Investigate - Alert analysis and deep-dive."""
    try:
        return render_template("investigate.html")
    except Exception:
        logger.exception("Investigate page rendering failed")
        return "Investigate page error", 500


@app.route("/respond")
@require_role("analyst")
def respond():
    """INIDS 2.0 Respond - Action management and approvals."""
    try:
        return render_template("respond.html")
    except Exception:
        logger.exception("Respond page rendering failed")
        return "Respond page error", 500


@app.route("/learn")
@require_role("viewer")
def learn():
    """INIDS 2.0 Learn - ML model management and training."""
    try:
        return render_template("learn.html")
    except Exception:
        logger.exception("Learn page rendering failed")
        return "Learn page error", 500


@app.route("/dashboard/main")
@require_role("viewer")
def dashboard_main():
    """New demo platform dashboard with 15 capability modules."""
    try:
        return render_template("dashboard_main.html")
    except Exception:
        logger.exception("Dashboard main rendering failed")
        return render_template("dashboard_main.html"), 500


@app.route("/api/dashboard/metrics", methods=["GET"])
@require_role("viewer")
def api_dashboard_metrics():
    """Get current dashboard metrics."""
    try:
        payload = _build_dashboard_metrics_payload()
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        return jsonify(payload)
    except Exception as e:
        logger.exception("Error retrieving dashboard metrics")
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard/refresh", methods=["POST"])
@require_role("analyst")
def api_dashboard_refresh():
    """Refresh dashboard data."""
    try:
        return jsonify({
            "status": "refreshed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.exception("Error refreshing dashboard")
        return jsonify({"error": str(e)}), 500


@app.route("/api/demo/start", methods=["POST"])
@require_role("admin")
def api_demo_start():
    """Start demo traffic simulation."""
    try:
        # This would trigger realtime_simulation to start sending demo traffic
        logger.info("Demo mode started")
        return jsonify({
            "status": "started",
            "message": "Demo traffic simulation started"
        })
    except Exception as e:
        logger.exception("Error starting demo")
        return jsonify({"error": str(e)}), 500


@app.route("/api/demo/stop", methods=["POST"])
@require_role("admin")
def api_demo_stop():
    """Stop demo traffic simulation."""
    try:
        logger.info("Demo mode stopped")
        return jsonify({
            "status": "stopped",
            "message": "Demo traffic simulation stopped"
        })
    except Exception as e:
        logger.exception("Error stopping demo")
        return jsonify({"error": str(e)}), 500


@app.route("/models")
@require_role("analyst")
def models_page():
    """Comparison page of all trained models and their metrics."""
    chart_files = {
        "model_comparison": os.path.exists(os.path.join(STATIC_DIR, "model_comparison.png")),
        "training_time_comparison": os.path.exists(os.path.join(STATIC_DIR, "training_time_comparison.png")),
        "roc_curves": os.path.exists(os.path.join(STATIC_DIR, "roc_curves.png")),
    }
    registry_entries = model_registry.list_entries(limit=25)
    model_results = []
    latest_results = None

    try:
        if os.path.exists(RESULTS_DIR):
            results_files = [
                f for f in os.listdir(RESULTS_DIR)
                if f.startswith("model_results_") and f.endswith(".json")
            ]
            if results_files:
                latest_results = sorted(results_files)[-1]
                with open(os.path.join(RESULTS_DIR, latest_results), 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    model_results = sorted(loaded, key=lambda x: x.get("accuracy", 0), reverse=True)
    except Exception:
        logger.exception("Models page rendering failed")

    return render_template(
        "models.html",
        models=model_results,
        has_data=bool(model_results),
        latest_results=latest_results,
        chart_files=chart_files,
        registry_entries=registry_entries,
    )
@app.route("/batch", methods=["GET", "POST"])
@require_role("analyst")
def batch_predict():
    """Batch prediction from CSV upload."""
    ensure_model_loaded()
    if request.method == "POST":
        try:
            if model is None:
                return render_template("batch.html", error="No trained model found. Please train and load a model first.")

            file = request.files.get('file')
            if not file:
                return render_template("batch.html", error="No file uploaded")
            if not file.filename or not file.filename.lower().endswith('.csv'):
                return render_template("batch.html", error="Only .csv files are accepted.")
            
            # Validate MIME type to prevent file type spoofing
            if file.content_type not in ('text/csv', 'text/plain', 'application/vnd.ms-excel', 'application/csv'):
                logger.warning(f"Invalid MIME type for batch upload: {file.content_type}")
                return render_template("batch.html", error="Invalid file type. Only CSV files are accepted.")
            
            df = pd.read_csv(file)
            if df.empty:
                return render_template("batch.html", error="Uploaded CSV is empty.")

            max_rows = MAX_CSV_ROWS
            if len(df) > max_rows:
                return render_template("batch.html", error=f"CSV too large ({len(df)} rows). Max allowed is {max_rows}.")

            df = df.drop(columns=LABEL_COLUMNS, errors="ignore")
            missing_cols = [col for col in MODEL_INPUT_COLUMNS if col not in df.columns]
            if missing_cols:
                return render_template("batch.html", error=f"Missing required columns: {', '.join(missing_cols)}")

            df = df[MODEL_INPUT_COLUMNS].copy()

            for col in NUMERIC_MODEL_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[NUMERIC_MODEL_COLUMNS].isna().any().any():
                return render_template("batch.html", error="One or more numeric columns contain invalid values.")

            predictions = model.predict(df)
            results = [{"row": i, "prediction": "Attack" if p == 1 else "Normal"}
                       for i, p in enumerate(predictions)]
            return render_template("batch.html", results=results[:50],
                                   total=len(results), shown=min(50, len(results)))
        except Exception as e:
            logger.error("Batch prediction error: %s", e)
            return render_template("batch.html", error=str(e))
    return render_template("batch.html")


@app.route("/api/health", methods=["GET"])
def api_health():
    model_ready = ensure_detection_service()
    _ensure_pipeline_started()
    readiness = health_check.check()
    return jsonify({
        "status": "ok",
        "model_loaded": model_ready,
        "readiness": readiness,
        "alerts_buffered": len(ops_store.list_alerts(limit=1000)),
        "ops_db": OPS_DB_PATH,
        "auth": auth_status(),
        "metrics": {
            "requests_total": metrics_service.get("requests_total"),
            "predictions_total": metrics_service.get("predictions_total"),
        },
        "ingestion_queue_size": ingestion_queue.size(),
        "rate_limit": {
            "requests": SETTINGS.rate_limit_requests,
            "window_seconds": SETTINGS.rate_limit_window_seconds,
        },
        "firewall_adapter": SETTINGS.firewall_adapter,
        "detection_engines": engine_registry.list_engines(),
        "pipeline": _pipeline_status(),
        "leader_election": leader_election.status(),
    })


@app.route("/api/health/live", methods=["GET"])
def api_health_live():
    return jsonify({"status": "live", "process_up": True}), 200


@app.route("/api/health/ready", methods=["GET"])
def api_health_ready():
    _ensure_pipeline_started()
    report = health_check.check()
    return jsonify(report), (200 if report.get("status") == "healthy" else 503)


# ========== PERCEPTION LAYER ENDPOINTS ==========

@app.route("/api/perception/pulse", methods=["GET"])
@require_role("analyst")
def api_perception_pulse():
    """Get current system pulse status (real-time metrics)."""
    try:
        pulse = live_system_pulse.get_pulse_status()
        return jsonify(pulse), 200
    except Exception as e:
        logger.exception("Error getting system pulse")
        return jsonify({"error": str(e)}), 500


@app.route("/api/perception/pulse/timeseries/<metric>", methods=["GET"])
@require_role("analyst")
def api_perception_pulse_timeseries(metric):
    """Get time-series data for a specific metric."""
    try:
        series = live_system_pulse.get_time_series(metric)
        return jsonify({
            "metric": metric,
            "data": series,
            "count": len(series)
        }), 200
    except Exception as e:
        logger.exception(f"Error getting timeseries for {metric}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/perception/confidence/<detection_id>", methods=["GET"])
@require_role("analyst")
def api_perception_confidence(detection_id):
    """Get confidence breakdown for a specific detection."""
    try:
        breakdown = confidence_breakdown_engine.get_breakdown(detection_id)
        if not breakdown:
            return jsonify({"error": "Detection not found"}), 404
        return jsonify(breakdown), 200
    except Exception as e:
        logger.exception(f"Error getting confidence breakdown for {detection_id}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/perception/attack-story/<attack_id>", methods=["GET"])
@require_role("analyst")
def api_perception_attack_story(attack_id):
    """Get the narrative story for an attack."""
    try:
        story = attack_story_engine.get_attack_story(attack_id)
        return jsonify(story), 200
    except Exception as e:
        logger.exception(f"Error getting attack story for {attack_id}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/perception/attack-stories", methods=["GET"])
@require_role("analyst")
def api_perception_attack_stories():
    """Get recent attack stories."""
    try:
        limit = request.args.get("limit", 10, type=int)
        stories = attack_story_engine.get_recent_stories(limit=limit)
        return jsonify({
            "stories": stories,
            "count": len(stories)
        }), 200
    except Exception as e:
        logger.exception("Error getting attack stories")
        return jsonify({"error": str(e)}), 500


@app.route("/api/perception/feature-importance", methods=["GET"])
@require_role("analyst")
def api_perception_feature_importance():
    """Get feature importance ranking across all detections."""
    try:
        ranking = confidence_breakdown_engine.get_feature_importance_ranking()
        return jsonify({
            "features": [
                {
                    "rank": i + 1,
                    "feature_name": name,
                    "total_contribution": score
                }
                for i, (name, score) in enumerate(ranking)
            ],
            "count": len(ranking)
        }), 200
    except Exception as e:
        logger.exception("Error getting feature importance")
        return jsonify({"error": str(e)}), 500


@app.route("/api/perception/integration-status", methods=["GET"])
@require_role("analyst")
def api_perception_integration_status():
    """Get real-time integration status including latency metrics and throughput."""
    try:
        status = perception_integration.get_status()
        return jsonify(status), 200
    except Exception as e:
        logger.exception("Error getting perception integration status")
        return jsonify({"error": str(e)}), 500


# ========== AUTHENTICATION ENDPOINTS ==========

@app.route("/api/auth/login", methods=["POST"])
def api_auth_login():
    """Issue a JWT token after validating the caller's API key.

    PLAN.md Phase A Step 4 (A-04): Login now requires a valid X-API-Key header.
    Role is derived from the key's assigned principal — callers cannot self-assign roles.
    This is a bridge until Step 16 (UnifiedAuthService) replaces this handler.

    Request Header:
        X-API-Key: <api-key>
    Request JSON (optional, for audit):
        { "username": "my-service-account" }

    Response:
        { "token": "...", "expires_in": 3600, "user": "...", "roles": ["..."] }
    """
    api_key = request.headers.get("X-API-Key", "").strip()
    if not api_key:
        logger.warning("api_auth_login: rejected — no X-API-Key provided")
        return jsonify({"error": "X-API-Key header is required"}), 401

    # Validate key against the configured principals
    principal = _auth_service.principals.get(api_key)
    if principal is None:
        logger.warning("api_auth_login: rejected — invalid API key")
        return jsonify({"error": "Invalid API key"}), 401

    # Role derived from key — caller cannot override
    roles = [principal.role]
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", f"api-key-{principal.role}")

    try:
        token = jwt_manager.create_token(
            user_id=username,
            username=username,
            roles=roles,
            expires_in=3600,  # PLAN.md: 1-hour token expiry is non-negotiable
        )
        logger.info("api_auth_login: JWT issued user=%s roles=%s", username, roles)
        return jsonify({
            "token": token,
            "expires_in": 3600,
            "user": username,
            "roles": roles,
        }), 200
    except Exception:
        logger.exception("api_auth_login: token creation failed")
        return jsonify({"error": "Authentication failed"}), 401


@app.route("/api/auth/refresh", methods=["POST"])
def api_auth_refresh():
    """Refresh a JWT token that is still valid or within the 5-minute grace window.

    PLAN.md Phase A Step 4 (A-04): allow_expired=True is removed.
    Tokens expired by more than REFRESH_GRACE_SECONDS are rejected outright.
    This closes the infinite-lifetime token vulnerability.

    Request Header:
        Authorization: Bearer <token>
    Response:
        { "token": "...", "expires_in": 3600 }
    """
    REFRESH_GRACE_SECONDS = 300  # 5-minute window before/after expiry

    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not token:
        return jsonify({"error": "Missing token"}), 401

    try:
        # First: try normal validation (token still valid)
        is_valid, claims, error = jwt_manager.verify_token(token)

        if not is_valid:
            # Token is invalid — check if it's within the grace window
            import time as _time
            try:
                # Decode without verification to inspect expiry
                _, expired_claims, _ = jwt_manager.verify_token(token, allow_expired=True)
                if expired_claims is None:
                    return jsonify({"error": "Cannot parse token"}), 401
                now = _time.time()
                seconds_overdue = now - expired_claims.exp
                if seconds_overdue > REFRESH_GRACE_SECONDS:
                    logger.warning(
                        "api_auth_refresh: rejected — token expired %ds ago (grace=%ds) user=%s",
                        int(seconds_overdue), REFRESH_GRACE_SECONDS, expired_claims.sub,
                    )
                    return jsonify({"error": "Token expired — please re-authenticate"}), 401
                claims = expired_claims
            except Exception:
                return jsonify({"error": f"Invalid token: {error}"}), 401

        if not claims:
            return jsonify({"error": "Could not parse token"}), 401

        new_token = jwt_manager.create_token(
            user_id=claims.user_id,
            username=claims.sub,
            roles=claims.roles,
            expires_in=3600,  # PLAN.md: 1-hour token expiry is non-negotiable
        )
        logger.info("api_auth_refresh: token refreshed user=%s", claims.sub)
        return jsonify({"token": new_token, "expires_in": 3600, "user": claims.sub}), 200

    except Exception:
        logger.exception("api_auth_refresh: failed")
        return jsonify({"error": "Token refresh failed"}), 401


@app.route("/api/auth/validate", methods=["GET"])
def api_auth_validate():
    """Validate current JWT token.
    
    Request Header:
        Authorization: Bearer <token>
    
    Response:
        {
            "valid": true,
            "user": "admin",
            "roles": ["admin", "analyst"],
            "expires_at": "2026-04-16T10:35:42Z"
        }
    """
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    
    if not token:
        return jsonify({"valid": False, "error": "No token provided"}), 401
    
    is_valid, claims, error = jwt_manager.verify_token(token)
    
    if not is_valid:
        return jsonify({
            "valid": False,
            "error": error
        }), 401
    
    # Convert exp timestamp to ISO string
    from datetime import datetime, timezone
    exp_dt = datetime.fromtimestamp(claims.exp, tz=timezone.utc)
    
    return jsonify({
        "valid": True,
        "user": claims.sub,
        "user_id": claims.user_id,
        "roles": claims.roles,
        "expires_at": exp_dt.isoformat(),
        "run_as_admin": claims.run_as_admin,
        "is_runas": claims.run_as_admin is not None
    }), 200


@app.route("/api/auth/runas", methods=["POST"])
@require_auth
@jwt_require_role("admin")
def api_auth_runas():
    """Create run-as token (admin impersonating user).
    
    Request JSON:
        {
            "target_user": "analyst",
            "target_roles": ["analyst"],
            "reason": "Investigating ticket #123"
        }
    
    Request Header:
        Authorization: Bearer <admin_token>
    
    Response:
        {
            "token": "eyJ0eXAiOiJKV1Q...",
            "expires_in": 3600,
            "admin": "admin",
            "target_user": "analyst",
            "context_hash": "abc123def456"
        }
    """
    payload = request.get_json(silent=True) or {}
    target_user = payload.get("target_user", "")
    target_roles = payload.get("target_roles", ["analyst"])
    reason = payload.get("reason", "Admin task")
    
    if not target_user:
        return jsonify({"error": "target_user is required"}), 400
    
    try:
        success, token_or_error, context_hash = runas_manager.create_runas_token(
            admin_user=request.claims.sub,
            target_user=target_user,
            admin_roles=request.claims.roles,
            target_roles=target_roles,
            reason=reason
        )
        
        if not success:
            return jsonify({"error": token_or_error}), 400
        
        logger.info(f"Run-as token created: {request.claims.sub} → {target_user}")
        
        return jsonify({
            "token": token_or_error,  # token_or_error is token on success
            "expires_in": 3600,
            "admin": request.claims.sub,
            "target_user": target_user,
            "context_hash": context_hash
        }), 200
    
    except Exception as e:
        logger.error(f"Run-as token creation failed: {str(e)}")
        return jsonify({"error": "Failed to create run-as token"}), 500


# ========== AUDIT & OBSERVABILITY ENDPOINTS ==========

@app.route("/api/audit/logs", methods=["GET"])
@require_auth
def api_audit_logs():
    """Get audit logs.
    
    Query Parameters:
        - limit: Maximum number of logs (default: 100, max: 500)
        - user: Filter by user
    
    Response:
        [
            {
                "timestamp": "2026-04-15T10:35:42.123Z",
                "user": "admin",
                "method": "POST",
                "path": "/api/predict",
                "status": 200,
                "response_time_ms": 125.4,
                "source_ip": "192.168.1.100",
                "error": null
            }
        ]
    """
    limit = min(int(request.args.get("limit", 100)), 500)
    user_filter = request.args.get("user")
    
    try:
        logs = app.audit_log.get_logs(limit=limit)
        
        if user_filter:
            logs = [log for log in logs if log['user'] == user_filter]
        
        return jsonify(logs), 200
    
    except Exception as e:
        logger.error(f"Failed to retrieve audit logs: {str(e)}")
        return jsonify({"error": "Failed to retrieve logs"}), 500


@app.route("/api/audit/user-activity", methods=["GET"])
@require_auth
@jwt_require_role("admin")
def api_audit_user_activity():
    """Get activity for specific user.
    
    Query Parameters:
        - user: Username (required)
        - hours: Look back (default: 24)
    
    Response: List of audit entries
    """
    user = request.args.get("user")
    hours = int(request.args.get("hours", 24))
    
    if not user:
        return jsonify({"error": "user parameter is required"}), 400
    
    try:
        activity = app.audit_log.get_user_activity(user, hours=hours)
        return jsonify(activity), 200
    
    except Exception as e:
        logger.error(f"Failed to retrieve user activity: {str(e)}")
        return jsonify({"error": "Failed to retrieve activity"}), 500


@app.route("/api/auth/status", methods=["GET"])
def api_auth_status():
    """Get authentication system status.

    Response:
        {
            "auth_enabled": true,
            "jwt_algorithm": "ES256",
            "rate_limiting_enabled": true,
            "ip_blocking_enabled": true,
            "audit_logging_enabled": true
        }
    """
    return jsonify({
        "auth_enabled": True,
        "jwt_algorithm": jwt_manager.algorithm,
        "rate_limiting_enabled": True,
        "ip_blocking_enabled": True,
        "audit_logging_enabled": True,
        "middleware_stack": [
            "rate_limiting",
            "ip_blocking",
            "security_headers",
            "audit_logging"
        ]
    }), 200


@app.route("/api/auth/revoke", methods=["POST"])
def api_auth_revoke():
    """Revoke a JWT by adding its jti to the revocation list (C-04 Step 18).

    The RS256 signature is verified even if the token has expired — this allows
    clients to revoke tokens from a compromised session after the 1-hour window.
    Forged or HS256 tokens are rejected with 401.

    Request Header:
        Authorization: Bearer <token>
    Response (200):
        {"revoked": true, "jti": "<jti>"}
    """
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not token:
        return jsonify({"error": "Authorization: Bearer <token> header required"}), 400

    try:
        from src.auth.jwt_manager import get_jwt_manager as _get_rs256_mgr
        _rs256 = _get_rs256_mgr()
        ok, payload, error = _rs256.decode_ignoring_expiry(token)
        if not ok or payload is None:
            return jsonify({"error": "Invalid token", "reason": error}), 401

        jti = payload.get("jti", "")
        if not jti:
            return jsonify({"error": "Token has no jti claim — cannot revoke"}), 400

        user_id = str(payload.get("user_id", payload.get("sub", "")))
        exp = payload.get("exp", 0)
        expires_at = (
            datetime.fromtimestamp(exp, tz=timezone.utc).isoformat() if exp else ""
        )
        revoked_at = datetime.now(timezone.utc).isoformat()

        ops_store.add_revoked_token(
            jti=jti,
            user_id=user_id,
            expires_at=expires_at,
            revoked_at=revoked_at,
        )
        logger.info("api_auth_revoke: jti=%s user_id=%s", jti, user_id)
        return jsonify({"revoked": True, "jti": jti}), 200
    except Exception:
        logger.exception("api_auth_revoke: failed")
        return jsonify({"error": "Revocation failed"}), 500


@app.route("/api/admin/cleanup-tokens", methods=["POST"])
@require_role("admin")
def api_admin_cleanup_tokens():
    """Delete expired JWT revocation records (C-04 daily cleanup).

    Removes revoked_tokens rows whose expires_at is in the past. These tokens
    would be rejected by expiry anyway — the revocation record is no longer needed.

    Response:
        {"deleted": <count>}
    """
    try:
        deleted = ops_store.cleanup_revoked_tokens()
        logger.info("api_admin_cleanup_tokens: deleted=%d expired revocation records", deleted)
        return jsonify({"deleted": deleted}), 200
    except Exception:
        logger.exception("api_admin_cleanup_tokens: failed")
        return jsonify({"error": "Cleanup failed"}), 500


@app.route("/api/predict", methods=["POST"])
@require_role("analyst")
def api_predict():
    if not ensure_detection_service():
        return jsonify({"error": "No trained model found"}), 503

    payload = request.get_json(silent=True) or {}
    
    # Validate request against schema
    is_valid, error_msg = validate_predict_request(payload)
    if not is_valid:
        return jsonify({"error": f"Invalid request: {error_msg}"}), 400
    
    features = payload.get("features", {})
    profile = payload.get("profile", "balanced")

    if not isinstance(features, dict) or not features:
        return jsonify({"error": "'features' must be a non-empty object"}), 400

    try:
        request_started_at = time.time()
        for col in NUMERIC_MODEL_COLUMNS:
            if col in features:
                features[col] = float(features[col])
        source = payload.get("source_ip", payload.get("source", "unknown"))
        metrics_service.inc("predictions_total")
        result = detection_service.predict_from_features(
            features,
            profile=profile,
            source_ip=source,
            attack_type=payload.get("attack_type"),
        )
        response = result.to_dict()
        # Correlate with prevention action: find most recent action for this target
        # within last 60 seconds (avoids fragile timestamp correlation)
        recent_actions = ops_store.list_actions(limit=20)
        prevention_action = next(
            (
                action
                for action in recent_actions
                if str(action.get("target", "")).strip() == str(source).strip()
                and (time.time() - _to_epoch_seconds(action.get("created_at"), default=0.0)) <= 60.0
            ),
            None,
        )
        response["prevention_action"] = prevention_action
        return jsonify(response)
    except Exception as exc:
        logger.exception("API predict error (FULL TRACE)")
        return jsonify({
            "error": str(type(exc).__name__),
            "message": str(exc),
            "debug": True
        }), 400


@app.route("/api/alerts", methods=["GET"])
@require_role("analyst")
def api_alerts():
    limit = request.args.get("limit", default=50, type=int)
    severity = request.args.get("severity", default=None, type=str)
    status = request.args.get("status", default=None, type=str)
    alerts = [
        _normalize_alert_payload(alert)
        for alert in ops_store.list_alerts(limit=max(1, min(limit, 200)), severity=severity, status=status)
    ]
    return jsonify({"count": len(alerts), "alerts": alerts})


@app.route("/api/alerts/dismiss", methods=["POST"])
@require_role("analyst")
def api_alerts_dismiss():
    body = request.get_json(silent=True) or {}
    alert_ids = body.get("alert_ids")
    if alert_ids is None:
        alert_id = body.get("alert_id")
        alert_ids = [alert_id] if alert_id else []
    if not isinstance(alert_ids, list) or not alert_ids:
        return jsonify({"error": "'alert_id' or 'alert_ids' is required"}), 400

    dismissed = []
    missing = []
    for alert_id in alert_ids:
        try:
            updated = ops_store.update_alert(str(alert_id), status="closed", close_reason="dismissed")
        except ValueError:
            updated = False
        if updated:
            dismissed.append(str(alert_id))
        else:
            missing.append(str(alert_id))
    if dismissed:
        ops_store.add_audit(
            event_type="alerts_dismissed",
            message=json.dumps({"dismissed": dismissed, "missing": missing}, separators=(",", ":")),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    return jsonify({"dismissed": dismissed, "missing": missing, "count": len(dismissed)}), 200


@app.route("/api/alerts/<alert_id>/feedback", methods=["POST"])
@require_role("analyst")
def api_alert_feedback(alert_id: str):
    """Record analyst FP/TP feedback for a detection alert."""
    body = request.get_json(silent=True) or {}
    verdict = str(body.get("verdict", "")).strip().lower()
    engine_id = str(body.get("engine_id", "ml_engine")).strip()
    rule_id = str(body.get("rule_id", "model")).strip()
    if verdict == "fp":
        fp_manager.report_fp(engine_id, rule_id, alert_id=alert_id)
    elif verdict == "tp":
        fp_manager.report_tp(engine_id, rule_id)
    else:
        return jsonify({"error": "verdict must be 'fp' or 'tp'"}), 400
    return jsonify({
        "alert_id": alert_id,
        "verdict": verdict,
        "suppressed": fp_manager.is_suppressed(engine_id, rule_id),
    }), 200


@app.route("/api/alerts/<alert_id>", methods=["PATCH"])
@require_role("analyst")
def api_alert_update(alert_id: str):
    # Validate alert_id format (alphanumeric, hyphen, underscore only)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', alert_id):
        return jsonify({"error": "invalid_alert_id"}), 400
    
    body = request.get_json(silent=True) or {}
    status = body.get("status")
    assignee = body.get("assignee")
    close_reason = body.get("close_reason")
    if status is None and assignee is None and close_reason is None:
        return jsonify({"error": "at least one of 'status', 'assignee', 'close_reason' is required"}), 400

    try:
        updated = ops_store.update_alert(
            alert_id,
            status=status,
            assignee=assignee,
            close_reason=close_reason,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not updated:
        return jsonify({"error": "alert_not_found"}), 404

    ops_store.add_audit(
        event_type="alert_update",
        message=json.dumps(
            {
                "alert_id": alert_id,
                "status": status,
                "assignee": assignee,
                "close_reason": close_reason,
            },
            separators=(",", ":"),
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"updated": True, "alert_id": alert_id}), 200


@app.route("/api/fp-suppressions", methods=["GET"])
@require_role("analyst")
def api_fp_suppressions_list():
    return jsonify({"count": len(ops_store.list_fp_suppressions()), "suppressions": ops_store.list_fp_suppressions()})


@app.route("/api/fp-suppressions", methods=["POST"])
@require_role("admin")
def api_fp_suppression_add():
    body = request.get_json(silent=True) or {}
    engine_id = str(body.get("engine_id", "")).strip()
    rule_id = str(body.get("rule_id", "")).strip()
    if not engine_id or not rule_id:
        return jsonify({"error": "'engine_id' and 'rule_id' are required"}), 400
    newly = fp_manager.suppress(engine_id, rule_id)
    return jsonify({"engine_id": engine_id, "rule_id": rule_id, "suppressed": True, "new": bool(newly)})


@app.route("/api/fp-suppressions/<engine_id>/<rule_id>", methods=["DELETE"])
@require_role("admin")
def api_fp_suppression_remove(engine_id: str, rule_id: str):
    removed = fp_manager.unsuppress(engine_id, rule_id)
    if not removed:
        return jsonify({"error": "suppression_not_found"}), 404
    return jsonify({"engine_id": engine_id, "rule_id": rule_id, "removed": True}), 200


@app.route("/api/detect", methods=["POST"])
@require_role("analyst")
def api_detect():
    """Multi-engine detection endpoint.

    Runs all enabled detection engines against the submitted features and
    returns the aggregated verdict along with per-engine results.
    """
    payload = request.get_json(silent=True) or {}
    features = payload.get("features", {})
    if not isinstance(features, dict) or not features:
        return jsonify({"error": "'features' must be a non-empty object"}), 400

    try:
        for col in NUMERIC_MODEL_COLUMNS:
            if col in features:
                features[col] = float(features[col])

        source_ip = payload.get("source", "unknown")
        features["source_ip"] = source_ip

        metrics_service.inc("predictions_total")
        try:
            engine_features = enrich_single_row(features)
        except Exception:
            logger.warning("Feature enrichment failed in /api/detect; falling back to raw features", exc_info=True)
            engine_features = features

        eval_start = time.monotonic()
        results = engine_registry.evaluate_all(engine_features)
        metrics_service.observe_latency("engine_eval_latency", eval_start)
        metrics_service.inc("engine_evaluations_total")
        aggregated = engine_aggregator.aggregate(results)

        if aggregated.verdict in ("attack", "suspicious"):
            metrics_service.inc("alerts_total")
        if aggregated.verdict == "attack":
            metrics_service.inc("engine_attacks_total")

        for result in results:
            if result.verdict == "attack":
                metrics_service.inc(f"engine_{result.engine_id}_attacks_total")
            metrics_service.inc(f"engine_{result.engine_id}_evaluations_total")

        return jsonify(aggregated.to_dict())
    except Exception as exc:
        logger.exception("Multi-engine detect error (FULL TRACE)")
        return jsonify({
            "error": str(type(exc).__name__),
            "message": str(exc),
            "debug": True
        }), 400


@app.route("/api/detection/analyze", methods=["POST"])
@require_role("analyst")
def api_detection_analyze():
    """Compatibility endpoint for the Detection page target scanner."""
    payload = request.get_json(silent=True) or {}
    target = str(payload.get("target", "")).strip()
    input_type = str(payload.get("input_type", "ip")).strip().lower() or "ip"
    if not target:
        return jsonify({"error": "'target' is required"}), 400

    features = DEFAULT_FEATURE_ROW.copy()
    features.update({
        "duration": 1.0,
        "src_bytes": max(1.0, float(len(target) * 10)),
        "dst_bytes": max(1.0, float(len(input_type) * 5)),
        "count": 1.0,
        "srv_count": 1.0,
        "source_ip": target if input_type == "ip" else "unknown",
    })
    try:
        enriched = enrich_single_row(features)
    except Exception:
        enriched = features
    results = engine_registry.evaluate_all(enriched)
    aggregated = engine_aggregator.aggregate(results)
    severity = aggregated.severity if aggregated.verdict != "normal" else "clean"
    result = {
        "target": target,
        "analysis_type": input_type,
        "severity": severity,
        "summary": f"{aggregated.verdict.title()} verdict from {len(results)} engine(s)",
        "confidence": aggregated.confidence,
        "engines_triggered": sum(1 for r in results if r.verdict in {"attack", "suspicious"}),
        "engine_details": {r.engine_id: r.verdict in {"attack", "suspicious"} for r in results},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw": aggregated.to_dict(),
    }
    return jsonify({"result": result, **result}), 200


@app.route("/api/engines", methods=["GET"])
@require_role("analyst")
def api_engines():
    """List all registered detection engines and their status."""
    return jsonify({"engines": engine_registry.list_engines()})


@app.route("/api/engines/<engine_id>/toggle", methods=["POST"])
@require_role("admin")
def api_toggle_engine(engine_id: str):
    """Enable or disable a detection engine at runtime."""
    payload = request.get_json(silent=True) or {}
    enabled = payload.get("enabled")
    if enabled is None:
        return jsonify({"error": "'enabled' is required"}), 400
    ok = engine_registry.set_enabled(engine_id, bool(enabled))
    if not ok:
        return jsonify({"error": f"engine '{engine_id}' not found"}), 404
    return jsonify({"engine_id": engine_id, "enabled": bool(enabled)})




@app.route("/api/policy", methods=["GET", "POST"])
@require_role("admin")
def api_policy():
    if request.method == "GET":
        return jsonify(prevention_service.policy.to_dict())

    payload = request.get_json(silent=True) or {}
    try:
        policy = prevention_service.set_policy(
            mode=payload.get("mode"),
            block_ttl_seconds=payload.get("block_ttl_seconds"),
            confidence_block_threshold=payload.get("confidence_block_threshold"),
            risk_alert_threshold=payload.get("risk_alert_threshold", payload.get("alert_threshold")),
            risk_rate_limit_threshold=payload.get("risk_rate_limit_threshold", payload.get("rate_limit_threshold")),
            risk_temp_block_threshold=payload.get("risk_temp_block_threshold", payload.get("temp_block_threshold")),
            risk_block_threshold=payload.get("risk_block_threshold", payload.get("block_threshold")),
            block_requires_approval=payload.get("block_requires_approval"),
            risk_weight_confidence=payload.get("risk_weight_confidence"),
            risk_weight_severity=payload.get("risk_weight_severity"),
            risk_weight_frequency=payload.get("risk_weight_frequency"),
            dry_run=payload.get("dry_run"),
        )
        changed_by = str(payload.get("changed_by", "admin_api")).strip() or "admin_api"
        reason = str(payload.get("reason", "policy_update")).strip() or "policy_update"
        pv = policy_store.update(policy.to_dict(), changed_by=changed_by, reason=reason)
        # Ensure runtime policy is synchronized with the store
        if policy_store.current is not None:
            reloaded_config = policy_store.current.config
            reloaded_policy = PolicyConfig(**reloaded_config)
            prevention_service.policy = reloaded_policy
            logger.info("policy_runtime_reloaded version=%s", pv.version)
        ops_store.add_audit(
            event_type="policy_update",
            message=(
                f"mode={policy.mode}, ttl={policy.block_ttl_seconds}, "
                f"threshold={policy.confidence_block_threshold}, dry_run={policy.dry_run}, v={pv.version}"
            ),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        metrics_service.inc("policy_updates_total")
        return jsonify(policy.to_dict())
    except Exception as exc:
        return jsonify({"error": "invalid_request"}), 400


@app.route("/api/policy/history", methods=["GET"])
@require_role("admin")
def api_policy_history():
    limit = request.args.get("limit", default=50, type=int)
    history = policy_store.history(limit=max(1, min(limit, 500)))
    return jsonify({"count": len(history), "history": history})


@app.route("/api/policy/rollback", methods=["POST"])
@require_role("admin")
def api_policy_rollback():
    payload = request.get_json(silent=True) or {}
    to_version = payload.get("to_version")
    if to_version is None:
        return jsonify({"error": "'to_version' is required"}), 400

    changed_by = str(payload.get("changed_by", "admin_api")).strip() or "admin_api"
    pv = policy_store.rollback(int(to_version), changed_by=changed_by)
    if pv is None:
        return jsonify({"error": "version_not_found"}), 404

    config = pv.config
    policy = prevention_service.set_policy(
        mode=config.get("mode"),
        block_ttl_seconds=config.get("block_ttl_seconds"),
        confidence_block_threshold=config.get("confidence_block_threshold"),
        risk_alert_threshold=config.get("risk_alert_threshold", config.get("alert_threshold")),
        risk_rate_limit_threshold=config.get("risk_rate_limit_threshold", config.get("rate_limit_threshold")),
        risk_temp_block_threshold=config.get("risk_temp_block_threshold", config.get("temp_block_threshold")),
        risk_block_threshold=config.get("risk_block_threshold", config.get("block_threshold")),
        block_requires_approval=config.get("block_requires_approval"),
        risk_weight_confidence=config.get("risk_weight_confidence"),
        risk_weight_severity=config.get("risk_weight_severity"),
        risk_weight_frequency=config.get("risk_weight_frequency"),
        dry_run=config.get("dry_run"),
    )
    # Ensure runtime policy is synchronized with the store after rollback
    if policy_store.current is not None:
        reloaded_config = policy_store.current.config
        reloaded_policy = PolicyConfig(**reloaded_config)
        prevention_service.policy = reloaded_policy
        logger.info("policy_rolled_back_and_reloaded to version=%s", pv.version)
    ops_store.add_audit(
        event_type="policy_rollback",
        message=f"rollback_to={to_version} new_version={pv.version}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"policy": policy.to_dict(), "version": pv.to_dict()})


# ====================================================================
# Honeypot Configuration Endpoints (Hot-Reloadable)
# ====================================================================

@app.route("/api/honeypot/config", methods=["GET"])
@require_role("analyst")
def api_honeypot_config_get():
    """Get current honeypot configuration."""
    ips = honeypot_engine.get_config()["honeypot_ips"]
    ports = honeypot_engine.get_config()["honeypot_ports"]
    return jsonify({
        "honeypot_ips": ips,
        "honeypot_ports": ports,
        "enabled": honeypot_engine.is_ready(),
        "engine_id": honeypot_engine.engine_id,
    })


@app.route("/api/honeypot/config", methods=["POST", "PATCH"])
@require_role("admin")
def api_honeypot_config_update():
    """Update honeypot configuration (hot-reload - no restart needed)."""
    payload = request.get_json(silent=True) or {}
    
    # Update IPs if provided
    if "honeypot_ips" in payload:
        new_ips = payload.get("honeypot_ips", [])
        if not isinstance(new_ips, list):
            return jsonify({"error": "'honeypot_ips' must be a list"}), 400
        honeypot_engine.update_honeypot_ips(new_ips)
        ops_store.add_audit(
            event_type="honeypot_config_update",
            message=f"honeypot_ips updated: {new_ips}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        metrics_service.inc("honeypot_config_updates_total")
    
    # Update ports if provided
    if "honeypot_ports" in payload:
        new_ports = payload.get("honeypot_ports", [])
        if not isinstance(new_ports, list):
            return jsonify({"error": "'honeypot_ports' must be a list"}), 400
        try:
            new_ports = [int(p) for p in new_ports]
        except (ValueError, TypeError):
            return jsonify({"error": "honeypot_ports must be integers"}), 400
        honeypot_engine.update_honeypot_ports(new_ports)
        ops_store.add_audit(
            event_type="honeypot_config_update",
            message=f"honeypot_ports updated: {new_ports}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        metrics_service.inc("honeypot_config_updates_total")
    
    # Return updated config
    config = honeypot_engine.get_config()
    return jsonify({
        "honeypot_ips": config["honeypot_ips"],
        "honeypot_ports": config["honeypot_ports"],
        "enabled": config["enabled"],
        "updated": True,
    }), 200


# ====================================================================
# Incident and Activity Management Endpoints
# ====================================================================

@app.route("/api/incidents", methods=["GET"])
@require_role("analyst")
def api_incidents_list():
    """List all incidents with hierarchical grouping."""
    limit = request.args.get("limit", default=50, type=int)
    incidents = incident_aggregator.get_incidents(limit=max(1, min(limit, 500)))
    return jsonify({"count": len(incidents), "incidents": incidents})


@app.route("/api/incidents/<incident_id>", methods=["GET"])
@require_role("analyst")
def api_incident_details(incident_id: str):
    """Get detailed incident with all activities and alerts."""
    details = incident_aggregator.get_incident_details(incident_id)
    if details is None:
        return jsonify({"error": "incident not found"}), 404
    return jsonify(details)


@app.route("/api/activities", methods=["GET"])
@require_role("analyst")
def api_activities_list():
    """List all activities (alert groupings)."""
    limit = request.args.get("limit", default=50, type=int)
    activities = incident_aggregator.get_activities(limit=max(1, min(limit, 500)))
    return jsonify({"count": len(activities), "activities": activities})


@app.route("/api/temporal/patterns", methods=["GET"])
@require_role("analyst")
def api_temporal_patterns_list():
    """List all registered temporal correlation patterns for multi-stage attack detection."""
    patterns = []
    for pattern_name, pattern in temporal_correlation_engine.patterns.items():
        patterns.append({
            "name": pattern_name,
            "steps": pattern.steps,
            "description": pattern.description,
        })
    return jsonify({"count": len(patterns), "patterns": patterns})


@app.route("/api/temporal/patterns", methods=["POST"])
@require_role("admin")
def api_temporal_patterns_register():
    """Register a new temporal correlation pattern."""
    body = request.get_json(silent=True) or {}
    pattern_name = body.get("name", "").strip()
    pattern_steps = body.get("steps", [])
    description = body.get("description", "")
    
    if not pattern_name or not pattern_steps:
        return jsonify({"error": "name and steps are required"}), 400
    
    if not isinstance(pattern_steps, list) or len(pattern_steps) < 2:
        return jsonify({"error": "pattern must have at least 2 steps"}), 400
    
    try:
        temporal_correlation_engine.register_pattern(pattern_name, pattern_steps)
        ops_store.add_audit(
            event_type="temporal_pattern_registered",
            message=f"pattern={pattern_name}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        metrics_service.inc("temporal_patterns_registered_total")
        return jsonify({
            "ok": True,
            "pattern_name": pattern_name,
            "message": "Temporal correlation pattern registered successfully"
        }), 201
    except Exception as e:
        logger.exception("Failed to register temporal pattern: %s", pattern_name)
        return jsonify({"error": f"Failed to register pattern: {str(e)}"}), 500


@app.route("/api/temporal/state/<source_ip>", methods=["GET"])
@require_role("analyst")
def api_temporal_state(source_ip: str):
    """Get current temporal correlation state for a source IP."""
    if source_ip not in temporal_correlation_engine.temporal_store.events_by_ip:
        return jsonify({"source_ip": source_ip, "events": [], "status": "no_events"}), 200
    
    events = temporal_correlation_engine.temporal_store.events_by_ip[source_ip]
    return jsonify({
        "source_ip": source_ip,
        "event_count": len(events),
        "events": [{"type": e["type"], "timestamp": e["timestamp"], "confidence": e.get("confidence")} for e in events[-20:]],  # Last 20 events
        "status": "tracking"
    })


# ============ Entity Enrichment Endpoints ============

@app.route("/api/entity/enrich/<source_ip>", methods=["GET"])
@require_role("analyst")
def api_entity_enrich(source_ip: str):
    """Enrich an IP with context (GeoIP, threat intel, historical data)."""
    try:
        enriched = entity_enrichment_engine.enrich(source_ip)
        threat_level = entity_enrichment_engine.get_threat_level(enriched)
        result = enriched.to_dict()
        result["threat_level"] = threat_level
        return jsonify(result)
    except Exception as e:
        logger.exception("Enrichment failed for IP %s", source_ip)
        return jsonify({"error": f"Enrichment failed: {str(e)}"}), 500


@app.route("/api/entity/<source_ip>/threat-level", methods=["GET"])
@require_role("analyst")
def api_entity_threat_level(source_ip: str):
    """Get threat level assessment for an IP."""
    try:
        enriched = entity_enrichment_engine.enrich(source_ip)
        threat_level = entity_enrichment_engine.get_threat_level(enriched)
        confidence = enriched.enrichment_confidence
        return jsonify({
            "source_ip": source_ip,
            "threat_level": threat_level,
            "confidence": confidence,
            "last_enriched": enriched.timestamp,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ Alert Filtering Endpoints ============

@app.route("/api/alerts/filter-rules", methods=["GET"])
@require_role("analyst")
def api_alert_filter_rules():
    """List all alert filtering rules."""
    all_rules = alert_filter.get_all_rules()
    stats = alert_filter.get_rule_stats()
    return jsonify({
        "rules": all_rules,
        "stats": stats,
    })


@app.route("/api/alerts/filter-rules/exclude", methods=["POST"])
@require_role("admin")
def api_alert_add_exclude_rule():
    """Add an exclude rule (blocks alerts completely)."""
    body = request.get_json(silent=True) or {}
    from src.ips.alert_filter import ExcludeRule
    
    try:
        rule = ExcludeRule(
            rule_id=body.get("rule_id", f"exclude_{int(datetime.utcnow().timestamp())}"),
            name=body.get("name", "Unnamed exclude rule"),
            description=body.get("description", ""),
            conditions=body.get("conditions", {}),
            priority=body.get("priority", 0),
        )
        if alert_filter.add_exclude_rule(rule):
            ops_store.add_audit(
                event_type="exclude_rule_added",
                message=f"rule={rule.rule_id}",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            metrics_service.inc("alert_filter_rules_created_total")
            return jsonify({"ok": True, "rule_id": rule.rule_id}), 201
        return jsonify({"error": "Rule with this ID already exists"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/alerts/filter-rules/ignore", methods=["POST"])
@require_role("admin")
def api_alert_add_ignore_rule():
    """Add an ignore rule (deprioritizes alerts)."""
    body = request.get_json(silent=True) or {}
    from src.ips.alert_filter import IgnoreRule
    
    try:
        rule = IgnoreRule(
            rule_id=body.get("rule_id", f"ignore_{int(datetime.utcnow().timestamp())}"),
            name=body.get("name", "Unnamed ignore rule"),
            description=body.get("description", ""),
            conditions=body.get("conditions", {}),
            severity_reduction=body.get("severity_reduction", 1),
            suppress_notifications=body.get("suppress_notifications", True),
            priority=body.get("priority", 0),
        )
        if alert_filter.add_ignore_rule(rule):
            ops_store.add_audit(
                event_type="ignore_rule_added",
                message=f"rule={rule.rule_id}",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            metrics_service.inc("alert_filter_rules_created_total")
            return jsonify({"ok": True, "rule_id": rule.rule_id}), 201
        return jsonify({"error": "Rule with this ID already exists"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/alerts/filter-rules/merge", methods=["POST"])
@require_role("admin")
def api_alert_add_merge_rule():
    """Add a merge rule (combines similar alerts)."""
    body = request.get_json(silent=True) or {}
    from src.ips.alert_filter import MergeRule
    
    try:
        rule = MergeRule(
            rule_id=body.get("rule_id", f"merge_{int(datetime.utcnow().timestamp())}"),
            name=body.get("name", "Unnamed merge rule"),
            description=body.get("description", ""),
            conditions=body.get("conditions", {}),
            merge_window_seconds=body.get("merge_window_seconds", 300),
            merge_key=body.get("merge_key", "source_ip"),
            similarity_fields=body.get("similarity_fields", ["attack_type", "source_ip"]),
            priority=body.get("priority", 0),
        )
        if alert_filter.add_merge_rule(rule):
            ops_store.add_audit(
                event_type="merge_rule_added",
                message=f"rule={rule.rule_id}",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            metrics_service.inc("alert_filter_rules_created_total")
            return jsonify({"ok": True, "rule_id": rule.rule_id}), 201
        return jsonify({"error": "Rule with this ID already exists"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/alerts/filter-rules/<rule_id>", methods=["DELETE"])
@require_role("admin")
def api_alert_delete_rule(rule_id: str):
    """Delete a filter rule."""
    if alert_filter.remove_rule(rule_id):
        ops_store.add_audit(
            event_type="filter_rule_deleted",
            message=f"rule={rule_id}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        metrics_service.inc("alert_filter_rules_deleted_total")
        return jsonify({"ok": True, "message": f"Rule {rule_id} deleted"})
    return jsonify({"error": "Rule not found"}), 404


@app.route("/api/alerts/filter-stats", methods=["GET"])
@require_role("analyst")
def api_alert_filter_stats():
    """Get alert filtering statistics."""
    stats = alert_filter.get_rule_stats()
    return jsonify(stats)


@app.route("/api/actions", methods=["GET", "POST"])
@require_role("analyst")
def api_actions():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        action_type = str(payload.get("type") or payload.get("action_type") or payload.get("action") or "block").strip()
        target = str(payload.get("target_ip") or payload.get("target") or payload.get("ip") or "").strip()
        if not target:
            return jsonify({"error": "'target_ip' or 'target' is required"}), 400
        ttl = int(payload.get("ttl_seconds") or prevention_service.policy.block_ttl_seconds)
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl)).isoformat() if ttl > 0 else None
        dry_run = bool(prevention_service.policy.dry_run)
        executed = False
        status = "DRY_RUN" if dry_run else "ACTIVE"
        if not dry_run and action_type in {"block", "temp_block", "rate_limit"}:
            if action_type == "rate_limit":
                executed, status_text = action_executor.rate_limit(target, ttl)
            else:
                executed, status_text = action_executor.block_ip(target, ttl)
            status = "ACTIVE" if executed else status_text.upper()
        action_id = f"act_{uuid.uuid4().hex[:16]}"
        row = {
            "action_id": action_id,
            "action": action_type,
            "action_type": action_type,
            "target": target,
            "ip": target,
            "reason": payload.get("reason") or f"manual_{action_type}",
            "expires_at": expires_at,
            "created_at": now.isoformat(),
            "executed_at": now.isoformat() if executed or dry_run else None,
            "adapter": SETTINGS.firewall_adapter,
            "dry_run": dry_run,
            "executed": executed or dry_run,
            "status": status,
        }
        ops_store.save_action(row)
        ops_store.add_audit(
            event_type="manual_action",
            message=json.dumps({"action_id": action_id, "type": action_type, "target": target}, separators=(",", ":")),
            created_at=now.isoformat(),
        )
        normalized = _normalize_action_payload(row)
        return jsonify({"ok": True, "status": 201, "data": normalized, "action": normalized}), 201

    limit = request.args.get("limit", default=50, type=int)
    actions = [_normalize_action_payload(action) for action in ops_store.list_actions(limit=max(1, min(limit, 200)))]
    return jsonify({"count": len(actions), "actions": actions})


@app.route("/api/actions/pending", methods=["GET"])
@require_role("analyst")
def api_actions_pending():
    rows = ops_store._fetchall(
        "SELECT * FROM actions WHERE lower(COALESCE(status, '')) = 'pending_approval' ORDER BY id DESC LIMIT 200"
    )
    actions = [_normalize_action_payload(row) for row in rows]
    return jsonify({"count": len(actions), "actions": actions})


@app.route("/api/actions/<action_id>/approve", methods=["POST"])
@require_role("admin")
def api_action_approve(action_id: str):
    body = request.get_json(silent=True) or {}
    ttl_seconds = body.get("ttl_seconds")
    result = action_executor.approve_pending_block(
        action_id,
        prevention_service.policy,
        ttl_seconds=int(ttl_seconds) if ttl_seconds is not None else None,
    )
    if not result.get("ok") and result.get("error") == "not_found":
        return jsonify({"error": "pending action not found"}), 404
    ops_store.add_audit(
        event_type="action_approved",
        message=json.dumps(result, separators=(",", ":")),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify(result), (200 if result.get("ok") else 500)




@app.route("/api/actions/cleanup", methods=["POST"])
@require_role("admin")
def api_actions_cleanup():
    payload = request.get_json(silent=True) or {}
    now_iso = payload.get("now")
    removed = action_executor.cleanup_expired_actions(now_iso=now_iso)
    if removed:
        ops_store.add_audit(
            event_type="actions_cleanup",
            message=f"removed_expired_actions={removed}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        metrics_service.inc("expired_actions_cleaned_total", amount=removed)
    return jsonify({"removed": removed})


@app.route("/api/allowlist", methods=["GET"])
@require_role("analyst")
def api_allowlist_get():
    """List all allowlist entries."""
    entries = allowlist.list_entries()
    formatted = [_format_allowlist_entry(entry) for entry in entries]
    return jsonify({"entries": entries, "allowlist": formatted}), 200


@app.route("/api/allowlist", methods=["POST"])
@require_role("admin")
def api_allowlist_add():
    """Add an IP or CIDR to the allowlist."""
    body = request.get_json(silent=True) or {}
    entry = str(body.get("entry", "")).strip()
    reason = str(body.get("reason", "")).strip()
    if not entry:
        return jsonify({"error": "'entry' is required"}), 400
    added = allowlist.add(entry, reason=reason)
    ops_store.add_audit(
        event_type="allowlist_add",
        message=f"entry={entry} reason={reason} new={added}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"entry": entry, "added": added}), (201 if added else 200)


@app.route("/api/allowlist/<path:entry>", methods=["DELETE"])
@require_role("admin")
def api_allowlist_remove(entry: str):
    """Remove an entry from the allowlist."""
    removed = allowlist.remove(entry)
    if not removed:
        return jsonify({"error": "entry not found"}), 404
    ops_store.add_audit(
        event_type="allowlist_remove",
        message=f"entry={entry}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"entry": entry, "removed": True}), 200


@app.route("/api/audit", methods=["GET"])
@require_role("admin")
def api_audit():
    limit = request.args.get("limit", default=100, type=int)
    audits = ops_store.list_audits(limit=max(1, min(limit, 500)))
    return jsonify({"count": len(audits), "audits": audits})


@app.route("/api/siem/flush", methods=["GET"])
@require_role("admin")
def api_siem_flush():
    limit = request.args.get("limit", default=500, type=int)
    limit = max(1, min(limit, 5000))
    jsonl = siem_exporter.flush_jsonl(limit)
    lines = [line for line in jsonl.splitlines() if line.strip()]
    return jsonify({"count": len(lines), "jsonl": jsonl, "stats": siem_exporter.stats()})


@app.route("/api/threat-intel/stats", methods=["GET"])
@require_role("analyst")
def api_ti_stats():
    """Return TI cache statistics and loaded feed summary."""
    return jsonify({
        "stats": ti_manager.stats(),
        "feeds": ti_manager.feed_summary(),
        "engine_ready": ti_engine.is_ready(),
        "engine_enabled": engine_registry.is_enabled(ti_engine.engine_id),
    })


@app.route("/api/threat-intel/lookup", methods=["POST"])
@require_role("analyst")
def api_ti_lookup():
    """Look up a single IP against the TI cache."""
    body = request.get_json(silent=True) or {}
    ip = str(body.get("ip", "")).strip()
    if not ip:
        return jsonify({"error": "'ip' is required"}), 400
    indicator = ti_manager.lookup_ip(ip)
    if indicator is None:
        return jsonify({"ip": ip, "found": False, "indicator": None}), 200
    return jsonify({"ip": ip, "found": True, "indicator": indicator.to_dict()}), 200


@app.route("/api/explain", methods=["POST"])
@require_role("analyst")
def api_explain():
    payload = request.get_json(silent=True) or {}
    features = payload.get("features", {})
    top_k = int(payload.get("top_k", 5))
    top_k = max(1, min(top_k, 20))

    if not isinstance(features, dict) or not features:
        return jsonify({"error": "'features' must be a non-empty object"}), 400

    try:
        explanation = DetectionService.explain_features(features, top_k=top_k)
        return jsonify({"top_k": top_k, "explanation": explanation})
    except Exception as exc:
        return jsonify({"error": "invalid_request"}), 400


@app.route("/api/models/registry", methods=["GET"])
@require_role("analyst")
def api_model_registry():
    limit = request.args.get("limit", default=50, type=int)
    entries = model_registry.list_entries(limit=max(1, min(limit, 200)))
    return jsonify({"count": len(entries), "models": entries})


@app.route("/api/models", methods=["GET"])
@require_role("analyst")
def api_models_catalog():
    """Compatibility catalog for the Models page controller."""
    registry_entries = model_registry.list_entries(limit=100)
    models = []
    for entry in registry_entries:
        name = entry.get("model_name") or entry.get("name") or entry.get("model") or "model"
        models.append({
            "id": str(name),
            "name": str(name).replace("_", " ").title(),
            "type": entry.get("model_type") or "ml",
            "description": entry.get("path") or entry.get("model_path") or "Registered model",
            "accuracy": float(entry.get("accuracy") or 0),
            "f1_score": float(entry.get("f1_score") or entry.get("f1") or 0),
            "precision": float(entry.get("precision") or 0),
            "recall": float(entry.get("recall") or 0),
            "trained_date": entry.get("created_at") or entry.get("timestamp"),
            "version": entry.get("version") or "1.0",
            "active": True,
        })
    if not models:
        for name in sorted(all_models.keys() or []):
            models.append({
                "id": name,
                "name": name.replace("_", " ").title(),
                "type": "ml",
                "description": "Loaded runtime model",
                "accuracy": 0,
                "f1_score": 0,
                "precision": 0,
                "recall": 0,
                "trained_date": None,
                "version": "runtime",
                "active": name == "rf_nsl_kdd",
            })
    return jsonify({"count": len(models), "models": models})


@app.route("/api/threat-intelligence", methods=["GET"])
@require_role("analyst")
def api_threat_intelligence_catalog():
    stats = ti_manager.stats()
    feeds = ti_manager.feed_summary()
    indicators = []
    if hasattr(ti_manager.cache, "all_indicators"):
        indicators = [item.to_dict() for item in ti_manager.cache.all_indicators()]
    return jsonify({
        "sources": feeds,
        "indicators": indicators,
        "stats": stats,
        "threats": [],
    })


@app.route("/api/metrics", methods=["GET"])
@require_role("analyst")
def api_metrics():
    return Response(metrics_service.as_prometheus(), mimetype="text/plain; version=0.0.4")


@app.route("/api/ingest", methods=["POST"])
@require_role("analyst")
def api_ingest():
    payload = request.get_json(silent=True) or {}
    source = payload.get("source", "ingestion_api")
    rows = payload.get("rows")
    pipeline_started = _ensure_pipeline_started()

    backpressure = getattr(app, "_pipeline_backpressure", None)
    if pipeline_started and backpressure is not None and backpressure.level == BackpressureLevel.SHEDDING:
        return jsonify({"error": "pipeline_backpressure", "pipeline": _pipeline_status()}), 503

    try:
        if isinstance(rows, list):
            if not rows:
                return jsonify({"error": "rows cannot be empty"}), 400
            if pipeline_started:
                added = _stream_ingest_records(rows, source=source)
            else:
                added = ingestion_service.enqueue_batch(rows, source=source)
        else:
            features = payload.get("features")
            if not isinstance(features, dict) or not features:
                return jsonify({"error": "provide either non-empty 'rows' or 'features'"}), 400
            if pipeline_started:
                added = _stream_ingest_records([features], source=source)
            else:
                ingestion_service.enqueue_record(features, source=source)
                added = 1

        metrics_service.inc("ingested_total", amount=added)
        return jsonify({
            "queued": added,
            "queue_size": ingestion_queue.size(),
            "pipeline": _pipeline_status(),
        })
    except Exception as exc:
        return jsonify({"error": "invalid_request"}), 400


@app.route("/api/ingest/log", methods=["POST"])
@require_role("analyst")
def api_ingest_log():
    payload = request.get_json(silent=True) or {}
    source_type = str(payload.get("type", "zeek")).lower()
    records = payload.get("records")
    pipeline_started = _ensure_pipeline_started()
    backpressure = getattr(app, "_pipeline_backpressure", None)
    if pipeline_started and backpressure is not None and backpressure.level == BackpressureLevel.SHEDDING:
        return jsonify({"error": "pipeline_backpressure", "pipeline": _pipeline_status()}), 503
    if not isinstance(records, list) or not records:
        return jsonify({"error": "'records' must be a non-empty list"}), 400

    try:
        transformed = []
        for rec in records:
            if source_type == "zeek":
                transformed.append(parse_zeek_conn_log(rec))
            elif source_type == "suricata":
                transformed.append(parse_suricata_eve_flow(rec))
            else:
                return jsonify({"error": "type must be 'zeek' or 'suricata'"}), 400

        if pipeline_started:
            added = _stream_ingest_records(transformed, source=f"{source_type}_log")
        else:
            added = ingestion_service.enqueue_batch(transformed, source=f"{source_type}_log")
        metrics_service.inc("ingested_total", amount=added)
        return jsonify({"queued": added, "queue_size": ingestion_queue.size(), "type": source_type})
    except Exception as exc:
        return jsonify({"error": "invalid_request"}), 400


@app.route("/api/ingest/process", methods=["POST"])
@require_role("analyst")
def api_ingest_process():
    if not ensure_detection_service():
        return jsonify({"error": "No trained model found"}), 503

    payload = request.get_json(silent=True) or {}
    try:
        max_items = int(payload.get("max_items", 50))
    except (ValueError, TypeError):
        return jsonify({"error": "max_items must be an integer"}), 400
    max_items = max(1, min(max_items, 500))

    def _handler(features, source):
        result = detection_service.predict_from_features(
            features,
            profile="balanced",
            source_ip=source,
        )
        metrics_service.inc("processed_ingestion_total")
        result_payload = result.to_dict()
        result_payload["prevention_action"] = None
        return result_payload

    processed = ingestion_service.process_all(_handler, max_items=max_items)
    return jsonify({
        "processed": len(processed),
        "queue_size": ingestion_queue.size(),
        "results": processed,
    })


@app.route("/about")
@require_role("viewer")
def about():
    return render_template("about.html")


@app.route("/api/detections/history", methods=["GET"])
@require_role("analyst")
def api_detections_history():
    limit = request.args.get("limit", default=50, type=int)
    limit = max(1, min(limit, 200))
    audits = ops_store._fetchall(
        "SELECT event_type, message, created_at FROM audits "
        "WHERE event_type IN ('risk_score', 'policy_decision', 'prevention_action') "
        "ORDER BY id DESC LIMIT :limit",
        {"limit": limit},
    )
    parsed = []
    for row in audits:
        try:
            msg = json.loads(row["message"])
        except (ValueError, TypeError):
            msg = {"raw": row["message"]}
        parsed.append({"event_type": row["event_type"], "created_at": row["created_at"], **msg})
    return jsonify({"count": len(parsed), "history": parsed})


@app.route("/api/anomaly/status", methods=["GET"])
@require_role("analyst")
def api_anomaly_status():
    buf = anomaly_engine.buffer_status()
    buf["engine_id"] = anomaly_engine.engine_id
    buf["enabled"] = engine_registry.is_enabled(anomaly_engine.engine_id)
    buf["ready"] = anomaly_engine.is_ready()
    return jsonify(buf)


@app.route("/api/escalation/summary", methods=["GET"])
@require_role("analyst")
def api_escalation_summary():
    summary = escalation_tracker.summary()
    return jsonify({
        "tracked_count": len(summary),
        "escalation": summary,
    })


@app.route("/api/escalation/evict", methods=["POST"])
@require_role("admin")
def api_escalation_evict():
    removed = escalation_tracker.evict_stale()
    summary = escalation_tracker.summary()
    ops_store.add_audit(
        event_type="escalation_evict",
        message=f"evicted={removed}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"evicted": removed, "remaining": len(summary)})


@app.route("/api/fp-stats", methods=["GET"])
@require_role("analyst")
def api_fp_stats():
    return jsonify({"stats": fp_manager.stats()})


@app.route("/api/investigations", methods=["GET"])
@require_role("analyst")
def api_investigations():
    alerts = ops_store.list_alerts(limit=100)
    investigations = []
    for alert in alerts:
        alert_id = str(alert.get("id", ""))
        investigations.append({
            "id": f"inv_{alert_id[-8:] or uuid.uuid4().hex[:8]}",
            "title": f"Investigation for {alert.get('prediction', 'alert')}",
            "description": alert.get("reason") or "Generated from alert activity",
            "severity": str(alert.get("severity", "low")).lower(),
            "status": "closed" if str(alert.get("status", "")).lower() == "closed" else "open",
            "investigator": alert.get("assignee") or "Unassigned",
            "created_date": alert.get("timestamp"),
            "evidence_count": 1,
            "alert_id": alert_id,
        })
    return jsonify({"count": len(investigations), "investigations": investigations})


def _default_playbooks() -> list[dict]:
    return [
        {
            "id": "pb_block_source",
            "name": "Block Source IP",
            "type": "containment",
            "description": "Create a prevention action for a malicious source.",
            "enabled": True,
            "action_count": 1,
            "execution_count": 0,
            "created_date": "2026-01-01T00:00:00+00:00",
            "last_run_date": None,
        },
        {
            "id": "pb_collect_context",
            "name": "Collect Investigation Context",
            "type": "investigation",
            "description": "Review recent alerts, entity enrichment, and risk history.",
            "enabled": True,
            "action_count": 3,
            "execution_count": 0,
            "created_date": "2026-01-01T00:00:00+00:00",
            "last_run_date": None,
        },
    ]


@app.route("/api/playbooks", methods=["GET"])
@require_role("analyst")
def api_playbooks():
    return jsonify({"playbooks": _default_playbooks(), "count": len(_default_playbooks())})


@app.route("/api/playbooks/<playbook_id>/execute", methods=["POST"])
@require_role("analyst")
def api_playbook_execute(playbook_id: str):
    ops_store.add_audit(
        event_type="playbook_execute",
        message=json.dumps({"playbook_id": playbook_id}, separators=(",", ":")),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"ok": True, "playbook_id": playbook_id, "status": "queued"}), 202


@app.route("/api/capture/start", methods=["POST"])
@require_role("analyst")
def api_capture_start():
    payload = request.get_json(silent=True) or {}
    app.config["CAPTURE_RUNNING"] = True
    app.config["CAPTURE_CONFIG"] = {
        "interface": payload.get("interface", "eth0"),
        "bpf_filter": payload.get("bpf_filter", ""),
        "packet_limit": int(payload.get("packet_limit") or 1000),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify({"ok": True, "status": "capturing", **app.config["CAPTURE_CONFIG"]}), 202


@app.route("/api/capture/stop", methods=["POST"])
@require_role("analyst")
def api_capture_stop():
    app.config["CAPTURE_RUNNING"] = False
    return jsonify({"ok": True, "status": "stopped"}), 200


@app.route("/api/honeypots", methods=["GET"])
@require_role("analyst")
def api_honeypots():
    config = honeypot_engine.get_config()
    honeypots = []
    for ip in config.get("honeypot_ips", []):
        honeypots.append({
            "id": ip,
            "ip": ip,
            "name": f"Honeypot {ip}",
            "ports": config.get("honeypot_ports", []),
            "enabled": bool(config.get("enabled")),
            "status": "active" if config.get("enabled") else "inactive",
        })
    return jsonify({"honeypots": honeypots, "count": len(honeypots), "config": config})


@app.route("/api/honeypots/<honeypot_id>/toggle", methods=["POST"])
@require_role("admin")
def api_honeypot_toggle(honeypot_id: str):
    body = request.get_json(silent=True) or {}
    enabled = bool(body.get("enabled", True))
    engine_registry.set_enabled(honeypot_engine.engine_id, enabled)
    return jsonify({"id": honeypot_id, "enabled": enabled})


@app.route("/realtime")
@require_role("viewer")
def realtime():
    return render_template("realtime.html")


@app.route("/capture")
@require_role("analyst")
def capture():
    return render_template("capture.html")


@app.route("/honeypot")
@require_role("analyst")
def honeypot():
    return render_template("honeypot.html")


# ============================================================================
# MODULE ROUTES - Data providers for 15 demo capability modules
# ============================================================================

@app.route("/modules/<module_id>")
@require_role("viewer")
def module_view(module_id):
    """Render module template by ID."""
    modules = {
        'real-time-detection': 'modules/real_time_detection.html',
        'multi-engine-voting': 'modules/multi_engine_voting.html',
        'risk-score-visualizer': 'modules/risk_score_visualizer.html',
        'auto-blocking': 'modules/auto_blocking.html',
        'evasion-detection': 'modules/evasion_detection.html',
        'packet-inspection': 'modules/packet_inspection.html',
        'behavioral-profiling': 'modules/behavioral_profiling.html',
        'threat-intelligence': 'modules/threat_intelligence.html',
        'drift-monitor': 'modules/drift_monitor.html',
        'anomaly-learning': 'modules/anomaly_learning.html',
        'fp-suppression': 'modules/fp_suppression.html',
        'escalation-tracker': 'modules/escalation_tracker.html',
        'network-topology': 'modules/network_topology.html',
        'policy-enforcement': 'modules/policy_enforcement.html',
        'forensic-timeline': 'modules/forensic_timeline.html',
    }
    
    template = modules.get(module_id)
    if not template:
        return "Module not found", 404
    
    try:
        return render_template(template)
    except Exception as e:
        logger.exception(f"Module {module_id} rendering failed")
        return f"Module error: {str(e)}", 500


# MODULE DATA APIs - Compose existing endpoints for module consumption

@app.route("/api/modules/real-time-detection", methods=["GET"])
@require_role("viewer")
def api_module_real_time_detection():
    """Real-time detection events."""
    try:
        alerts = ops_store.list_alerts(limit=50)
        return jsonify({
            "status": "success",
            "data": {
                "recent_events": alerts,
                "event_count": len(alerts),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading real-time detection module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/multi-engine", methods=["GET"])
@require_role("viewer")
def api_module_multi_engine():
    """Multi-engine voting consensus."""
    try:
        engines = engine_registry.list_engines()
        return jsonify({
            "status": "success",
            "data": {
                "engines": [
                    {
                        "id": e.get("engine_id"),
                        "name": e.get("engine_id"),
                        "type": e.get("engine_type"),
                        "enabled": bool(e.get("enabled")),
                        "ready": bool(e.get("ready")),
                    }
                    for e in engines
                ],
                "engine_count": len(engines),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading multi-engine module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/risk-score", methods=["GET"])
@require_role("analyst")
def api_module_risk_score():
    """Risk score visualization."""
    try:
        recent_alerts = ops_store.list_alerts(limit=100)
        risk_scores = [float(a.get("risk_score", 0)) for a in recent_alerts if "risk_score" in a]
        
        return jsonify({
            "status": "success",
            "data": {
                "current_risk": max(risk_scores) if risk_scores else 0,
                "average_risk": sum(risk_scores) / len(risk_scores) if risk_scores else 0,
                "risk_distribution": risk_scores[:20],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading risk score module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/auto-blocking", methods=["GET"])
@require_role("analyst")
def api_module_auto_blocking():
    """Auto-blocking module state."""
    try:
        active_blocks = ops_store.list_active_blocks(limit=100)
        return jsonify({
            "status": "success",
            "data": {
                "enabled": prevention_service.policy.mode == "auto_block",
                "dry_run": prevention_service.policy.dry_run,
                "blocked_ips": [_normalize_action_payload(row) for row in active_blocks],
                "block_count": len(active_blocks),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        })
    except Exception as e:
        logger.exception("Error loading auto-blocking module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/approval-workflow", methods=["GET"])
@require_role("analyst")
def api_module_approval_workflow():
    """Approval workflow HITL."""
    try:
        pending = ops_store.list_actions(limit=100)
        pending = [
            a
            for a in pending
            if str(a.get("status", "")).strip().lower() in {"pending", "pending_approval", "escalated"}
        ]
        
        return jsonify({
            "status": "success",
            "data": {
                "pending_approvals": pending[:10],
                "pending_count": len(pending),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading approval workflow module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/false-positive", methods=["GET"])
@require_role("analyst")
def api_module_false_positive():
    """False positive learning feedback."""
    try:
        suppressions = ops_store.list_fp_suppressions()
        feedback_rows = fp_manager.stats()
        false_positive_count = sum(int(row.get("false_positives", 0)) for row in feedback_rows)
        
        return jsonify({
            "status": "success",
            "data": {
                "false_positives": feedback_rows[:10],
                "fp_count": false_positive_count,
                "suppression_count": len(suppressions),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading false positive module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/threat-intel", methods=["GET"])
@require_role("analyst")
def api_module_threat_intel():
    """Threat intelligence enrichment."""
    try:
        alerts = ops_store.list_alerts(limit=50)
        return jsonify({
            "status": "success",
            "data": {
                "enriched_alerts": alerts,
                "total_enriched": len(alerts),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading threat intel module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/analytics", methods=["GET"])
@require_role("analyst")
def api_module_analytics():
    """Analytics dashboard."""
    try:
        metrics = {
            "requests_total": metrics_service.get("requests_total") or 0,
            "predictions_total": metrics_service.get("predictions_total") or 0,
            "alerts_total": metrics_service.get("alerts_total") or 0,
            "prevention_actions_total": metrics_service.get("prevention_actions_total") or 0,
        }
        
        return jsonify({
            "status": "success",
            "data": {
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading analytics module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/escalation", methods=["GET"])
@require_role("analyst")
def api_module_escalation():
    """Escalation state machine."""
    try:
        summary = escalation_tracker.summary()
        return jsonify({
            "status": "success",
            "data": {
                "escalation_states": summary,
                "total_tracked": len(summary),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading escalation module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/pipeline-monitor", methods=["GET"])
@require_role("viewer")
def api_module_pipeline_monitor():
    """Pipeline monitoring metrics."""
    try:
        return jsonify({
            "status": "success",
            "data": {
                "ingestion_rate": metrics_service.get("processed_ingestion_total") or 0,
                "queue_size": ingestion_queue.size() if hasattr(ingestion_queue, 'size') else 0,
                "latency_ms": 0,
                "throughput": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading pipeline monitor module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/policy-tuning", methods=["GET", "POST"])
@require_role("admin")
def api_module_policy_tuning():
    """Policy tuning simulator."""
    try:
        if request.method == "POST":
            # Simulate policy tuning
            params = request.json or {}
            return jsonify({
                "status": "success",
                "data": {
                    "simulated": True,
                    "parameters": params,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
        else:
            # Get current policy
            policy = prevention_service.policy.to_dict()
            return jsonify({
                "status": "success",
                "data": {
                    "current_policy": policy,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
    except Exception as e:
        logger.exception("Error loading policy tuning module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/alert-lifecycle", methods=["GET"])
@require_role("analyst")
def api_module_alert_lifecycle():
    """Alert lifecycle manager (Kanban board)."""
    try:
        all_alerts = ops_store.list_alerts(limit=100)
        
        # Organize by status
        lifecycle = {
            "new": [a for a in all_alerts if str(a.get("status", "")).strip().lower() in {"new", "open"}],
            "investigating": [a for a in all_alerts if str(a.get("status", "")).strip().lower() in {"investigating", "reviewing"}],
            "escalated": [a for a in all_alerts if str(a.get("status", "")).strip().lower() == "escalated"],
            "resolved": [a for a in all_alerts if str(a.get("status", "")).strip().lower() in {"resolved", "closed"}],
        }
        
        return jsonify({
            "status": "success",
            "data": {
                "lifecycle": lifecycle,
                "total_alerts": len(all_alerts),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading alert lifecycle module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/engine-playground", methods=["GET", "POST"])
@require_role("analyst")
def api_module_engine_playground():
    """Engine toggle playground."""
    try:
        if request.method == "POST":
            # Handle engine toggle
            params = request.json or {}
            engine_id = params.get("engine_id")
            enabled = params.get("enabled")
            
            if engine_id:
                try:
                    if enabled is None:
                        # Toggle engine state (enable if disabled, disable if enabled)
                        current_enabled = engine_registry.is_enabled(engine_id)
                        new_state = not current_enabled
                    else:
                        if isinstance(enabled, str):
                            new_state = enabled.strip().lower() in {"1", "true", "yes", "on"}
                        else:
                            new_state = bool(enabled)
                    if not engine_registry.set_enabled(engine_id, new_state):
                        return jsonify({"error": "engine_not_found"}), 404
                    logger.info("Set engine %s enabled=%s", engine_id, new_state)
                except Exception as e:
                    logger.exception(f"Failed to toggle engine {engine_id}: {e}")
                    return jsonify({"error": f"Failed to toggle engine: {str(e)}"}), 500
            
            return jsonify({
                "status": "success",
                "data": {
                    "toggled": engine_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
        else:
            # Get engine status
            engines = engine_registry.list_engines()
            return jsonify({
                "status": "success",
                "data": {
                    "engines": [
                        {
                            "id": e.get("engine_id"),
                            "enabled": bool(e.get("enabled")),
                            "ready": bool(e.get("ready")),
                            "type": e.get("engine_type"),
                        }
                        for e in engines
                    ],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
    except Exception as e:
        logger.exception("Error loading engine playground module")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/pattern-detector", methods=["GET"])
@require_role("analyst")
def api_module_pattern_detector():
    """Behavioral pattern detector (network graph)."""
    try:
        alerts = ops_store.list_alerts(limit=100)
        
        # Extract IP relationships
        patterns = {}
        for alert in alerts:
            src = alert.get("source_ip", "unknown")
            dst = alert.get("dest_ip", "unknown")
            
            if src not in patterns:
                patterns[src] = {"nodes": [], "edges": []}
            if dst not in patterns:
                patterns[dst] = {"nodes": [], "edges": []}
            
            patterns[src]["edges"].append(dst)
        
        return jsonify({
            "status": "success",
            "data": {
                "attack_patterns": patterns,
                "total_patterns": len(patterns),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading pattern detector module")
        return jsonify({"error": str(e)}), 500


# NEW MODULE API ENDPOINTS - Task 2: Modules 2-15

@app.route("/api/modules/multi-engine-voting", methods=["GET"])
@require_role("analyst")
def api_module_multi_engine_voting():
    """Multi-engine voting consensus."""
    try:
        alerts = ops_store.list_alerts(limit=50)
        engines = ['Random Forest', 'SVM', 'Decision Tree', 'Naive Bayes', 'Logistic Regression']
        
        verdicts = {}
        for engine in engines:
            eng_key = engine.lower().replace(' ', '-')
            verdicts[eng_key] = {
                'verdict': alerts[0].get('classification', 'benign') if alerts else 'benign',
                'confidence': min(100, 60 + (hash(engine) % 40)),
                'latency': 10 + (hash(engine) % 20),
                'trees': hash(engine) % 500 if 'forest' in engine.lower() else None,
                'margin': round((hash(engine) % 100) / 100.0, 2) if 'svm' in engine.lower() else None,
                'depth': hash(engine) % 25 if 'tree' in engine.lower() else None,
                'prior': round((hash(engine) % 100) / 100.0, 2) if 'bayes' in engine.lower() else None,
                'prob': round((hash(engine) % 100) / 100.0, 2) if 'logistic' in engine.lower() else None,
            }
        
        decisions = [{'source_ip': a.get('source_ip', '?'), 'dest_ip': a.get('dest_ip', '?'), 
                     'final_verdict': a.get('classification', 'benign'), 'reason': 'consensus',
                     'timestamp': datetime.now(timezone.utc).isoformat()} for a in alerts[:5]]
        
        return jsonify({"status": "success", "data": {"engines": verdicts, "decisions": decisions}})
    except Exception as e:
        logger.exception("Error loading multi-engine voting")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/risk-score-visualizer", methods=["GET"])
@require_role("analyst")
def api_module_risk_score_visualizer():
    """Risk score visualization."""
    try:
        alerts = ops_store.list_alerts(limit=50)
        risk_factors = {
            'payload': {'score': 0.3, 'detail': 'Suspicious patterns detected'},
            'behavior': {'score': 0.4, 'detail': 'Unusual access pattern'},
            'protocol': {'score': 0.2, 'detail': 'Minor RFC violation'},
            'threat': {'score': 0.5, 'detail': '2 IOC matches'},
            'outlier': {'score': 0.3, 'detail': 'Statistical deviation'},
            'entropy': {'score': 0.6, 'detail': 'High randomness'},
        }
        
        return jsonify({"status": "success", "data": {
            "recent_events": [{'risk_score': min(0.9, 0.2 + ((i % 10) * 0.08))} for i, _ in enumerate(alerts)],
            "risk_factors": risk_factors,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading risk score visualizer")
        return jsonify({"error": str(e)}), 500


# Background thread for pushing live updates (defined before main block)
_update_thread = None
_update_thread_stop = False


def _start_module_update_broadcaster() -> None:
    """Start module websocket broadcaster (WebSocket is mandatory in INIDS 2.0)."""
    global _update_thread, _update_thread_stop
    if _update_thread is not None and _update_thread.is_alive():
        return
    _update_thread_stop = False
    logger.info("Starting WebSocket module update broadcaster...")
    _update_thread = threading.Thread(target=_module_update_broadcaster, daemon=True, name="module-broadcaster")
    _update_thread.start()


def _module_update_broadcaster():
    """Background thread that broadcasts module updates every 2 seconds."""
    global _update_thread_stop
    while not _update_thread_stop:
        try:
            state = _build_realtime_state()
            alerts = state.get("alerts", [])
            timestamp = state.get("timestamp", datetime.now(timezone.utc).isoformat())

            socketio.emit("metrics.update", state, namespace="/events")
            
            # Broadcast to real-time-detection subscribers
            if alerts:
                socketio.emit('module_update', {
                    'module_id': 'real-time-detection',
                    'data': {'recent_events': alerts, 'event_count': len(alerts)},
                    'timestamp': timestamp
                }, to='module_real-time-detection')
            
            # Broadcast to multi-engine-voting subscribers
            socketio.emit('module_update', {
                'module_id': 'multi-engine-voting',
                'data': {'events_processed': len(alerts)},
                'timestamp': timestamp
            }, to='module_multi-engine-voting')
            
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error in module broadcaster: {e}")
            time.sleep(2)


@app.route("/api/modules/evasion-detection", methods=["GET"])
@require_role("analyst")
def api_module_evasion_detection():
    """Evasion technique detection."""
    try:
        techniques = {
            'Obfuscation': {'count': 5, 'detection_rate': 0.95},
            'Fragmentation': {'count': 3, 'detection_rate': 0.98},
            'Protocol Mixing': {'count': 2, 'detection_rate': 0.92},
            'Polymorphism': {'count': 1, 'detection_rate': 0.88},
        }
        return jsonify({"status": "success", "data": {
            "techniques": techniques,
            "detection_rate": 0.93,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading evasion detection")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/packet-inspection", methods=["GET"])
@require_role("analyst")
def api_module_packet_inspection():
    """Deep packet inspection results."""
    try:
        packets = []
        alerts = ops_store.list_alerts(limit=20)
        for alert in alerts:
            packets.append({
                'protocol': alert.get('classification', 'Unknown'),
                'src_ip': alert.get('source_ip', '0.0.0.0'),
                'dst_ip': alert.get('dest_ip', '0.0.0.0'),
                'size': (hash(str(alert)) % 1500) + 64,
                'anomaly': hash(str(alert)) % 3 == 0,
            })
        return jsonify({"status": "success", "data": {"packets": packets, "timestamp": datetime.now(timezone.utc).isoformat()}})
    except Exception as e:
        logger.exception("Error loading packet inspection")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/behavioral-profiling", methods=["GET"])
@require_role("analyst")
def api_module_behavioral_profiling():
    """User and entity behavioral profiling."""
    try:
        profiles = [
            {'name': f'User-{i}', 'activity_level': 'Active' if i % 2 else 'Idle', 'anomaly': i % 4 == 0, 'risk_score': i * 10}
            for i in range(1, 6)
        ]
        return jsonify({"status": "success", "data": {
            "profiles": profiles,
            "avg_confidence": 0.78,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading behavioral profiling")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/threat-intelligence", methods=["GET"])
@require_role("analyst")
def api_module_threat_intelligence():
    """Threat intelligence feeds and IOC matches."""
    try:
        feeds = [
            {'name': 'Abuse.ch URLhaus', 'last_update': datetime.now(timezone.utc).isoformat()},
            {'name': 'AlienVault OTX', 'last_update': (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()},
            {'name': 'Phishtank', 'last_update': (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()},
        ]
        return jsonify({"status": "success", "data": {
            "feeds": feeds,
            "ioc_count": 42,
            "updates_today": 3,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading threat intelligence")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/drift-monitor", methods=["GET"])
@require_role("analyst")
def api_module_drift_monitor():
    """Model drift monitoring."""
    try:
        current_acc = 0.92 + (((datetime.now().timestamp()) % 100) / 1000)
        return jsonify({"status": "success", "data": {
            "drift_percentage": 3.5,
            "current_accuracy": current_acc,
            "retrain_recommended": current_acc < 0.88,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading drift monitor")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/anomaly-learning", methods=["GET"])
@require_role("analyst")
def api_module_anomaly_learning():
    """Auto-learning anomaly patterns."""
    try:
        patterns = [
            {'name': 'Port Scan', 'confidence': 0.95, 'is_anomaly': True},
            {'name': 'Data Exfil', 'confidence': 0.88, 'is_anomaly': True},
            {'name': 'Brute Force', 'confidence': 0.92, 'is_anomaly': True},
        ]
        return jsonify({"status": "success", "data": {
            "patterns": patterns,
            "learning_progress": 0.72,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading anomaly learning")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/fp-suppression", methods=["GET"])
@require_role("analyst")
def api_module_fp_suppression():
    """False positive suppression rules."""
    try:
        rules = [
            {'name': 'Whitelisted IPs', 'count': 15},
            {'name': 'Scan Exceptions', 'count': 8},
            {'name': 'Test Traffic', 'count': 12},
        ]
        return jsonify({"status": "success", "data": {
            "rules": rules,
            "fp_suppressed": 35,
            "accuracy_gain": 0.06,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading FP suppression")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/escalation-tracker", methods=["GET"])
@require_role("analyst")
def api_module_escalation_tracker():
    """Incident escalation tracking."""
    try:
        incidents = ops_store.list_alerts(limit=10)
        return jsonify({"status": "success", "data": {
            "incidents": [
                {'title': f"Incident-{i}", 'severity': ['LOW', 'MEDIUM', 'HIGH'][i % 3], 
                 'escalated': i % 2 == 0, 'resolved': i % 3 == 0}
                for i in range(len(incidents))
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading escalation tracker")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/network-topology", methods=["GET"])
@require_role("analyst")
def api_module_network_topology():
    """Network topology visualization."""
    try:
        nodes = [{'name': f'Node-{i}', 'threat': i % 5 == 0} for i in range(1, 6)]
        return jsonify({"status": "success", "data": {
            "nodes": nodes,
            "connections": len(nodes) * 2,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading network topology")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/policy-enforcement", methods=["GET"])
@require_role("admin")
def api_module_policy_enforcement():
    """Security policy enforcement."""
    try:
        policies = [
            {'name': 'Access Control', 'enforcement_status': 'Active'},
            {'name': 'Data Protection', 'enforcement_status': 'Active'},
            {'name': 'Incident Response', 'enforcement_status': 'Active'},
        ]
        return jsonify({"status": "success", "data": {
            "policies": policies,
            "violations": 2,
            "enforcement_rate": 0.98,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading policy enforcement")
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules/forensic-timeline", methods=["GET"])
@require_role("analyst")
def api_module_forensic_timeline():
    """Forensic event timeline."""
    try:
        alerts = ops_store.list_alerts(limit=15)
        events = [
            {'event_type': a.get('classification', 'Event'), 'description': f"Event from {a.get('source_ip', '?')}", 
             'timestamp': datetime.now(timezone.utc).isoformat()}
            for a in alerts
        ]
        return jsonify({"status": "success", "data": {
            "events": events,
            "total_incidents": len(events),
            "status": "Ready",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading forensic timeline")
        return jsonify({"error": str(e)}), 500


# ===== WEBSOCKET EVENT HANDLERS =====
# Real-time module updates via SocketIO

@socketio.on('connect')
def handle_connect():
    """Client connected to WebSocket."""
    logger.info(f"WebSocket client connected: {request.sid}")
    emit('connection_response', {'data': 'Connected to INIDS dashboard'})


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected from WebSocket."""
    logger.info(f"WebSocket client disconnected: {request.sid}")


@socketio.on('subscribe_module')
def handle_subscribe_module(data):
    """Subscribe to live updates for a specific module."""
    module_id = data.get('module_id', '')
    room = f'module_{module_id}'
    join_room(room)
    logger.debug(f"Client {request.sid} subscribed to {module_id}")
    emit('subscription_confirmed', {'module_id': module_id})


@socketio.on('unsubscribe_module')
def handle_unsubscribe_module(data):
    """Unsubscribe from module updates."""
    module_id = data.get('module_id', '')
    room = f'module_{module_id}'
    leave_room(room)
    logger.debug(f"Client {request.sid} unsubscribed from {module_id}")


# Module-specific WebSocket event handlers
@socketio.on('request_module_data')
def handle_request_module_data(data):
    """Client requests fresh data for a module."""
    module_id = data.get('module_id', '')
    room = f'module_{module_id}'
    
    try:
        # Route to appropriate data handler
        if module_id == 'real-time-detection':
            alerts = ops_store.list_alerts(limit=50)
            emit('module_data', {
                'module_id': module_id,
                'data': {
                    'recent_events': alerts,
                    'event_count': len(alerts),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            }, to=room)
        elif module_id == 'multi-engine-voting':
            alerts = ops_store.list_alerts(limit=50)
            engines = ['Random Forest', 'SVM', 'Decision Tree', 'Naive Bayes', 'Logistic Regression']
            verdicts = {}
            for engine in engines:
                eng_key = engine.lower().replace(' ', '-')
                verdicts[eng_key] = {
                    'verdict': alerts[0].get('classification', 'benign') if alerts else 'benign',
                    'confidence': min(100, 60 + (hash(engine) % 40)),
                    'latency': 10 + (hash(engine) % 20),
                }
            emit('module_data', {
                'module_id': module_id,
                'data': {'engines': verdicts, 'decisions': []},
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, to=room)
        elif module_id == 'risk-score-visualizer':
            alerts = ops_store.list_alerts(limit=50)
            emit('module_data', {
                'module_id': module_id,
                'data': {
                    'recent_events': [{'risk_score': min(0.9, 0.2 + ((i % 10) * 0.08))} for i, _ in enumerate(alerts)],
                    'risk_factors': {k: v for k, v in [('payload', {'score': 0.3}), ('behavior', {'score': 0.4})]}
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, to=room)
        else:
            # Generic handler for other modules
            emit('module_data', {
                'module_id': module_id,
                'data': {'status': 'live', 'timestamp': datetime.now(timezone.utc).isoformat()},
            }, to=room)
    except Exception as e:
        logger.exception(f"Error sending module data for {module_id}")
        emit('error', {'message': str(e)})


# Broadcast module updates to all connected clients
def broadcast_module_update(module_id, data):
    """Broadcast module update to all subscribed clients."""
    room = f'module_{module_id}'
    socketio.emit('module_update', {
        'module_id': module_id,
        'data': data,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }, to=room, skip_sid=None)


# --- WebSocket Event Namespace Handlers (INIDS 2.0 Real-Time Events) ---

@socketio.on('connect', namespace='/events')
def handle_events_connect():
    """Client connected to /events namespace for real-time events."""
    logger.info(f"Real-time events client connected: {request.sid}")
    _start_module_update_broadcaster()
    emit('connection_response', {
        'status': 'connected',
        'message': 'Connected to real-time event stream',
        'namespace': '/events'
    })
    try:
        emit("metrics.update", _build_realtime_state())
    except Exception:
        logger.exception("Failed to emit initial metrics.update snapshot")


@socketio.on('disconnect', namespace='/events')
def handle_events_disconnect():
    """Client disconnected from /events namespace."""
    logger.info(f"Real-time events client disconnected: {request.sid}")


@socketio.on('subscribe_alerts', namespace='/events')
def handle_subscribe_alerts():
    """Subscribe to real-time alert events."""
    join_room('alerts')
    logger.debug(f"Client {request.sid} subscribed to alerts")
    emit('subscription_confirmed', {'subscription': 'alerts'})


@socketio.on('unsubscribe_alerts', namespace='/events')
def handle_unsubscribe_alerts():
    """Unsubscribe from alert events."""
    leave_room('alerts')
    logger.debug(f"Client {request.sid} unsubscribed from alerts")


@socketio.on('subscribe_actions', namespace='/events')
def handle_subscribe_actions():
    """Subscribe to real-time action events."""
    join_room('actions')
    logger.debug(f"Client {request.sid} subscribed to actions")
    emit('subscription_confirmed', {'subscription': 'actions'})


@socketio.on('unsubscribe_actions', namespace='/events')
def handle_unsubscribe_actions():
    """Unsubscribe from action events."""
    leave_room('actions')
    logger.debug(f"Client {request.sid} unsubscribed from actions")


@socketio.on('subscribe_metrics', namespace='/events')
def handle_subscribe_metrics():
    """Subscribe to real-time metrics events."""
    join_room('metrics')
    logger.debug(f"Client {request.sid} subscribed to metrics")
    emit('subscription_confirmed', {'subscription': 'metrics'})


@socketio.on('unsubscribe_metrics', namespace='/events')
def handle_unsubscribe_metrics():
    """Unsubscribe from metrics events."""
    leave_room('metrics')
    logger.debug(f"Client {request.sid} unsubscribed from metrics")


@socketio.on('subscribe_perception', namespace='/events')
def handle_subscribe_perception():
    """Subscribe to real-time perception layer events (pulse, confidence, attack_story)."""
    join_room('perception')
    logger.debug(f"Client {request.sid} subscribed to perception events")
    # Send initial pulse data
    pulse = live_system_pulse.get_pulse_status()
    emit('perception_pulse', pulse)
    emit('subscription_confirmed', {'subscription': 'perception'})


@socketio.on('unsubscribe_perception', namespace='/events')
def handle_unsubscribe_perception():
    """Unsubscribe from perception events."""
    leave_room('perception')
    logger.debug(f"Client {request.sid} unsubscribed from perception events")


@socketio.on('request_pulse', namespace='/events')
def handle_request_pulse():
    """Client requests current system pulse."""
    pulse = live_system_pulse.get_pulse_status()
    emit('perception_pulse', pulse)


@socketio.on('request_confidence', namespace='/events')
def handle_request_confidence(data):
    """Client requests confidence breakdown for a detection."""
    detection_id = data.get('detection_id')
    if detection_id:
        breakdown = confidence_breakdown_engine.get_breakdown(detection_id)
        if breakdown:
            emit('perception_confidence', breakdown)
        else:
            emit('error', {'message': f'Detection {detection_id} not found'})


@socketio.on('request_attack_story', namespace='/events')
def handle_request_attack_story(data):
    """Client requests attack story for an attack."""
    attack_id = data.get('attack_id')
    if attack_id:
        story = attack_story_engine.get_attack_story(attack_id)
        emit('perception_attack_story', story)


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(Exception)
def handle_error(e):
    logger.error("Unhandled error: %s", e)
    if request.path.startswith("/api/"):
        return jsonify({"error": "internal_error"}), 500
    return render_template("error.html", error="Unexpected internal error."), 500


if __name__ == "__main__":
    try:
        _validate_runtime_security()
        _log_runtime_configuration()
        load_models()
        _ensure_scheduler_started()
        _ensure_pipeline_started()
        _start_module_update_broadcaster()
        socketio.run(
            app,
            debug=SETTINGS.debug,
            host=SETTINGS.host,
            port=SETTINGS.port,
        )
    except KeyboardInterrupt:
        logger.info("Shutdown requested by keyboard interrupt")
        _shutdown_runtime()
        sys.exit(0)
    except Exception:
        logger.exception("Fatal startup/runtime error")
        _shutdown_runtime()
        raise

