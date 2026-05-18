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
from src.ha.health_check import HealthCheck
from src.ha.leader_election import LeaderElection
from src.input_sanitizer import SanitizationError, sanitize_id, sanitize_ip_address, sanitize_string
from src.correlation_tracing import correlation_id_middleware, get_correlation_id
from src.csrf_protection import csrf_protect_middleware, require_csrf_token

try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
except ImportError as e:
    raise ImportError(
        "flask_socketio is required for INIDS 2.0. WebSocket is mandatory. "
        "Install with: pip install flask-socketio python-socketio"
    ) from e

try:
    from flask_compress import Compress as _FlaskCompress
except ImportError:
    _FlaskCompress = None


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

from src.detection_service import DetectionService
from src.prevention_service import PreventionService
from src.ops_store import OpsStore
from src.auth.decorators import require_roles, AuthStoreUnboundError
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
from src.auth.jwt_manager import get_jwt_manager
from src.auth.auth_service import UnifiedAuthService
from src.validation_schemas import validate_predict_request, validate_detect_request, validate_policy
from src.elasticsearch_client import ElasticsearchConfig, ElasticsearchStore
from src.elasticsearch_audit_bridge import init_elasticsearch_audit_bridge, get_elasticsearch_audit_bridge
from src.async_utils import get_async_executor

_APP_START_TIME = time.time()

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
app.ops_store = ops_store  # F-AUTH-REMOVE: wired so require_roles() finds it via current_app
assert app.ops_store is not None, "ops_store failed to attach to app"
logger.info("auth.ops_store_bound=true")

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

@app.before_request
def _inject_request_context():
    action_executor.ops_store = ops_store
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

# FIX-017: Only register temporal engine when patterns are loaded; no-op registration
# wastes CPU on every pipeline event.
if temporal_correlation_engine.pattern_count() > 0:
    engine_registry.register(temporal_correlation_engine)
else:
    logger.info("engine.temporal.skipped reason=no_patterns")

# Entity context enrichment engine — enriches alerts with GeoIP, threat intel, history
entity_enrichment_engine = EntityEnrichmentEngine(
    ops_store=ops_store,
    ti_manager=ti_manager,
    internal_cidrs=SETTINGS.internal_cidrs or None
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
                    "id": str(uuid.uuid4()),
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
            from src._telemetry import anomaly_add_sample_errors
            anomaly_add_sample_errors.inc()
            logger.warning(
                "anomaly.add_sample_failed",
                exc_info=True,
                extra={"engine": "anomaly"},
            )

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
        "system_uptime": round(time.time() - _APP_START_TIME, 1),
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
    _start_alert_retention_thread()
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


def _run_alert_retention() -> int:
    """Delete alerts older than INIDS_ALERT_RETENTION_DAYS. Returns count deleted.

    D-08: Returns 0 and no-ops if INIDS_ALERT_RETENTION_DAYS is 0 or unset.
    Rollback: set INIDS_ALERT_RETENTION_DAYS=0 to disable immediately.
    WARNING: destructive — any alert deleted cannot be recovered without backup.
    """
    try:
        days_str = os.environ.get("INIDS_ALERT_RETENTION_DAYS", "0").strip()
        days = int(days_str) if days_str else 0
    except (ValueError, TypeError):
        days = 0
    if days <= 0:
        return 0  # retention disabled
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        deleted = ops_store.delete_alerts_older_than(cutoff)
        if deleted > 0:
            logger.info("alert_retention: deleted %d alerts older than %d days", deleted, days)
        return deleted
    except Exception:
        logger.exception("alert_retention: deletion failed")
        return 0


def _start_alert_retention_thread() -> None:
    """Start a daily daemon thread for the alert retention job (D-08)."""
    if getattr(app, "_alert_retention_started", False):
        return
    import threading as _threading

    def _retention_loop() -> None:
        while True:
            time.sleep(86400)  # run once per day
            try:
                if not leader_election.is_leader():
                    logger.info("retention.skipped reason=not_leader")
                    continue
                logger.info("retention.runs_total")
                _run_alert_retention()
            except Exception:
                logger.exception("Alert retention loop failed")

    t = _threading.Thread(target=_retention_loop, daemon=True, name="alert-retention")
    t.start()
    app._alert_retention_started = True
    logger.info("Alert retention thread started (INIDS_ALERT_RETENTION_DAYS=%s)",
                os.environ.get("INIDS_ALERT_RETENTION_DAYS", "0 (disabled)"))


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
        # D-07: health probe is read-only — no INSERT to audit log.
        # A test write would pollute the audit table with synthetic entries
        # and could mask genuine audit record volume in monitoring.
        try:
            ops_store.list_alerts(limit=1)
            ops_store.list_audits(limit=1)
            return {"ready": True}
        except Exception as exc:
            return {"ready": False, "error": str(exc), "note": "database_read_failed"}

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
    _validate_all_routes_have_auth_decorator()


# PLAN.md Phase A Step 3 (A-03): PUBLIC_ROUTES are exempt from auth decorator requirement.
# All other routes must have @require_roles (F-AUTH-REMOVE: legacy decorators removed).
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


def _validate_all_routes_have_auth_decorator() -> None:
    """Fail-closed startup check: every non-public route must have a require_roles decorator.

    F-AUTH-REMOVE: checks _required_roles attribute set by require_roles() on the
    original function; works with functools.wraps because @wraps copies __dict__.
    """
    uncovered = []
    for rule in app.url_map.iter_rules():
        if rule.rule in PUBLIC_ROUTES:
            continue
        if rule.rule.startswith("/static/"):
            continue
        view_func = app.view_functions.get(rule.endpoint)
        if view_func is None:
            continue
        # require_roles() sets _required_roles on the original func; @wraps copies __dict__
        covered = hasattr(view_func, "_required_roles")
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
            f"Add @require_roles() or mark as public: {uncovered}"
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


from web_app.blueprints.health import health_bp
from web_app.blueprints.auth import auth_bp
from web_app.blueprints.ingest import ingest_bp
from web_app.blueprints.observability import observability_bp
from web_app.blueprints.detection import detection_bp
from web_app.blueprints.pages import pages_bp
from web_app.blueprints.dashboard import dashboard_bp
from web_app.blueprints.prevention import prevention_bp
from web_app.blueprints.intel import intel_bp
from web_app.blueprints.system import system_bp
from web_app.blueprints.modules import modules_bp
app.register_blueprint(health_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(ingest_bp)
app.register_blueprint(observability_bp)
app.register_blueprint(detection_bp)
app.register_blueprint(pages_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(prevention_bp)
app.register_blueprint(intel_bp)
app.register_blueprint(system_bp)
app.register_blueprint(modules_bp)


@app.errorhandler(AuthStoreUnboundError)
def _handle_auth_unbound(e):
    return jsonify({"error": "service_unavailable", "reason": "auth_store_unbound"}), 503


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

