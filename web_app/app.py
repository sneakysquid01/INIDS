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
from datetime import datetime, timezone

# Support both `python -m web_app.app` and direct script runs (`python web_app/app.py`).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.settings import load_settings
from src.rate_limiter import InMemoryRateLimiter, RateLimitConfig
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

try:
    from flask_socketio import SocketIO
except Exception:
    SocketIO = None

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
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.detection_service import DetectionService, InMemoryAlertStore
from src.prevention_service import PreventionService
from src.ops_store import OpsStore
from src.auth_service import require_role, auth_status
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
from src.pipeline.backpressure import BackpressureController, BackpressureLevel
from src.pipeline.stream_processor import StreamProcessor
from src.pipeline.worker import PipelineWorker

app = Flask(__name__)
app.config["SECRET_KEY"] = SETTINGS.flask_secret_key
if SocketIO is not None:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    SOCKETIO_ENABLED = True
else:
    SOCKETIO_ENABLED = False

    class _NoopSocketIO:
        def emit(self, *args, **kwargs):
            return None

        def run(self, flask_app, **kwargs):
            flask_app.run(**kwargs)

    socketio = _NoopSocketIO()

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


model = None
all_models = {}
event_bus = EventBus()
alert_store = InMemoryAlertStore(max_items=1000)
detection_service = None
prevention_service = PreventionService(adapter=_build_firewall_adapter())
ops_store = OpsStore(OPS_DB_PATH)
allowlist = Allowlist(ops_store)
escalation_tracker = EscalationTracker(cooldown_seconds=300.0)
fp_manager = FalsePositiveManager()
metrics_service = MetricsService()
siem_exporter = SiemExporter()
ingestion_queue = InMemoryIngestionQueue(max_items=10000)
ingestion_service = IngestionService(queue=ingestion_queue)
model_registry = ModelRegistry(os.path.join(RESULTS_DIR, "model_registry.json"))
rate_limiter = InMemoryRateLimiter(
    RateLimitConfig(requests=SETTINGS.rate_limit_requests, window_seconds=SETTINGS.rate_limit_window_seconds)
)

# --- Multi-engine detection framework ---
engine_registry = EngineRegistry()
engine_aggregator = EngineAggregator(AggregationStrategy.ANY_TRIGGER)
RULES_PATH = os.path.join(BASE_DIR, "rules", "default_rules.yaml")
signature_engine = SignatureEngine(RULES_PATH if os.path.exists(RULES_PATH) else None)
threshold_engine = ThresholdEngine()
anomaly_engine = AnomalyEngine()

engine_registry.register(signature_engine)
engine_registry.register(threshold_engine)
engine_registry.register(anomaly_engine, enabled=False)  # enabled after fit()

# Threat intelligence engine — starts disabled until feeds are loaded.
# is_ready() returns True automatically once ti_manager.cache.size() > 0.
ti_manager = ThreatIntelManager()
ti_engine = TIEngine(ti_manager)
engine_registry.register(ti_engine)  # enabled=True; gated by is_ready() returning False when cache is empty

risk_engine = RiskEngine()
policy_engine = PolicyEngine()
policy_store = PolicyStore(initial_config=prevention_service.policy.to_dict())
action_executor = ActionExecutor(
    adapter=prevention_service.adapter,
    adapter_name=SETTINGS.firewall_adapter,
    ops_store=ops_store,
    event_bus=event_bus,
)
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
    except Exception:
        logger.warning("Redis unavailable for pipeline runtime; continuing without streaming", exc_info=True)
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
        except Exception:
            logger.exception("Failed to close Redis client")


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


def _on_detection_event(event: DetectionEvent) -> None:
    if allowlist.contains(event.source_ip):
        logger.info("Allowlist: bypassing prevention pipeline for %s", event.source_ip)
        return
    risk_event = risk_engine.calculate(event)
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
    decision_event = policy_engine.decide(event, prevention_service.policy)
    event_bus.publish(decision_event)
    try:
        ops_store.add_audit(
            event_type="policy_decision",
            message=json.dumps(
                {
                    "source_ip": event.detection.source_ip,
                    "prediction": event.detection.prediction,
                    "risk_score": event.risk_score,
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
    if str(event.decision).strip().upper() not in {"BLOCK", "TEMP_BLOCK", "RATE_LIMIT"}:
        return
    action = action_executor.execute(event, prevention_service.policy)
    if action is not None:
        event_bus.publish(action)
    escalation_tracker.record_hit(
        event.risk.detection.source_ip,
        event.risk.detection.severity,
    )


def _emit_realtime(event_name: str, payload: dict) -> None:
    if not SOCKETIO_ENABLED:
        return
    try:
        socketio.emit(event_name, payload, namespace="/events")
    except Exception:
        logger.exception("Failed to emit websocket event '%s'", event_name)


def _on_detection_realtime(event: DetectionEvent) -> None:
    metrics_service.inc("detection_events_total")
    _emit_realtime("DetectionEvent", event.to_dict())


def _on_risk_realtime(event: RiskScoreEvent) -> None:
    _emit_realtime("RiskScoreEvent", event.to_dict())


def _on_action_realtime(event: ActionEvent) -> None:
    metrics_service.inc("action_events_total")
    _emit_realtime("ActionEvent", event.to_dict())


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
        ops_store.list_alerts(limit=1)
        return {"ready": True}

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
        detection_service = DetectionService(model=model, alert_store=alert_store, event_bus=event_bus)
    return True


def _normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().rstrip(".")


def load_models():
    """Load all available models into memory for live or selectable prediction."""
    global model, all_models, detection_service
    if all_models:
        return
    model_files = ['rf_nsl_kdd.pkl', 'gb_nsl_kdd.pkl', 'dt_nsl_kdd.pkl',
                   'ab_nsl_kdd.pkl', 'mlp_nsl_kdd.pkl', 'rf_nsl_kdd_multi.pkl']
    for model_file in model_files:
        path = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(path):
            model_name = model_file.replace('.pkl', '')
            all_models[model_name] = joblib.load(path)
            logger.info("Loaded model %s", model_name)
    if 'rf_nsl_kdd' in all_models:
        model = all_models['rf_nsl_kdd']
        detection_service = DetectionService(model=model, alert_store=alert_store, event_bus=event_bus)
        # Register the primary ML model as a detection engine.
        ml_engine = MLEngine(model, engine_id="ml_primary")
        engine_registry.register(ml_engine)


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
        buf.close()
        plt.close(fig)

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
    return stats


def _safe_created_at(item: dict, fallback: str) -> str:
    value = item.get("created_at")
    return value if isinstance(value, str) and value else fallback


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
            client_key = f"{request.remote_addr or 'unknown'}:{request.path}"
            allowed, retry_after = rate_limiter.allow(client_key)
            if not allowed:
                metrics_service.inc('rate_limited_total')
                return jsonify({
                    "error": "rate_limited",
                    "retry_after_seconds": retry_after,
                }), 429


@app.after_request
def _after_request_metrics(response):
    if request.path.startswith('/api/') and response.status_code == 401:
        metrics_service.inc('unauthorized_total')
    return response


@app.route("/")
def home():
    """Landing page with navigation."""
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
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


@app.route("/dashboard")
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
@app.route("/models")
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
            df = pd.read_csv(file)
            if df.empty:
                return render_template("batch.html", error="Uploaded CSV is empty.")

            max_rows = 50000
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
        "alerts_buffered": len(alert_store.list_alerts(limit=1000)),
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


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not ensure_detection_service():
        return jsonify({"error": "No trained model found"}), 503

    payload = request.get_json(silent=True) or {}
    features = payload.get("features", {})
    profile = payload.get("profile", "balanced")

    if not isinstance(features, dict) or not features:
        return jsonify({"error": "'features' must be a non-empty object"}), 400

    try:
        for col in NUMERIC_MODEL_COLUMNS:
            if col in features:
                features[col] = float(features[col])
        source = payload.get("source", "unknown")
        metrics_service.inc("predictions_total")
        result = detection_service.predict_from_features(
            features,
            profile=profile,
            source_ip=source,
            attack_type=payload.get("attack_type"),
        )
        if result.alert:
            ops_store.save_alert(result.alert.to_dict())
            metrics_service.inc("alerts_total")

        action = prevention_service.evaluate(result.prediction, result.confidence, source=source)
        if action:
            ops_store.save_action(action.to_dict())
            metrics_service.inc("prevention_actions_total")
            ops_store.add_audit(
                event_type="prevention_action",
                message=f"{action.action} target={action.target} reason={action.reason}",
                created_at=datetime.now(timezone.utc).isoformat(),
            )

        response = result.to_dict()
        response["prevention_action"] = action.to_dict() if action else None
        return jsonify(response)
    except Exception as exc:
        logger.error("API predict error: %s", exc)
        return jsonify({"error": "invalid_request"}), 400


@app.route("/api/alerts", methods=["GET"])
@require_role("analyst")
def api_alerts():
    limit = request.args.get("limit", default=50, type=int)
    severity = request.args.get("severity", default=None, type=str)
    alerts = ops_store.list_alerts(limit=max(1, min(limit, 200)), severity=severity)
    return jsonify({"count": len(alerts), "alerts": alerts})


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
        logger.error("Multi-engine detect error: %s", exc)
        return jsonify({"error": "invalid_request"}), 400


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
            dry_run=payload.get("dry_run"),
        )
        changed_by = str(payload.get("changed_by", "admin_api")).strip() or "admin_api"
        reason = str(payload.get("reason", "policy_update")).strip() or "policy_update"
        pv = policy_store.update(policy.to_dict(), changed_by=changed_by, reason=reason)
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
        dry_run=config.get("dry_run"),
    )
    ops_store.add_audit(
        event_type="policy_rollback",
        message=f"rollback_to={to_version} new_version={pv.version}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"policy": policy.to_dict(), "version": pv.to_dict()})


@app.route("/api/actions", methods=["GET"])
@require_role("analyst")
def api_actions():
    limit = request.args.get("limit", default=50, type=int)
    actions = ops_store.list_actions(limit=max(1, min(limit, 200)))
    return jsonify({"count": len(actions), "actions": actions})




@app.route("/api/actions/cleanup", methods=["POST"])
@require_role("admin")
def api_actions_cleanup():
    payload = request.get_json(silent=True) or {}
    now_iso = payload.get("now")
    removed = ops_store.cleanup_expired_actions(now_iso=now_iso)
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
    return jsonify({"entries": allowlist.list_entries()}), 200


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
    max_items = int(payload.get("max_items", 50))
    max_items = max(1, min(max_items, 500))

    def _handler(features, source):
        result = detection_service.predict_from_features(
            features,
            profile="balanced",
            source_ip=source,
        )
        if result.alert:
            ops_store.save_alert(result.alert.to_dict())
            metrics_service.inc("alerts_total")

        action = prevention_service.evaluate(result.prediction, result.confidence, source=source)
        if action:
            ops_store.save_action(action.to_dict())
            metrics_service.inc("prevention_actions_total")
            ops_store.add_audit(
                event_type="prevention_action",
                message=f"{action.action} target={action.target} reason={action.reason}",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        metrics_service.inc("processed_ingestion_total")
        result_payload = result.to_dict()
        result_payload["prevention_action"] = action.to_dict() if action else None
        return result_payload

    processed = ingestion_service.process_all(_handler, max_items=max_items)
    return jsonify({
        "processed": len(processed),
        "queue_size": ingestion_queue.size(),
        "results": processed,
    })


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/realtime")
def realtime():
    return render_template("realtime.html", socketio_enabled=SOCKETIO_ENABLED)


@app.route("/capture")
def capture():
    return render_template("capture.html")


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

