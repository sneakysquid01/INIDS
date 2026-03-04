<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
from __future__ import annotations

import atexit
import base64
import io
import json
import logging
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
from flask import Flask, Response, jsonify, render_template, request
import joblib
>>>>>>> theirs
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import joblib
import matplotlib
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
import pandas as pd
from flask import Flask, Response, current_app, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import logging
import json
from datetime import datetime, timezone
<<<<<<< ours
<<<<<<< ours
=======
=======
>>>>>>> theirs

from src.settings import load_settings
from src.rate_limiter import InMemoryRateLimiter, RateLimitConfig
from src.firewall_adapters import MockFirewallAdapter, UfwFirewallAdapter, NftablesFirewallAdapter
<<<<<<< ours
>>>>>>> theirs

from src.settings import load_settings
from src.rate_limiter import InMemoryRateLimiter, RateLimitConfig
from src.firewall_adapters import MockFirewallAdapter, UfwFirewallAdapter, NftablesFirewallAdapter
>>>>>>> theirs
=======
>>>>>>> theirs

matplotlib.use("Agg")
import matplotlib.pyplot as plt

<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.auth_service import auth_status, authorize_request, configure_auth, require_role
from src.detection_service import DetectionService, InMemoryAlertStore
from src.firewall_adapters import (
    IptablesFirewallAdapter,
    MockFirewallAdapter,
    NftablesFirewallAdapter,
    PfFirewallAdapter,
    UfwFirewallAdapter,
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
SETTINGS = load_settings()
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")
OPS_DB_PATH = SETTINGS.ops_db_path if os.path.isabs(SETTINGS.ops_db_path) else os.path.join(BASE_DIR, SETTINGS.ops_db_path)

from src.detection_service import DetectionService, InMemoryAlertStore
from src.prevention_service import PreventionService
from src.ops_store import OpsStore
from src.auth_service import require_role, auth_status
from src.metrics_service import MetricsService
from src.ingestion_service import InMemoryIngestionQueue, IngestionService
from src.log_parsers import parse_zeek_conn_log, parse_suricata_eve_flow
from src.model_registry import ModelRegistry
from src.schema import (
    COLUMNS,
    LABEL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    DEFAULT_FEATURE_ROW,
>>>>>>> theirs
)
from src.ingestion_service import InMemoryIngestionQueue, IngestionService
from src.ips.scheduler import PreventionScheduler
from src.log_parsers import parse_suricata_eve_flow, parse_zeek_conn_log
from src.logging_config import configure_logging
from src.metrics_service import MetricsService
from src.model_registry import ModelRegistry
from src.ops_store import OpsStore
from src.prevention_service import PreventionService
from src.rate_limiter import InMemoryRateLimiter, RateLimitConfig
from src.schema import COLUMNS, DEFAULT_FEATURE_ROW, FEATURE_COLUMNS, LABEL_COLUMNS, NUMERIC_FEATURES
from src.settings import Settings, load_settings

configure_logging()
logger = logging.getLogger(__name__)

<<<<<<< ours
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STATIC_DIR = os.path.join(BASE_DIR, "web_app", "static")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")

MODEL_FILES = [
    "rf_nsl_kdd.pkl",
    "gb_nsl_kdd.pkl",
    "dt_nsl_kdd.pkl",
    "ab_nsl_kdd.pkl",
    "mlp_nsl_kdd.pkl",
    "rf_nsl_kdd_multi.pkl",
]

INPUT_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    "serror_rate",
    "same_srv_rate",
=======
app = Flask(__name__)
app.config["SECRET_KEY"] = SETTINGS.flask_secret_key


def _build_firewall_adapter():
    adapter_name = SETTINGS.firewall_adapter
    if adapter_name == "ufw":
        return UfwFirewallAdapter()
    if adapter_name == "nftables":
        return NftablesFirewallAdapter()
    return MockFirewallAdapter()


model = None
all_models = {}
alert_store = InMemoryAlertStore(max_items=1000)
detection_service = None
prevention_service = PreventionService(adapter=_build_firewall_adapter())
ops_store = OpsStore(OPS_DB_PATH)
metrics_service = MetricsService()
ingestion_queue = InMemoryIngestionQueue(max_items=10000)
ingestion_service = IngestionService(queue=ingestion_queue)
model_registry = ModelRegistry(os.path.join(RESULTS_DIR, "model_registry.json"))
rate_limiter = InMemoryRateLimiter(
    RateLimitConfig(requests=SETTINGS.rate_limit_requests, window_seconds=SETTINGS.rate_limit_window_seconds)
)

INPUT_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "count",
    "srv_count", "serror_rate", "same_srv_rate",
>>>>>>> theirs
]

MODEL_INPUT_COLUMNS = FEATURE_COLUMNS
NUMERIC_MODEL_COLUMNS = NUMERIC_FEATURES


<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
@dataclass
class ServiceContainer:
    settings: Settings
    ops_db_path: str
    model: Any = None
    all_models: dict[str, Any] = field(default_factory=dict)
    alert_store: InMemoryAlertStore = field(default_factory=lambda: InMemoryAlertStore(max_items=1000))
    detection_service: DetectionService | None = None
    prevention_service: PreventionService | None = None
    ops_store: OpsStore | None = None
    metrics_service: MetricsService = field(default_factory=MetricsService)
    ingestion_queue: InMemoryIngestionQueue = field(default_factory=lambda: InMemoryIngestionQueue(max_items=10000))
    ingestion_service: IngestionService | None = None
    model_registry: ModelRegistry | None = None
    rate_limiter: InMemoryRateLimiter | None = None
    model_lock: Lock = field(default_factory=Lock)
    dashboard_cache_lock: Lock = field(default_factory=Lock)
    dashboard_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    scheduler: PreventionScheduler | None = None


def _build_firewall_adapter(adapter_name: str):
    if adapter_name == "iptables":
        return IptablesFirewallAdapter()
    if adapter_name == "pf":
        return PfFirewallAdapter()
    if adapter_name == "ufw":
        return UfwFirewallAdapter()
    if adapter_name == "nftables":
        return NftablesFirewallAdapter()
    return MockFirewallAdapter()


def _build_services(settings: Settings) -> ServiceContainer:
    ops_db_path = settings.ops_db_path if os.path.isabs(settings.ops_db_path) else os.path.join(BASE_DIR, settings.ops_db_path)

    configure_auth(
        admin_api_key=settings.admin_api_key,
        sensor_api_key=settings.sensor_api_key,
        viewer_api_key=settings.viewer_api_key,
    )

    ops_store = OpsStore(ops_db_path)
    rate_limiter = InMemoryRateLimiter(
        RateLimitConfig(
            requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
            max_keys=settings.rate_limit_max_keys,
        )
    )
    model_registry = ModelRegistry(os.path.join(RESULTS_DIR, "model_registry.json"))
    adapter = _build_firewall_adapter(settings.firewall_adapter)
    prevention_service = PreventionService(adapter=adapter, ops_store=ops_store)
    ingestion_queue = InMemoryIngestionQueue(max_items=10000)
    ingestion_service = IngestionService(queue=ingestion_queue)

    services = ServiceContainer(
        settings=settings,
        ops_db_path=ops_db_path,
        alert_store=InMemoryAlertStore(max_items=1000),
        prevention_service=prevention_service,
        ops_store=ops_store,
        ingestion_queue=ingestion_queue,
        ingestion_service=ingestion_service,
        model_registry=model_registry,
        rate_limiter=rate_limiter,
    )
    if settings.enable_ips_scheduler and prevention_service.action_executor is not None:
        scheduler = PreventionScheduler(
            prevention_service.action_executor,
            interval_seconds=settings.scheduler_interval_seconds,
            reconcile_every=settings.scheduler_reconcile_every,
        )
        scheduler.start()
        services.scheduler = scheduler
    return services


def create_app(settings: Settings | None = None) -> Flask:
    resolved_settings = settings or load_settings()
    flask_app = Flask(__name__)
    flask_app.config["SECRET_KEY"] = resolved_settings.flask_secret_key
    flask_app.config["DEBUG"] = resolved_settings.debug
    flask_app.extensions["services"] = _build_services(resolved_settings)
    return flask_app


SETTINGS = load_settings()
app = create_app(SETTINGS)


def _sync_globals_from_services(services: ServiceContainer) -> None:
    globals()["model"] = services.model
    globals()["all_models"] = services.all_models
    globals()["detection_service"] = services.detection_service
    globals()["alert_store"] = services.alert_store
    globals()["prevention_service"] = services.prevention_service
    globals()["ops_store"] = services.ops_store
    globals()["metrics_service"] = services.metrics_service
    globals()["ingestion_queue"] = services.ingestion_queue
    globals()["ingestion_service"] = services.ingestion_service
    globals()["model_registry"] = services.model_registry
    globals()["rate_limiter"] = services.rate_limiter
    globals()["OPS_DB_PATH"] = services.ops_db_path


def _sync_services_from_globals(services: ServiceContainer) -> None:
    alias_map = {
        "model": "model",
        "all_models": "all_models",
        "detection_service": "detection_service",
        "alert_store": "alert_store",
        "prevention_service": "prevention_service",
        "ops_store": "ops_store",
        "metrics_service": "metrics_service",
        "ingestion_queue": "ingestion_queue",
        "ingestion_service": "ingestion_service",
        "model_registry": "model_registry",
        "rate_limiter": "rate_limiter",
    }
    for global_name, field_name in alias_map.items():
        if global_name in globals():
            candidate = globals()[global_name]
            if candidate is not None and getattr(services, field_name) is not candidate:
                setattr(services, field_name, candidate)
    if "OPS_DB_PATH" in globals():
        services.ops_db_path = globals()["OPS_DB_PATH"]


def _services() -> ServiceContainer:
    services = current_app.extensions["services"]
    _sync_services_from_globals(services)
    return services


_sync_globals_from_services(app.extensions["services"])


@atexit.register
def _stop_scheduler() -> None:
    services = app.extensions.get("services")
    if services and services.scheduler is not None:
        services.scheduler.stop()


def _request_id() -> str:
    return getattr(g, "request_id", "unknown")


def _api_error(status_code: int, error: str, message: str | None = None):
    payload = {"error": error, "request_id": _request_id()}
    if message and (status_code < 500 or current_app.config.get("DEBUG", False)):
        payload["message"] = message
    return jsonify(payload), status_code


def _parse_bounded_int(value, *, name: str, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' must be an integer") from exc
    return max(minimum, min(parsed, maximum))
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
def ensure_model_loaded() -> None:
    """Lazily load models if not available in memory."""
    global model
    if model is None:
        load_models()
>>>>>>> theirs


def ensure_detection_service() -> bool:
    """Ensure detection service is initialized with loaded model."""
    global detection_service
    ensure_model_loaded()
    if model is None:
        return False
    if detection_service is None:
        detection_service = DetectionService(model=model, alert_store=alert_store)
    return True


def ensure_detection_service() -> bool:
    """Ensure detection service is initialized with loaded model."""
    global detection_service
    ensure_model_loaded()
    if model is None:
        return False
    if detection_service is None:
        detection_service = DetectionService(model=model, alert_store=alert_store)
    return True


def ensure_detection_service() -> bool:
    """Ensure detection service is initialized with loaded model."""
    global detection_service
    ensure_model_loaded()
    if model is None:
        return False
    if detection_service is None:
        detection_service = DetectionService(model=model, alert_store=alert_store)
    return True


def _normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().rstrip(".")


<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
def load_models() -> None:
    services = _services()
    with services.model_lock:
        if services.all_models:
            _sync_globals_from_services(services)
            return
        for model_file in MODEL_FILES:
            path = os.path.join(MODELS_DIR, model_file)
            if os.path.exists(path):
                model_name = model_file.replace(".pkl", "")
                services.all_models[model_name] = joblib.load(path)
                logger.info("model_loaded name=%s", model_name)
        if "rf_nsl_kdd" in services.all_models:
            services.model = services.all_models["rf_nsl_kdd"]
            services.detection_service = DetectionService(model=services.model, alert_store=services.alert_store)
    _sync_globals_from_services(services)


def ensure_model_loaded() -> None:
    services = _services()
    if services.model is None:
        load_models()


def ensure_detection_service() -> bool:
    services = _services()
    ensure_model_loaded()
    if services.model is None:
        return False
    if services.detection_service is None:
        services.detection_service = DetectionService(model=services.model, alert_store=services.alert_store)
    _sync_globals_from_services(services)
    return True


def _build_model_stats() -> dict[str, Any]:
    services = _services()
    model_stats = {
        "available": False,
        "error": None,
        "total": 0,
        "attacks": 0,
        "normal": 0,
        "accuracy": 0.0,
        "chart_data": None,
        "results": [],
    }
    if services.model is None:
        model_stats["error"] = "No trained model found. Train a model to unlock analytics."
        return model_stats
    if not os.path.exists(TEST_FILE):
        model_stats["error"] = f"Test data file not found: {TEST_FILE}"
        return model_stats
    try:
        df_test = pd.read_csv(TEST_FILE, names=COLUMNS)
        X_test = df_test.drop(columns=LABEL_COLUMNS)
        y_test = df_test["label"].apply(lambda x: 0 if _normalize_label(x) == "normal" else 1)
        y_pred = services.model.predict(X_test)

        total = len(y_test)
        attacks = int(sum(y_pred))
        normal = total - attacks
        accuracy = round(float(sum(y_pred == y_test)) / max(total, 1) * 100, 2)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            [normal, attacks],
            labels=["Normal", "Attack"],
            autopct="%1.1f%%",
            colors=["#198754", "#dc3545"],
            startangle=90,
        )
        ax.set_title("Test Data Predictions Distribution")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        chart_data = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close(fig)

        rows = []
        for i in range(min(20, len(y_test))):
            rows.append(
                {
                    "Index": i,
                    "True": "Normal" if y_test.iloc[i] == 0 else "Attack",
                    "Predicted": "Normal" if y_pred[i] == 0 else "Attack",
                    "Match": "OK" if y_test.iloc[i] == y_pred[i] else "MISS",
                }
            )

        model_stats.update(
            {
                "available": True,
                "total": total,
                "attacks": attacks,
                "normal": normal,
                "accuracy": accuracy,
                "chart_data": chart_data,
                "results": rows,
            }
        )
    except Exception:
        logger.exception("dashboard_model_stats_failed")
        model_stats["error"] = "Unable to build model analytics"
    return model_stats


def _cached_model_stats() -> dict[str, Any]:
    services = _services()
    cache_ttl_seconds = 30
    model_cache_key = f"{id(services.model)}::{os.path.getmtime(TEST_FILE) if os.path.exists(TEST_FILE) else 'missing'}"
    now = time.time()
    with services.dashboard_cache_lock:
        cached = services.dashboard_cache.get("model_stats")
        if cached and cached.get("key") == model_cache_key and cached.get("expires_at", 0) > now:
            return cached["value"]
    value = _build_model_stats()
    with services.dashboard_cache_lock:
        services.dashboard_cache["model_stats"] = {
            "key": model_cache_key,
            "expires_at": now + cache_ttl_seconds,
            "value": value,
        }
    return value


def _count_csv_rows(file_storage, max_rows: int) -> int:
    stream = file_storage.stream
    start = stream.tell()
    row_count = 0
    for _ in stream:
        row_count += 1
        if row_count > max_rows + 1:
            stream.seek(start)
            return max_rows + 1
    stream.seek(start)
    return max(0, row_count - 1)


@app.before_request
def _before_request() -> None:
    g.request_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:12]}")
    services = _services()
    if request.path.startswith("/api/"):
        services.metrics_service.inc("requests_total")
        if request.path != "/api/health":
            client_key = f"{request.remote_addr or 'unknown'}:{request.path}"
            allowed, retry_after = services.rate_limiter.allow(client_key)
            if not allowed:
                services.metrics_service.inc("rate_limited_total")
                return _api_error(429, "rate_limited", f"retry_after_seconds={retry_after}")
    return None


@app.after_request
def _after_request(response):
    response.headers["X-Request-ID"] = _request_id()
    if request.path.startswith("/api/") and response.status_code == 401:
        services = _services()
        services.metrics_service.inc("unauthorized_total")
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
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
            logging.info(f"✅ Loaded {model_name}")
    if 'rf_nsl_kdd' in all_models:
        model = all_models['rf_nsl_kdd']
        detection_service = DetectionService(model=model, alert_store=alert_store)




@app.before_request
def _before_request_metrics():
    if request.path.startswith('/api/'):
        metrics_service.inc('requests_total')

        if request.path != '/api/health':
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
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    return response


@app.route("/")
def home():
    ensure_model_loaded()
    services = _services()
    auth_info = auth_status()
    return render_template(
        "home.html",
        model_ready=services.model is not None,
        loaded_models_count=len(services.all_models),
        queue_size=services.ingestion_queue.size(),
        auth_enabled=auth_info["enabled"],
        firewall_adapter=services.settings.firewall_adapter,
    )





@app.route("/predict", methods=["GET", "POST"])
def predict():
<<<<<<< ours
=======
    """Live prediction page with suspicious activity logic."""
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    prediction = None
    error_message = None
    confidence = None
    is_suspicious = False
    model_ready = ensure_detection_service()
    if not model_ready:
        error_message = "No trained model found. Please train and load a model first."

<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    if model_ready and request.method == "POST":
        services = _services()
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    if not ensure_detection_service():
        return render_template("predict.html", features=INPUT_FEATURES,
                               prediction=None, error="No trained model found. Please train and load a model first.",
                               confidence=None, is_suspicious=False)

    if request.method == "POST":
>>>>>>> theirs
        try:
            values = []
            for feat in INPUT_FEATURES:
                v = request.form.get(feat, None)
                if v is None or not v.replace(".", "", 1).replace("-", "", 1).isdigit():
                    raise ValueError(f"Invalid input for {feat}")
                values.append(float(v))

            row = DEFAULT_FEATURE_ROW.copy()
<<<<<<< ours
            row.update(
                {
                    "duration": values[0],
                    "src_bytes": values[1],
                    "dst_bytes": values[2],
                    "count": values[3],
                    "srv_count": values[4],
                    "serror_rate": values[5],
                    "same_srv_rate": values[6],
                }
            )
            result = services.detection_service.predict_from_features(row, profile="balanced")
            confidence = result.confidence
            is_suspicious = result.suspicious
            prediction = "SUSPICIOUS - Low Confidence" if result.suspicious else result.prediction
        except Exception:
            logger.exception("predict_page_failed")
            error_message = "Prediction failed due to invalid input or processing error."

    return render_template(
        "predict.html",
        features=INPUT_FEATURES,
        prediction=prediction,
        error=error_message,
        confidence=confidence,
        is_suspicious=is_suspicious,
        model_ready=model_ready,
    )

=======
            row.update({
                "duration": values[0],
                "src_bytes": values[1],
                "dst_bytes": values[2],
                "count": values[3],
                "srv_count": values[4],
                "serror_rate": values[5],
                "same_srv_rate": values[6],
            })
            result = detection_service.predict_from_features(row, profile="balanced")
            confidence = result.confidence
            is_suspicious = result.suspicious
            prediction = "SUSPICIOUS - Low Confidence" if result.suspicious else result.prediction
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            error_message = f"Error: {e}"

    return render_template("predict.html", features=INPUT_FEATURES,
                           prediction=prediction, error=error_message,
                           confidence=confidence, is_suspicious=is_suspicious)

<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs

@app.route("/dashboard")
def dashboard():
    ensure_model_loaded()
<<<<<<< ours
    services = _services()
    model_stats = _cached_model_stats()

    recent_alerts = services.ops_store.list_alerts(limit=8)
    recent_actions = services.ops_store.list_actions(limit=8)
    recent_audits = services.ops_store.list_audits(limit=8)
    recent_registry = services.model_registry.list_entries(limit=8)
    active_blocks = services.ops_store.list_active_blocks(limit=20)
    reconcile_summary = services.prevention_service.reconcile()
    policy = services.prevention_service.policy.to_dict()

    action_timeline = []
    for action in recent_actions:
        action_timeline.append(
            {
                "when": action.get("created_at"),
                "type": "action",
                "message": f"{action.get('action')} {action.get('target')} ({action.get('status', 'n/a')})",
            }
        )
    for audit in recent_audits:
        action_timeline.append(
            {
                "when": audit.get("created_at"),
                "type": "audit",
                "message": f"{audit.get('event_type')}: {audit.get('message')}",
            }
        )
    action_timeline = sorted(action_timeline, key=lambda x: x.get("when", ""), reverse=True)[:12]

    metrics_snapshot = {
        "requests_total": services.metrics_service.get("requests_total"),
        "predictions_total": services.metrics_service.get("predictions_total"),
        "alerts_total": services.metrics_service.get("alerts_total"),
        "prevention_actions_total": services.metrics_service.get("prevention_actions_total"),
        "rate_limited_total": services.metrics_service.get("rate_limited_total"),
        "unauthorized_total": services.metrics_service.get("unauthorized_total"),
        "ingested_total": services.metrics_service.get("ingested_total"),
        "processed_ingestion_total": services.metrics_service.get("processed_ingestion_total"),
    }

    return render_template(
        "dashboard.html",
        model_stats=model_stats,
        recent_alerts=recent_alerts,
        recent_actions=recent_actions,
        recent_audits=recent_audits,
        recent_registry=recent_registry,
        policy=policy,
        metrics_snapshot=metrics_snapshot,
        auth_info=auth_status(),
        queue_size=services.ingestion_queue.size(),
        firewall_adapter=services.settings.firewall_adapter,
        rate_limit_requests=services.settings.rate_limit_requests,
        rate_limit_window_seconds=services.settings.rate_limit_window_seconds,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        active_blocks=active_blocks,
        action_timeline=action_timeline,
        reconcile_summary=reconcile_summary,
    )
=======
    try:
        if model is None:
            raise FileNotFoundError("No trained model found. Please train and load a model first.")

        if not os.path.exists(TEST_FILE):
            raise FileNotFoundError("Test data file not found!")

        df_test = pd.read_csv(TEST_FILE, names=COLUMNS)
        X_test = df_test.drop(columns=LABEL_COLUMNS)
        y_test = df_test["label"].apply(lambda x: 0 if _normalize_label(x) == "normal" else 1)
        y_pred = model.predict(X_test)

        total = len(y_test)
        attacks = int(sum(y_pred))
        normal = total - attacks
        accuracy = round(float(sum(y_pred == y_test)) / total * 100, 2)

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
            results.append({
                "Index": i,
                "True": "Normal" if y_test.iloc[i] == 0 else "Attack",
                "Predicted": "Normal" if y_pred[i] == 0 else "Attack",
                "Match": "✓" if y_test.iloc[i] == y_pred[i] else "✗"
            })

        return render_template("dashboard.html", total=total, attacks=attacks,
                               normal=normal, accuracy=accuracy,
                               chart_data=chart_data, results=results)
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs

=======
>>>>>>> theirs

=======
>>>>>>> theirs



@app.route("/models")
def models_page():
    services = _services()
    try:
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        model_results: list[dict] = []
        latest_results = None
        if os.path.exists(RESULTS_DIR):
            results_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("model_results_") and f.endswith(".json")]
            if results_files:
                latest_results = sorted(results_files)[-1]
                with open(os.path.join(RESULTS_DIR, latest_results), "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    model_results = [r for r in loaded if isinstance(r, dict)]
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
        if not os.path.exists(RESULTS_DIR):
            return render_template("models.html", models=[], has_data=False)

        results_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
        if not results_files:
            return render_template("models.html", models=[], has_data=False)

        latest_results = sorted(results_files)[-1]
        with open(os.path.join(RESULTS_DIR, latest_results), 'r', encoding='utf-8') as f:
            model_results = json.load(f)
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
        model_results = sorted(model_results, key=lambda x: x.get("accuracy", 0), reverse=True)
        chart_files = {
            "model_comparison": os.path.exists(os.path.join(STATIC_DIR, "model_comparison.png")),
            "training_time_comparison": os.path.exists(os.path.join(STATIC_DIR, "training_time_comparison.png")),
            "roc_curves": os.path.exists(os.path.join(STATIC_DIR, "roc_curves.png")),
        }
        return render_template(
            "models.html",
            models=model_results,
            has_data=bool(model_results),
            latest_results=latest_results,
            chart_files=chart_files,
            registry_entries=services.model_registry.list_entries(limit=12),
        )
    except Exception:
        logger.exception("models_page_failed")
        return render_template(
            "models.html",
            models=[],
            has_data=False,
            latest_results=None,
            chart_files={"model_comparison": False, "training_time_comparison": False, "roc_curves": False},
            registry_entries=[],
        )





@app.route("/batch", methods=["GET", "POST"])
def batch_predict():
    ensure_model_loaded()
    services = _services()
    if request.method == "POST":
        try:
            if services.model is None:
                return render_template("batch.html", error="No trained model found. Please train and load a model first.")

            file = request.files.get("file")
            if not file:
                return render_template("batch.html", error="No file uploaded")

            max_rows = 50000
            row_count = _count_csv_rows(file, max_rows=max_rows)
            if row_count > max_rows:
                return render_template("batch.html", error=f"CSV too large (> {max_rows} rows).")

            file.stream.seek(0)
            df = pd.read_csv(file)
            if df.empty:
                return render_template("batch.html", error="Uploaded CSV is empty.")

<<<<<<< ours
=======
            max_rows = 50000
            if len(df) > max_rows:
                return render_template("batch.html", error=f"CSV too large ({len(df)} rows). Max allowed is {max_rows}.")

<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
            df = df.drop(columns=LABEL_COLUMNS, errors="ignore")
            missing_cols = [col for col in MODEL_INPUT_COLUMNS if col not in df.columns]
            if missing_cols:
                return render_template("batch.html", error=f"Missing required columns: {', '.join(missing_cols)}")

            df = df[MODEL_INPUT_COLUMNS].copy()
            for col in NUMERIC_MODEL_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[NUMERIC_MODEL_COLUMNS].isna().any().any():
                return render_template("batch.html", error="One or more numeric columns contain invalid values.")

<<<<<<< ours
            predictions = services.model.predict(df)
            results = [{"row": i, "prediction": "Attack" if p == 1 else "Normal"} for i, p in enumerate(predictions)]
            return render_template("batch.html", results=results[:50], total=len(results), shown=min(50, len(results)))
        except Exception:
            logger.exception("batch_predict_failed")
            return render_template("batch.html", error="Batch prediction failed")
=======
            predictions = model.predict(df)
            results = [{"row": i, "prediction": "Attack" if p == 1 else "Normal"}
                       for i, p in enumerate(predictions)]
            return render_template("batch.html", results=results[:50],
                                   total=len(results), shown=min(50, len(results)))
        except Exception as e:
            logging.error(f"Batch prediction error: {e}")
            return render_template("batch.html", error=str(e))
>>>>>>> theirs
    return render_template("batch.html")


@app.route("/api/health", methods=["GET"])
def api_health():
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    services = _services()
    model_ready = ensure_detection_service()
    return jsonify(
        {
            "status": "ok",
            "request_id": _request_id(),
            "model_loaded": model_ready,
            "alerts_buffered": len(services.alert_store.list_alerts(limit=1000)),
            "ops_db": services.ops_db_path,
            "auth": auth_status(),
            "metrics": {
                "requests_total": services.metrics_service.get("requests_total"),
                "predictions_total": services.metrics_service.get("predictions_total"),
            },
            "ingestion_queue_size": services.ingestion_queue.size(),
            "rate_limit": {
                "requests": services.settings.rate_limit_requests,
                "window_seconds": services.settings.rate_limit_window_seconds,
            },
            "firewall_adapter": services.settings.firewall_adapter,
        }
    )


@app.route("/api/predict", methods=["POST"])
@require_role("sensor")
def api_predict():
    services = _services()
    if not ensure_detection_service():
        return _api_error(503, "model_unavailable")
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    model_ready = ensure_detection_service()
    return jsonify({
        "status": "ok",
        "model_loaded": model_ready,
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
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not ensure_detection_service():
        return jsonify({"error": "No trained model found"}), 503
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs

    payload = request.get_json(silent=True) or {}
    features = payload.get("features", {})
    profile = payload.get("profile", "balanced")

    if not isinstance(features, dict) or not features:
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        return _api_error(400, "invalid_request", "'features' must be a non-empty object")

    try:
        normalized = dict(features)
        for col in NUMERIC_MODEL_COLUMNS:
            if col in normalized:
                normalized[col] = float(normalized[col])
        services.metrics_service.inc("predictions_total")
        result = services.detection_service.predict_from_features(normalized, profile=profile)
        if result.alert:
            services.ops_store.save_alert(result.alert.to_dict())
            services.metrics_service.inc("alerts_total")

        source = payload.get("source", "unknown")
        action = services.prevention_service.evaluate(result.prediction, result.confidence, source=source)
        if action:
            services.metrics_service.inc("prevention_actions_total")

        response = result.to_dict()
        response["prevention_action"] = action.to_dict() if action else None
        response["request_id"] = _request_id()
        return jsonify(response)
    except ValueError as exc:
        logger.warning("api_predict_invalid request_id=%s error=%s", _request_id(), exc)
        return _api_error(400, "invalid_request", str(exc))
    except Exception:
        logger.exception("api_predict_failed")
        return _api_error(500, "internal_error")


@app.route("/api/alerts", methods=["GET"])
@require_role("viewer")
def api_alerts():
    services = _services()
    limit = request.args.get("limit", default=50, type=int)
    severity = request.args.get("severity", default=None, type=str)
    alerts = services.ops_store.list_alerts(limit=max(1, min(limit, 200)), severity=severity)
    return jsonify({"count": len(alerts), "alerts": alerts, "request_id": _request_id()})


@app.route("/api/policy", methods=["GET", "POST"])
@require_role("viewer")
def api_policy():
    services = _services()
    if request.method == "GET":
        return jsonify({**services.prevention_service.policy.to_dict(), "request_id": _request_id()})

    ok, reason = authorize_request("admin")
    if not ok:
        return _api_error(401, "unauthorized", reason)

    payload = request.get_json(silent=True) or {}
    try:
        policy = services.prevention_service.set_policy(
            mode=payload.get("mode"),
            block_ttl_seconds=payload.get("block_ttl_seconds"),
            confidence_block_threshold=payload.get("confidence_block_threshold"),
            dry_run=payload.get("dry_run"),
            risk_alert_threshold=payload.get("risk_alert_threshold"),
            risk_rate_limit_threshold=payload.get("risk_rate_limit_threshold"),
            risk_block_threshold=payload.get("risk_block_threshold"),
        )
        services.ops_store.add_audit(
            event_type="policy_update",
            message=(
                f"mode={policy.mode}, ttl={policy.block_ttl_seconds}, threshold={policy.confidence_block_threshold}, "
                f"dry_run={policy.dry_run}, risk_alert={policy.risk_alert_threshold}, "
                f"risk_rate_limit={policy.risk_rate_limit_threshold}, risk_block={policy.risk_block_threshold}"
            ),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        services.metrics_service.inc("policy_updates_total")
        return jsonify({**policy.to_dict(), "request_id": _request_id()})
    except ValueError as exc:
        return _api_error(400, "invalid_request", str(exc))
    except Exception:
        logger.exception("api_policy_failed")
        return _api_error(500, "internal_error")


@app.route("/api/actions", methods=["GET"])
@require_role("viewer")
def api_actions():
    services = _services()
    limit = request.args.get("limit", default=50, type=int)
    actions = services.ops_store.list_actions(limit=max(1, min(limit, 200)))
    return jsonify({"count": len(actions), "actions": actions, "request_id": _request_id()})
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
        return jsonify({"error": "'features' must be a non-empty object"}), 400

    try:
        for col in NUMERIC_MODEL_COLUMNS:
            if col in features:
                features[col] = float(features[col])
        metrics_service.inc("predictions_total")
        result = detection_service.predict_from_features(features, profile=profile)
        if result.alert:
            ops_store.save_alert(result.alert.to_dict())
            metrics_service.inc("alerts_total")

        source = payload.get("source", "unknown")
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
        logging.error("API predict error: %s", exc)
        return jsonify({"error": str(exc)}), 400


@app.route("/api/alerts", methods=["GET"])
@require_role("analyst")
def api_alerts():
    limit = request.args.get("limit", default=50, type=int)
    severity = request.args.get("severity", default=None, type=str)
    alerts = ops_store.list_alerts(limit=max(1, min(limit, 200)), severity=severity)
    return jsonify({"count": len(alerts), "alerts": alerts})




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
        )
        ops_store.add_audit(
            event_type="policy_update",
            message=f"mode={policy.mode}, ttl={policy.block_ttl_seconds}, threshold={policy.confidence_block_threshold}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        metrics_service.inc("policy_updates_total")
        return jsonify(policy.to_dict())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/actions", methods=["GET"])
@require_role("analyst")
def api_actions():
    limit = request.args.get("limit", default=50, type=int)
    actions = ops_store.list_actions(limit=max(1, min(limit, 200)))
    return jsonify({"count": len(actions), "actions": actions})


<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs


@app.route("/api/actions/cleanup", methods=["POST"])
@require_role("admin")
def api_actions_cleanup():
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    services = _services()
    payload = request.get_json(silent=True) or {}
    now_iso = payload.get("now")
    try:
        removed = services.prevention_service.cleanup_expired_actions(now_iso=now_iso)
    except ValueError:
        return _api_error(400, "invalid_request", "'now' must be an ISO-8601 datetime string")
    if removed:
        services.metrics_service.inc("expired_actions_cleaned_total", amount=removed)
        services.ops_store.add_audit(
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    payload = request.get_json(silent=True) or {}
    now_iso = payload.get("now")
    removed = ops_store.cleanup_expired_actions(now_iso=now_iso)
    if removed:
        ops_store.add_audit(
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
            event_type="actions_cleanup",
            message=f"removed_expired_actions={removed}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    return jsonify({"removed": removed, "request_id": _request_id()})


@app.route("/api/actions/reconcile", methods=["POST"])
@require_role("admin")
def api_actions_reconcile():
    services = _services()
    summary = services.prevention_service.reconcile()
    return jsonify({"summary": summary, "request_id": _request_id()})
=======
        metrics_service.inc("expired_actions_cleaned_total", amount=removed)
    return jsonify({"removed": removed})
>>>>>>> theirs
=======
        metrics_service.inc("expired_actions_cleaned_total", amount=removed)
    return jsonify({"removed": removed})
>>>>>>> theirs
=======
        metrics_service.inc("expired_actions_cleaned_total", amount=removed)
    return jsonify({"removed": removed})
>>>>>>> theirs


@app.route("/api/audit", methods=["GET"])
@require_role("admin")
def api_audit():
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    services = _services()
    limit = request.args.get("limit", default=100, type=int)
    audits = services.ops_store.list_audits(limit=max(1, min(limit, 500)))
    return jsonify({"count": len(audits), "audits": audits, "request_id": _request_id()})


@app.route("/api/explain", methods=["POST"])
@require_role("viewer")
def api_explain():
    payload = request.get_json(silent=True) or {}
    features = payload.get("features", {})
    try:
        top_k = _parse_bounded_int(payload.get("top_k"), name="top_k", default=5, minimum=1, maximum=20)
    except ValueError as exc:
        return _api_error(400, "invalid_request", str(exc))

    if not isinstance(features, dict) or not features:
        return _api_error(400, "invalid_request", "'features' must be a non-empty object")

    try:
        explanation = DetectionService.explain_features(features, top_k=top_k)
        return jsonify({"top_k": top_k, "explanation": explanation, "request_id": _request_id()})
    except Exception:
        logger.exception("api_explain_failed")
        return _api_error(500, "internal_error")


@app.route("/api/models/registry", methods=["GET"])
@require_role("viewer")
def api_model_registry():
    services = _services()
    limit = request.args.get("limit", default=50, type=int)
    entries = services.model_registry.list_entries(limit=max(1, min(limit, 200)))
    return jsonify({"count": len(entries), "models": entries, "request_id": _request_id()})


@app.route("/api/metrics", methods=["GET"])
@require_role("viewer")
def api_metrics():
    services = _services()
    return Response(services.metrics_service.as_prometheus(), mimetype="text/plain; version=0.0.4")


@app.route("/api/ingest", methods=["POST"])
@require_role("sensor")
def api_ingest():
    services = _services()
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    limit = request.args.get("limit", default=100, type=int)
    audits = ops_store.list_audits(limit=max(1, min(limit, 500)))
    return jsonify({"count": len(audits), "audits": audits})


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
        return jsonify({"error": str(exc)}), 400


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
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    payload = request.get_json(silent=True) or {}
    source = payload.get("source", "ingestion_api")
    rows = payload.get("rows")

    try:
        if isinstance(rows, list):
            if not rows:
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
                return _api_error(400, "invalid_request", "rows cannot be empty")
            added = services.ingestion_service.enqueue_batch(rows, source=source)
        else:
            features = payload.get("features")
            if not isinstance(features, dict) or not features:
                return _api_error(400, "invalid_request", "provide either non-empty 'rows' or 'features'")
            services.ingestion_service.enqueue_record(features, source=source)
            added = 1

        services.metrics_service.inc("ingested_total", amount=added)
        return jsonify({"queued": added, "queue_size": services.ingestion_queue.size(), "request_id": _request_id()})
    except ValueError as exc:
        return _api_error(400, "invalid_request", str(exc))
    except Exception:
        logger.exception("api_ingest_failed")
        return _api_error(500, "internal_error")


@app.route("/api/ingest/log", methods=["POST"])
@require_role("sensor")
def api_ingest_log():
    services = _services()
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
                return jsonify({"error": "rows cannot be empty"}), 400
            added = ingestion_service.enqueue_batch(rows, source=source)
        else:
            features = payload.get("features")
            if not isinstance(features, dict) or not features:
                return jsonify({"error": "provide either non-empty 'rows' or 'features'"}), 400
            ingestion_service.enqueue_record(features, source=source)
            added = 1

        metrics_service.inc("ingested_total", amount=added)
        return jsonify({"queued": added, "queue_size": ingestion_queue.size()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/ingest/log", methods=["POST"])
@require_role("analyst")
def api_ingest_log():
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    payload = request.get_json(silent=True) or {}
    source_type = str(payload.get("type", "zeek")).lower()
    records = payload.get("records")
    if not isinstance(records, list) or not records:
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        return _api_error(400, "invalid_request", "'records' must be a non-empty list")
=======
        return jsonify({"error": "'records' must be a non-empty list"}), 400
>>>>>>> theirs
=======
        return jsonify({"error": "'records' must be a non-empty list"}), 400
>>>>>>> theirs
=======
        return jsonify({"error": "'records' must be a non-empty list"}), 400
>>>>>>> theirs

    try:
        transformed = []
        for rec in records:
            if source_type == "zeek":
                transformed.append(parse_zeek_conn_log(rec))
            elif source_type == "suricata":
                transformed.append(parse_suricata_eve_flow(rec))
            else:
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
                return _api_error(400, "invalid_request", "type must be 'zeek' or 'suricata'")

        added = services.ingestion_service.enqueue_batch(transformed, source=f"{source_type}_log")
        services.metrics_service.inc("ingested_total", amount=added)
        return jsonify(
            {
                "queued": added,
                "queue_size": services.ingestion_queue.size(),
                "type": source_type,
                "request_id": _request_id(),
            }
        )
    except ValueError as exc:
        return _api_error(400, "invalid_request", str(exc))
    except Exception:
        logger.exception("api_ingest_log_failed")
        return _api_error(500, "internal_error")


@app.route("/api/ingest/process", methods=["POST"])
@require_role("sensor")
def api_ingest_process():
    services = _services()
    if not ensure_detection_service():
        return _api_error(503, "model_unavailable")

    payload = request.get_json(silent=True) or {}
    try:
        max_items = _parse_bounded_int(payload.get("max_items"), name="max_items", default=50, minimum=1, maximum=500)
    except ValueError as exc:
        return _api_error(400, "invalid_request", str(exc))

    def _handler(features, source):
        result = services.detection_service.predict_from_features(features, profile="balanced")
        if result.alert:
            services.ops_store.save_alert(result.alert.to_dict())
            services.metrics_service.inc("alerts_total")

        action = services.prevention_service.evaluate(result.prediction, result.confidence, source=source)
        if action:
            services.metrics_service.inc("prevention_actions_total")
        services.metrics_service.inc("processed_ingestion_total")
        payload_local = result.to_dict()
        payload_local["prevention_action"] = action.to_dict() if action else None
        return payload_local

    processed = services.ingestion_service.process_all(_handler, max_items=max_items)
    return jsonify(
        {
            "processed": len(processed),
            "queue_size": services.ingestion_queue.size(),
            "results": processed,
            "request_id": _request_id(),
        }
    )
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
                return jsonify({"error": "type must be 'zeek' or 'suricata'"}), 400

        added = ingestion_service.enqueue_batch(transformed, source=f"{source_type}_log")
        metrics_service.inc("ingested_total", amount=added)
        return jsonify({"queued": added, "queue_size": ingestion_queue.size(), "type": source_type})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/ingest/process", methods=["POST"])
@require_role("analyst")
def api_ingest_process():
    if not ensure_detection_service():
        return jsonify({"error": "No trained model found"}), 503

    payload = request.get_json(silent=True) or {}
    max_items = int(payload.get("max_items", 50))
    max_items = max(1, min(max_items, 500))

    def _handler(features, source):
        result = detection_service.predict_from_features(features, profile="balanced")
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
        payload = result.to_dict()
        payload["prevention_action"] = action.to_dict() if action else None
        return payload

    processed = ingestion_service.process_all(_handler, max_items=max_items)
    return jsonify({
        "processed": len(processed),
        "queue_size": ingestion_queue.size(),
        "results": processed,
    })
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/realtime")
def realtime():
    return render_template("realtime.html")


@app.route("/capture")
def capture():
    return render_template("capture.html")


@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return _api_error(404, "not_found")
    return render_template("404.html"), 404


@app.errorhandler(Exception)
def handle_error(e):
    if isinstance(e, HTTPException):
        if request.path.startswith("/api/"):
            return _api_error(e.code or 500, "http_error")
        return e
    logger.exception("unhandled_application_error")
    if request.path.startswith("/api/"):
        return _api_error(500, "internal_error")
    return render_template("error.html", error="An unexpected error occurred."), 500





if __name__ == "__main__":
    load_models()
    app.run(
        debug=SETTINGS.debug,
        host=SETTINGS.host,
        port=SETTINGS.port,
    )
