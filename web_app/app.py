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
from datetime import datetime, timezone

from src.settings import load_settings
from src.rate_limiter import InMemoryRateLimiter, RateLimitConfig
from src.firewall_adapters import MockFirewallAdapter, UfwFirewallAdapter, NftablesFirewallAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = load_settings()
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")
OPS_DB_PATH = SETTINGS.ops_db_path if os.path.isabs(SETTINGS.ops_db_path) else os.path.join(BASE_DIR, SETTINGS.ops_db_path)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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
)

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
]

MODEL_INPUT_COLUMNS = FEATURE_COLUMNS
NUMERIC_MODEL_COLUMNS = NUMERIC_FEATURES


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
        detection_service = DetectionService(model=model, alert_store=alert_store)
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


@app.route("/dashboard")
def dashboard():
    """Visual dashboard of system predictions and accuracy."""
    ensure_model_loaded()
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

    except Exception as e:
        logging.error(f"Dashboard error: {e}")
        return render_template("error.html", error=str(e)), 500


@app.route("/models")
def models_page():
    """Comparison page of all trained models and their metrics."""
    try:
        if not os.path.exists(RESULTS_DIR):
            return render_template("models.html", models=[], has_data=False)

        results_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
        if not results_files:
            return render_template("models.html", models=[], has_data=False)

        latest_results = sorted(results_files)[-1]
        with open(os.path.join(RESULTS_DIR, latest_results), 'r', encoding='utf-8') as f:
            model_results = json.load(f)
        model_results = sorted(model_results, key=lambda x: x.get("accuracy", 0), reverse=True)
        return render_template("models.html", models=model_results, has_data=True)
    except Exception as e:
        logging.error(f"Models page error: {e}")
        return render_template("models.html", models=[], has_data=False)


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
            logging.error(f"Batch prediction error: {e}")
            return render_template("batch.html", error=str(e))
    return render_template("batch.html")


@app.route("/api/health", methods=["GET"])
def api_health():
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

    payload = request.get_json(silent=True) or {}
    features = payload.get("features", {})
    profile = payload.get("profile", "balanced")

    if not isinstance(features, dict) or not features:
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


@app.route("/api/audit", methods=["GET"])
@require_role("admin")
def api_audit():
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
    payload = request.get_json(silent=True) or {}
    source = payload.get("source", "ingestion_api")
    rows = payload.get("rows")

    try:
        if isinstance(rows, list):
            if not rows:
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
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(Exception)
def handle_error(e):
    logging.error(str(e))
    return render_template("error.html", error=str(e)), 500


if __name__ == "__main__":
    load_models()
    app.run(
        debug=SETTINGS.debug,
        host=SETTINGS.host,
        port=SETTINGS.port,
    )
