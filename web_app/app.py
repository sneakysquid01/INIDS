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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")
OPS_DB_PATH = os.getenv("OPS_DB_PATH", os.path.join(DATA_DIR, "inids_ops.db"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.detection_service import DetectionService, InMemoryAlertStore
from src.prevention_service import PreventionService
from src.ops_store import OpsStore
from src.auth_service import require_role, auth_status
from src.metrics_service import MetricsService
from src.schema import (
    COLUMNS,
    LABEL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    DEFAULT_FEATURE_ROW,
)

app = Flask(__name__)

model = None
all_models = {}
alert_store = InMemoryAlertStore(max_items=1000)
detection_service = None
prevention_service = PreventionService()
ops_store = OpsStore(OPS_DB_PATH)
metrics_service = MetricsService()

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


@app.route("/api/audit", methods=["GET"])
@require_role("admin")
def api_audit():
    limit = request.args.get("limit", default=100, type=int)
    audits = ops_store.list_audits(limit=max(1, min(limit, 500)))
    return jsonify({"count": len(audits), "audits": audits})


@app.route("/api/metrics", methods=["GET"])
@require_role("analyst")
def api_metrics():
    return Response(metrics_service.as_prometheus(), mimetype="text/plain; version=0.0.4")


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
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        host='0.0.0.0',
        port=int(os.getenv("PORT", "5000")),
    )
