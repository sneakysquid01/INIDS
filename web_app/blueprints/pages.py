"""HTML page routes — Step 39, pages blueprint."""
import json
import logging
import os

import pandas as pd
from flask import Blueprint, render_template, request

from src.auth.decorators import require_roles
from src.schema import LABEL_COLUMNS

logger = logging.getLogger(__name__)
pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
@require_roles("viewer")
def home():
    import web_app.app as _m
    metrics_snapshot = {
        "requests_total": _m.metrics_service.get("requests_total"),
        "predictions_total": _m.metrics_service.get("predictions_total"),
        "alerts_total": _m.metrics_service.get("alerts_total"),
        "prevention_actions_total": _m.metrics_service.get("prevention_actions_total"),
        "ingested_total": _m.metrics_service.get("ingested_total"),
        "processed_ingestion_total": _m.metrics_service.get("processed_ingestion_total"),
    }
    return render_template(
        "home.html",
        loaded_models_count=len(_m.all_models),
        queue_size=_m.ingestion_queue.size(),
        firewall_adapter=_m.SETTINGS.firewall_adapter,
        auth_enabled=True,
        metrics_snapshot=metrics_snapshot,
    )


@pages_bp.route("/predict", methods=["GET", "POST"])
@require_roles("viewer")
def predict():
    import web_app.app as _m
    prediction = None
    error_message = None
    confidence = None
    is_suspicious = False

    if not _m.ensure_detection_service():
        return render_template("predict.html", features=_m.INPUT_FEATURES,
                               prediction=None, error="No trained model found. Please train and load a model first.",
                               confidence=None, is_suspicious=False)

    if request.method == "POST":
        try:
            values = []
            for feat in _m.INPUT_FEATURES:
                v = request.form.get(feat, None)
                if v is None or not v.replace('.', '', 1).replace('-', '', 1).isdigit():
                    raise ValueError(f"Invalid input for {feat}")
                values.append(float(v))

            from src.schema import DEFAULT_FEATURE_ROW
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
            result = _m.detection_service.predict_from_features(
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

    return render_template("predict.html", features=_m.INPUT_FEATURES,
                           prediction=prediction, error=error_message,
                           confidence=confidence, is_suspicious=is_suspicious)


@pages_bp.route("/alerts")
@require_roles("viewer")
def alerts_page():
    return render_template("alerts.html")


@pages_bp.route("/actions")
@require_roles("viewer")
def actions_page():
    return render_template("actions.html")


@pages_bp.route("/detection")
@require_roles("viewer")
def detection_page():
    return render_template("detection.html")


@pages_bp.route("/engines")
@require_roles("viewer")
def engines_page():
    return render_template("engines.html")


@pages_bp.route("/policy")
@require_roles("viewer")
def policy_page():
    return render_template("policy.html")


@pages_bp.route("/allowlist")
@require_roles("viewer")
def allowlist_page():
    return render_template("allowlist.html")


@pages_bp.route("/threat-intel")
@require_roles("viewer")
def threat_intel_page():
    return render_template("threat_intel.html")


@pages_bp.route("/health")
def health_page():
    return render_template("health.html")


@pages_bp.route("/dashboard")
@require_roles("viewer")
def dashboard():
    import web_app.app as _m
    from datetime import datetime, timezone
    try:
        now = datetime.now(timezone.utc)
        recent_actions = _m.ops_store.list_actions(limit=20)
        active_blocks = []
        for action in recent_actions:
            active_blocks.append(
                {
                    "target": action.get("target", ""),
                    "status": _m._action_status(action.get("expires_at"), now),
                    "expires_at": action.get("expires_at"),
                }
            )

        action_timeline = []
        for action in recent_actions[:10]:
            action_timeline.append(
                {
                    "when": _m._safe_created_at(action, now.isoformat()),
                    "type": action.get("action", "action"),
                    "message": f"{action.get('action', '')} target={action.get('target', '')} reason={action.get('reason', '')}",
                }
            )

        recent_audits = _m.ops_store.list_audits(limit=20)
        for audit in recent_audits[:10]:
            action_timeline.append(
                {
                    "when": _m._safe_created_at(audit, now.isoformat()),
                    "type": audit.get("event_type", "audit"),
                    "message": audit.get("message", ""),
                }
            )
        action_timeline = sorted(action_timeline, key=lambda item: item.get("when", ""), reverse=True)[:15]

        adapter = _m.prevention_service.adapter
        firewall_rules = 0
        if hasattr(adapter, "blocked_targets"):
            blocked = getattr(adapter, "blocked_targets", {})
            firewall_rules = len(blocked) if isinstance(blocked, dict) else 0

        metrics_snapshot = {
            "requests_total": _m.metrics_service.get("requests_total"),
            "predictions_total": _m.metrics_service.get("predictions_total"),
            "alerts_total": _m.metrics_service.get("alerts_total"),
            "prevention_actions_total": _m.metrics_service.get("prevention_actions_total"),
            "ingested_total": _m.metrics_service.get("ingested_total"),
            "processed_ingestion_total": _m.metrics_service.get("processed_ingestion_total"),
            "rate_limited_total": _m.metrics_service.get("rate_limited_total"),
            "unauthorized_total": _m.metrics_service.get("unauthorized_total"),
        }

        return render_template(
            "dashboard.html",
            generated_at=now.isoformat(),
            auth_info={"enabled": True, "configured_roles": ["admin", "analyst", "viewer", "sensor"]},
            queue_size=_m.ingestion_queue.size(),
            rate_limit_requests=_m.SETTINGS.rate_limit_requests,
            rate_limit_window_seconds=_m.SETTINGS.rate_limit_window_seconds,
            firewall_adapter=_m.SETTINGS.firewall_adapter,
            model_stats=_m._model_stats(),
            policy=_m.prevention_service.policy.to_dict(),
            metrics_snapshot=metrics_snapshot,
            recent_alerts=_m.ops_store.list_alerts(limit=10),
            recent_actions=recent_actions[:10],
            recent_audits=recent_audits[:10],
            recent_registry=_m.model_registry.list_entries(limit=10),
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
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return render_template(
            "dashboard.html",
            generated_at=now.isoformat(),
            auth_info={"enabled": True, "configured_roles": ["admin", "analyst", "viewer", "sensor"]},
            queue_size=_m.ingestion_queue.size(),
            rate_limit_requests=_m.SETTINGS.rate_limit_requests,
            rate_limit_window_seconds=_m.SETTINGS.rate_limit_window_seconds,
            firewall_adapter=_m.SETTINGS.firewall_adapter,
            model_stats={"available": False, "error": "Dashboard is temporarily unavailable.", "results": []},
            policy=_m.prevention_service.policy.to_dict(),
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


@pages_bp.route("/monitor")
@require_roles("viewer")
def monitor():
    try:
        return render_template("monitor.html")
    except Exception:
        logger.exception("Monitor page rendering failed")
        return "Monitor page error", 500


@pages_bp.route("/investigate")
@require_roles("analyst")
def investigate():
    try:
        return render_template("investigate.html")
    except Exception:
        logger.exception("Investigate page rendering failed")
        return "Investigate page error", 500


@pages_bp.route("/respond")
@require_roles("analyst")
def respond():
    try:
        return render_template("respond.html")
    except Exception:
        logger.exception("Respond page rendering failed")
        return "Respond page error", 500


@pages_bp.route("/learn")
@require_roles("viewer")
def learn():
    try:
        return render_template("learn.html")
    except Exception:
        logger.exception("Learn page rendering failed")
        return "Learn page error", 500


@pages_bp.route("/dashboard/main")
@require_roles("viewer")
def dashboard_main():
    try:
        return render_template("dashboard_main.html")
    except Exception:
        logger.exception("Dashboard main rendering failed")
        return render_template("dashboard_main.html"), 500


@pages_bp.route("/models")
@require_roles("analyst")
def models_page():
    import web_app.app as _m
    chart_files = {
        "model_comparison": os.path.exists(os.path.join(_m.STATIC_DIR, "model_comparison.png")),
        "training_time_comparison": os.path.exists(os.path.join(_m.STATIC_DIR, "training_time_comparison.png")),
        "roc_curves": os.path.exists(os.path.join(_m.STATIC_DIR, "roc_curves.png")),
    }
    registry_entries = _m.model_registry.list_entries(limit=25)
    model_results = []
    latest_results = None

    try:
        if os.path.exists(_m.RESULTS_DIR):
            results_files = [
                f for f in os.listdir(_m.RESULTS_DIR)
                if f.startswith("model_results_") and f.endswith(".json")
            ]
            if results_files:
                latest_results = sorted(results_files)[-1]
                with open(os.path.join(_m.RESULTS_DIR, latest_results), 'r', encoding='utf-8') as f:
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


@pages_bp.route("/batch", methods=["GET", "POST"])
@require_roles("analyst")
def batch_predict():
    import web_app.app as _m
    _m.ensure_model_loaded()
    if request.method == "POST":
        try:
            if _m.model is None:
                return render_template("batch.html", error="No trained model found. Please train and load a model first.")

            file = request.files.get('file')
            if not file:
                return render_template("batch.html", error="No file uploaded")
            if not file.filename or not file.filename.lower().endswith('.csv'):
                return render_template("batch.html", error="Only .csv files are accepted.")

            if file.content_type not in ('text/csv', 'text/plain', 'application/vnd.ms-excel', 'application/csv'):
                logger.warning(f"Invalid MIME type for batch upload: {file.content_type}")
                return render_template("batch.html", error="Invalid file type. Only CSV files are accepted.")

            df = pd.read_csv(file)
            if df.empty:
                return render_template("batch.html", error="Uploaded CSV is empty.")

            max_rows = _m.MAX_CSV_ROWS
            if len(df) > max_rows:
                return render_template("batch.html", error=f"CSV too large ({len(df)} rows). Max allowed is {max_rows}.")

            df = df.drop(columns=LABEL_COLUMNS, errors="ignore")
            missing_cols = [col for col in _m.MODEL_INPUT_COLUMNS if col not in df.columns]
            if missing_cols:
                return render_template("batch.html", error=f"Missing required columns: {', '.join(missing_cols)}")

            df = df[_m.MODEL_INPUT_COLUMNS].copy()

            for col in _m.NUMERIC_MODEL_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[_m.NUMERIC_MODEL_COLUMNS].isna().any().any():
                return render_template("batch.html", error="One or more numeric columns contain invalid values.")

            predictions = _m.model.predict(df)
            results = [{"row": i, "prediction": "Attack" if p == 1 else "Normal"}
                       for i, p in enumerate(predictions)]
            return render_template("batch.html", results=results[:50],
                                   total=len(results), shown=min(50, len(results)))
        except Exception as e:
            logger.error("Batch prediction error: %s", e)
            return render_template("batch.html", error=str(e))
    return render_template("batch.html")


@pages_bp.route("/about")
@require_roles("viewer")
def about():
    return render_template("about.html")


@pages_bp.route("/realtime")
@require_roles("viewer")
def realtime():
    return render_template("realtime.html")


@pages_bp.route("/capture")
@require_roles("analyst")
def capture():
    return render_template("capture.html")


@pages_bp.route("/honeypot")
@require_roles("analyst")
def honeypot():
    return render_template("honeypot.html")
