"""Health and metrics endpoints — Step 39, first blueprint extraction."""
from flask import Blueprint, Response, jsonify

from src.auth.decorators import require_roles

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health", methods=["GET"])
def api_health():
    import web_app.app as _m
    model_ready = _m.ensure_detection_service()
    _m._ensure_pipeline_started()
    readiness = _m.health_check.check()
    return jsonify({
        "status": "ok",
        "model_loaded": model_ready,
        "readiness": readiness,
        "alerts_buffered": len(_m.ops_store.list_alerts(limit=1000)),
        "auth": {"enabled": True, "configured_roles": ["admin", "analyst", "viewer", "sensor"]},
        "metrics": {
            "requests_total": _m.metrics_service.get("requests_total"),
            "predictions_total": _m.metrics_service.get("predictions_total"),
        },
        "ingestion_queue_size": _m.ingestion_queue.size(),
        "rate_limit": {
            "requests": _m.SETTINGS.rate_limit_requests,
            "window_seconds": _m.SETTINGS.rate_limit_window_seconds,
        },
        "firewall_adapter": _m.SETTINGS.firewall_adapter,
        "detection_engines": _m.engine_registry.list_engines(),
        "pipeline": _m._pipeline_status(),
        "leader_election": _m.leader_election.status(),
        **__import__("src._telemetry", fromlist=["health_snapshot"]).health_snapshot(),
    })


@health_bp.route("/api/health/live", methods=["GET"])
def api_health_live():
    return jsonify({"status": "live", "process_up": True}), 200


@health_bp.route("/api/health/ready", methods=["GET"])
def api_health_ready():
    import web_app.app as _m
    _m._ensure_pipeline_started()
    report = _m.health_check.check()
    return jsonify(report), (200 if report.get("status") == "healthy" else 503)


@health_bp.route("/api/metrics", methods=["GET"])
@require_roles("analyst")
def api_metrics():
    import web_app.app as _m
    return Response(_m.metrics_service.as_prometheus(), mimetype="text/plain; version=0.0.4")
