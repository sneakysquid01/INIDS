"""Audit, SIEM, threat-intel, and explain endpoints — Step 39, fourth blueprint."""
import logging

from flask import Blueprint, jsonify, request

from src.auth.decorators import require_roles
from src.detection_service import DetectionService
from src.input_sanitizer import SanitizationError, sanitize_ip_address

logger = logging.getLogger(__name__)
observability_bp = Blueprint("observability", __name__)


@observability_bp.route("/api/audit", methods=["GET"])
@require_roles("admin")
def api_audit():
    import web_app.app as _m
    limit = request.args.get("limit", default=100, type=int)
    audits = _m.ops_store.list_audits(limit=max(1, min(limit, 500)))
    return jsonify({"count": len(audits), "audits": audits})


@observability_bp.route("/api/siem/flush", methods=["GET"])
@require_roles("admin")
def api_siem_flush():
    import web_app.app as _m
    limit = request.args.get("limit", default=500, type=int)
    limit = max(1, min(limit, 5000))
    jsonl = _m.siem_exporter.flush_jsonl(limit)
    lines = [line for line in jsonl.splitlines() if line.strip()]
    return jsonify({"count": len(lines), "jsonl": jsonl, "stats": _m.siem_exporter.stats()})


@observability_bp.route("/api/threat-intel/stats", methods=["GET"])
@require_roles("analyst")
def api_ti_stats():
    import web_app.app as _m
    return jsonify({
        "stats": _m.ti_manager.stats(),
        "feeds": _m.ti_manager.feed_summary(),
        "engine_ready": _m.ti_engine.is_ready(),
        "engine_enabled": _m.engine_registry.is_enabled(_m.ti_engine.engine_id),
    })


@observability_bp.route("/api/threat-intel/lookup", methods=["POST"])
@require_roles("analyst")
def api_ti_lookup():
    import web_app.app as _m
    body = request.get_json(silent=True) or {}
    ip = str(body.get("ip", "")).strip()
    if not ip:
        return jsonify({"error": "'ip' is required"}), 400
    try:
        ip = sanitize_ip_address(ip)
    except SanitizationError as exc:
        return jsonify({"error": f"invalid_ip: {exc}"}), 400
    indicator = _m.ti_manager.lookup_ip(ip)
    if indicator is None:
        return jsonify({"ip": ip, "found": False, "indicator": None}), 200
    return jsonify({"ip": ip, "found": True, "indicator": indicator.to_dict()}), 200


@observability_bp.route("/api/explain", methods=["POST"])
@require_roles("analyst")
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
    except Exception:
        return jsonify({"error": "invalid_request"}), 400
