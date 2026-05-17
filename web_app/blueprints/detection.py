"""Detection, prediction, alert management, engine control — Step 39."""
import json
import logging
import re
import time
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from src.auth.decorators import require_roles
from src.input_sanitizer import SanitizationError, sanitize_id, sanitize_string
from src.schema import DEFAULT_FEATURE_ROW, NUMERIC_FEATURES
from src.validation_schemas import validate_detect_request, validate_predict_request

logger = logging.getLogger(__name__)
detection_bp = Blueprint("detection", __name__)

NUMERIC_MODEL_COLUMNS = NUMERIC_FEATURES


@detection_bp.route("/api/predict", methods=["POST"])
@require_roles("analyst")
def api_predict():
    import web_app.app as _m
    if not _m.ensure_detection_service():
        return jsonify({"error": "No trained model found"}), 503

    payload = request.get_json(silent=True) or {}
    is_valid, error_msg = validate_predict_request(payload)
    if not is_valid:
        return jsonify({"error": f"Invalid request: {error_msg}"}), 400

    features = payload.get("features", {})
    profile = payload.get("profile", "balanced")

    if not isinstance(features, dict) or not features:
        return jsonify({"error": "'features' must be a non-empty object"}), 400

    try:
        for col in NUMERIC_MODEL_COLUMNS:
            if col in features:
                features[col] = float(features[col])
        source = payload.get("source_ip", payload.get("source", "unknown"))
        _m.metrics_service.inc("predictions_total")
        result = _m.detection_service.predict_from_features(
            features,
            profile=profile,
            source_ip=source,
            attack_type=payload.get("attack_type"),
        )
        response = result.to_dict()
        recent_actions = _m.ops_store.list_actions(limit=20)
        prevention_action = next(
            (
                action
                for action in recent_actions
                if str(action.get("target", "")).strip() == str(source).strip()
                and (time.time() - _m._to_epoch_seconds(action.get("created_at"), default=0.0)) <= 60.0
            ),
            None,
        )
        response["prevention_action"] = prevention_action
        return jsonify(response)
    except Exception as exc:
        logger.exception("API predict error")
        return jsonify({"error": str(type(exc).__name__), "message": str(exc), "debug": True}), 400


@detection_bp.route("/api/alerts", methods=["GET"])
@require_roles("analyst")
def api_alerts():
    import web_app.app as _m
    limit = request.args.get("limit", default=50, type=int)
    offset = request.args.get("offset", default=0, type=int)
    severity = request.args.get("severity", default=None, type=str)
    status = request.args.get("status", default=None, type=str)
    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)
    alerts = [
        _m._normalize_alert_payload(alert)
        for alert in _m.ops_store.list_alerts(
            limit=safe_limit, offset=safe_offset, severity=severity, status=status
        )
    ]
    return jsonify({"count": len(alerts), "offset": safe_offset, "limit": safe_limit, "alerts": alerts})


@detection_bp.route("/api/alerts/dismiss", methods=["POST"])
@require_roles("analyst")
def api_alerts_dismiss():
    import web_app.app as _m
    body = request.get_json(silent=True) or {}
    alert_ids = body.get("alert_ids")
    if alert_ids is None:
        alert_id = body.get("alert_id")
        alert_ids = [alert_id] if alert_id else []
    if not isinstance(alert_ids, list) or not alert_ids:
        return jsonify({"error": "'alert_id' or 'alert_ids' is required"}), 400

    dismissed, missing = [], []
    for alert_id in alert_ids:
        try:
            updated = _m.ops_store.update_alert(str(alert_id), status="closed", close_reason="dismissed")
        except ValueError:
            updated = False
        (dismissed if updated else missing).append(str(alert_id))
    if dismissed:
        _m.ops_store.add_audit(
            event_type="alerts_dismissed",
            message=json.dumps({"dismissed": dismissed, "missing": missing}, separators=(",", ":")),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    return jsonify({"dismissed": dismissed, "missing": missing, "count": len(dismissed)}), 200


@detection_bp.route("/api/alerts/<alert_id>/feedback", methods=["POST"])
@require_roles("analyst")
def api_alert_feedback(alert_id: str):
    import web_app.app as _m
    body = request.get_json(silent=True) or {}
    verdict = str(body.get("verdict", "")).strip().lower()
    try:
        engine_id = sanitize_id(body.get("engine_id", "ml_engine") or "ml_engine", max_length=100)
        rule_id = sanitize_id(body.get("rule_id", "model") or "model", max_length=100)
    except SanitizationError as exc:
        return jsonify({"error": f"invalid_input: {exc}"}), 400
    if verdict == "fp":
        _m.fp_manager.report_fp(engine_id, rule_id, alert_id=alert_id)
    elif verdict == "tp":
        _m.fp_manager.report_tp(engine_id, rule_id)
    else:
        return jsonify({"error": "verdict must be 'fp' or 'tp'"}), 400
    return jsonify({
        "alert_id": alert_id,
        "verdict": verdict,
        "suppressed": _m.fp_manager.is_suppressed(engine_id, rule_id),
    }), 200


@detection_bp.route("/api/alerts/<alert_id>", methods=["PATCH"])
@require_roles("analyst")
def api_alert_update(alert_id: str):
    import web_app.app as _m
    if not re.match(r'^[a-zA-Z0-9_-]+$', alert_id):
        return jsonify({"error": "invalid_alert_id"}), 400
    body = request.get_json(silent=True) or {}
    status = body.get("status")
    assignee = body.get("assignee")
    close_reason = body.get("close_reason")
    if status is None and assignee is None and close_reason is None:
        return jsonify({"error": "at least one of 'status', 'assignee', 'close_reason' is required"}), 400

    try:
        if status is not None:
            status = sanitize_string(status, max_length=50, allow_special_chars=False)
        if assignee is not None:
            assignee = sanitize_string(assignee, max_length=200, allow_special_chars=True, allow_spaces=True)
        if close_reason is not None:
            close_reason = sanitize_string(close_reason, max_length=500, allow_special_chars=True, allow_spaces=True)
    except SanitizationError as exc:
        return jsonify({"error": f"invalid_input: {exc}"}), 400

    try:
        updated = _m.ops_store.update_alert(alert_id, status=status, assignee=assignee, close_reason=close_reason)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not updated:
        return jsonify({"error": "alert_not_found"}), 404

    _m.ops_store.add_audit(
        event_type="alert_update",
        message=json.dumps({"alert_id": alert_id, "status": status, "assignee": assignee, "close_reason": close_reason}, separators=(",", ":")),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"updated": True, "alert_id": alert_id}), 200


@detection_bp.route("/api/fp-suppressions", methods=["GET"])
@require_roles("analyst")
def api_fp_suppressions_list():
    import web_app.app as _m
    return jsonify({"count": len(_m.ops_store.list_fp_suppressions()), "suppressions": _m.ops_store.list_fp_suppressions()})


@detection_bp.route("/api/fp-suppressions", methods=["POST"])
@require_roles("admin")
def api_fp_suppression_add():
    import web_app.app as _m
    body = request.get_json(silent=True) or {}
    try:
        engine_id = sanitize_id(body.get("engine_id", "") or "", max_length=100)
        rule_id = sanitize_id(body.get("rule_id", "") or "", max_length=100)
    except SanitizationError as exc:
        return jsonify({"error": f"invalid_input: {exc}"}), 400
    if not engine_id or not rule_id:
        return jsonify({"error": "'engine_id' and 'rule_id' are required"}), 400
    newly = _m.fp_manager.suppress(engine_id, rule_id)
    return jsonify({"engine_id": engine_id, "rule_id": rule_id, "suppressed": True, "new": bool(newly)})


@detection_bp.route("/api/fp-suppressions/<engine_id>/<rule_id>", methods=["DELETE"])
@require_roles("admin")
def api_fp_suppression_remove(engine_id: str, rule_id: str):
    import web_app.app as _m
    removed = _m.fp_manager.unsuppress(engine_id, rule_id)
    if not removed:
        return jsonify({"error": "suppression_not_found"}), 404
    return jsonify({"engine_id": engine_id, "rule_id": rule_id, "removed": True}), 200


@detection_bp.route("/api/detect", methods=["POST"])
@require_roles("analyst")
def api_detect():
    import web_app.app as _m
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

        _m.metrics_service.inc("predictions_total")
        try:
            engine_features = _m.enrich_single_row(features)
        except Exception:
            logger.warning("Feature enrichment failed in /api/detect; falling back to raw features", exc_info=True)
            engine_features = features

        eval_start = time.monotonic()
        results = _m.engine_registry.evaluate_all(engine_features)
        _m.metrics_service.observe_latency("engine_eval_latency", eval_start)
        _m.metrics_service.inc("engine_evaluations_total")
        aggregated = _m.engine_aggregator.aggregate(results)

        if aggregated.verdict in ("attack", "suspicious"):
            _m.metrics_service.inc("alerts_total")
        if aggregated.verdict == "attack":
            _m.metrics_service.inc("engine_attacks_total")

        for result in results:
            if result.verdict == "attack":
                _m.metrics_service.inc(f"engine_{result.engine_id}_attacks_total")
            _m.metrics_service.inc(f"engine_{result.engine_id}_evaluations_total")

        return jsonify(aggregated.to_dict())
    except Exception as exc:
        logger.exception("Multi-engine detect error")
        return jsonify({"error": str(type(exc).__name__), "message": str(exc), "debug": True}), 400


@detection_bp.route("/api/detection/analyze", methods=["POST"])
@require_roles("analyst")
def api_detection_analyze():
    import web_app.app as _m
    payload = request.get_json(silent=True) or {}
    target = str(payload.get("target", "")).strip()
    input_type = str(payload.get("input_type", "ip")).strip().lower() or "ip"
    if not target:
        return jsonify({"error": "'target' is required"}), 400
    try:
        target = sanitize_string(target, max_length=256, allow_special_chars=True, allow_spaces=False)
        input_type = sanitize_string(input_type, max_length=20, allow_special_chars=False, allow_spaces=False)
    except SanitizationError as exc:
        return jsonify({"error": f"invalid_input: {exc}"}), 400

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
        enriched = _m.enrich_single_row(features)
    except Exception:
        enriched = features
    results = _m.engine_registry.evaluate_all(enriched)
    aggregated = _m.engine_aggregator.aggregate(results)
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


@detection_bp.route("/api/engines", methods=["GET"])
@require_roles("analyst")
def api_engines():
    import web_app.app as _m
    return jsonify({"engines": _m.engine_registry.list_engines()})


@detection_bp.route("/api/engines/<engine_id>/toggle", methods=["POST"])
@require_roles("admin")
def api_toggle_engine(engine_id: str):
    import web_app.app as _m
    payload = request.get_json(silent=True) or {}
    enabled = payload.get("enabled")
    if enabled is None:
        return jsonify({"error": "'enabled' is required"}), 400
    ok = _m.engine_registry.set_enabled(engine_id, bool(enabled))
    if not ok:
        return jsonify({"error": f"engine '{engine_id}' not found"}), 404
    return jsonify({"engine_id": engine_id, "enabled": bool(enabled)})
