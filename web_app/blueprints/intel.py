"""Intel: incidents, temporal, entity, alert filters, models, TI, and detections history — Step 39."""
import json
import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from src.auth.decorators import require_roles

logger = logging.getLogger(__name__)
intel_bp = Blueprint("intel", __name__)


@intel_bp.route("/api/incidents", methods=["GET"])
@require_roles("analyst")
def api_incidents_list():
    import web_app.app as _m
    limit = request.args.get("limit", default=50, type=int)
    incidents = _m.incident_aggregator.get_incidents(limit=max(1, min(limit, 500)))
    return jsonify({"count": len(incidents), "incidents": incidents})


@intel_bp.route("/api/incidents/<incident_id>", methods=["GET"])
@require_roles("analyst")
def api_incident_details(incident_id: str):
    import web_app.app as _m
    details = _m.incident_aggregator.get_incident_details(incident_id)
    if details is None:
        return jsonify({"error": "incident not found"}), 404
    return jsonify(details)


@intel_bp.route("/api/activities", methods=["GET"])
@require_roles("analyst")
def api_activities_list():
    import web_app.app as _m
    limit = request.args.get("limit", default=50, type=int)
    activities = _m.incident_aggregator.get_activities(limit=max(1, min(limit, 500)))
    return jsonify({"count": len(activities), "activities": activities})


@intel_bp.route("/api/temporal/patterns", methods=["GET"])
@require_roles("analyst")
def api_temporal_patterns_list():
    import web_app.app as _m
    patterns = []
    for pattern_name, pattern in _m.temporal_correlation_engine.patterns.items():
        patterns.append({
            "name": pattern_name,
            "steps": pattern.steps,
            "description": pattern.description,
        })
    return jsonify({"count": len(patterns), "patterns": patterns})


@intel_bp.route("/api/temporal/patterns", methods=["POST"])
@require_roles("admin")
def api_temporal_patterns_register():
    import web_app.app as _m
    body = request.get_json(silent=True) or {}
    pattern_name = body.get("name", "").strip()
    pattern_steps = body.get("steps", [])
    description = body.get("description", "")

    if not pattern_name or not pattern_steps:
        return jsonify({"error": "name and steps are required"}), 400

    if not isinstance(pattern_steps, list) or len(pattern_steps) < 2:
        return jsonify({"error": "pattern must have at least 2 steps"}), 400

    try:
        _m.temporal_correlation_engine.register_pattern(pattern_name, pattern_steps)
        _m.ops_store.add_audit(
            event_type="temporal_pattern_registered",
            message=f"pattern={pattern_name}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        _m.metrics_service.inc("temporal_patterns_registered_total")
        return jsonify({
            "ok": True,
            "pattern_name": pattern_name,
            "message": "Temporal correlation pattern registered successfully"
        }), 201
    except Exception as e:
        logger.exception("Failed to register temporal pattern: %s", pattern_name)
        return jsonify({"error": f"Failed to register pattern: {str(e)}"}), 500


@intel_bp.route("/api/temporal/state/<source_ip>", methods=["GET"])
@require_roles("analyst")
def api_temporal_state(source_ip: str):
    import web_app.app as _m
    if source_ip not in _m.temporal_correlation_engine.temporal_store.events_by_ip:
        return jsonify({"source_ip": source_ip, "events": [], "status": "no_events"}), 200

    events = _m.temporal_correlation_engine.temporal_store.events_by_ip[source_ip]
    return jsonify({
        "source_ip": source_ip,
        "event_count": len(events),
        "events": [{"type": e["type"], "timestamp": e["timestamp"], "confidence": e.get("confidence")} for e in events[-20:]],
        "status": "tracking"
    })


@intel_bp.route("/api/entity/enrich/<source_ip>", methods=["GET"])
@require_roles("analyst")
def api_entity_enrich(source_ip: str):
    import web_app.app as _m
    try:
        enriched = _m.entity_enrichment_engine.enrich(source_ip)
        threat_level = _m.entity_enrichment_engine.get_threat_level(enriched)
        result = enriched.to_dict()
        result["threat_level"] = threat_level
        return jsonify(result)
    except Exception as e:
        logger.exception("Enrichment failed for IP %s", source_ip)
        return jsonify({"error": f"Enrichment failed: {str(e)}"}), 500


@intel_bp.route("/api/entity/<source_ip>/threat-level", methods=["GET"])
@require_roles("analyst")
def api_entity_threat_level(source_ip: str):
    import web_app.app as _m
    try:
        enriched = _m.entity_enrichment_engine.enrich(source_ip)
        threat_level = _m.entity_enrichment_engine.get_threat_level(enriched)
        confidence = enriched.enrichment_confidence
        return jsonify({
            "source_ip": source_ip,
            "threat_level": threat_level,
            "confidence": confidence,
            "last_enriched": enriched.timestamp,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@intel_bp.route("/api/alerts/filter-rules", methods=["GET"])
@require_roles("analyst")
def api_alert_filter_rules():
    import web_app.app as _m
    all_rules = _m.alert_filter.get_all_rules()
    stats = _m.alert_filter.get_rule_stats()
    return jsonify({
        "rules": all_rules,
        "stats": stats,
    })


@intel_bp.route("/api/alerts/filter-rules/exclude", methods=["POST"])
@require_roles("admin")
def api_alert_add_exclude_rule():
    import web_app.app as _m
    from src.ips.alert_filter import ExcludeRule
    body = request.get_json(silent=True) or {}
    try:
        rule = ExcludeRule(
            rule_id=body.get("rule_id", f"exclude_{int(datetime.utcnow().timestamp())}"),
            name=body.get("name", "Unnamed exclude rule"),
            description=body.get("description", ""),
            conditions=body.get("conditions", {}),
            priority=body.get("priority", 0),
        )
        if _m.alert_filter.add_exclude_rule(rule):
            _m.ops_store.add_audit(
                event_type="exclude_rule_added",
                message=f"rule={rule.rule_id}",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            _m.metrics_service.inc("alert_filter_rules_created_total")
            return jsonify({"ok": True, "rule_id": rule.rule_id}), 201
        return jsonify({"error": "Rule with this ID already exists"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@intel_bp.route("/api/alerts/filter-rules/ignore", methods=["POST"])
@require_roles("admin")
def api_alert_add_ignore_rule():
    import web_app.app as _m
    from src.ips.alert_filter import IgnoreRule
    body = request.get_json(silent=True) or {}
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
        if _m.alert_filter.add_ignore_rule(rule):
            _m.ops_store.add_audit(
                event_type="ignore_rule_added",
                message=f"rule={rule.rule_id}",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            _m.metrics_service.inc("alert_filter_rules_created_total")
            return jsonify({"ok": True, "rule_id": rule.rule_id}), 201
        return jsonify({"error": "Rule with this ID already exists"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@intel_bp.route("/api/alerts/filter-rules/merge", methods=["POST"])
@require_roles("admin")
def api_alert_add_merge_rule():
    import web_app.app as _m
    from src.ips.alert_filter import MergeRule
    body = request.get_json(silent=True) or {}
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
        if _m.alert_filter.add_merge_rule(rule):
            _m.ops_store.add_audit(
                event_type="merge_rule_added",
                message=f"rule={rule.rule_id}",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            _m.metrics_service.inc("alert_filter_rules_created_total")
            return jsonify({"ok": True, "rule_id": rule.rule_id}), 201
        return jsonify({"error": "Rule with this ID already exists"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@intel_bp.route("/api/alerts/filter-rules/<rule_id>", methods=["DELETE"])
@require_roles("admin")
def api_alert_delete_rule(rule_id: str):
    import web_app.app as _m
    if _m.alert_filter.remove_rule(rule_id):
        _m.ops_store.add_audit(
            event_type="filter_rule_deleted",
            message=f"rule={rule_id}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        _m.metrics_service.inc("alert_filter_rules_deleted_total")
        return jsonify({"ok": True, "message": f"Rule {rule_id} deleted"})
    return jsonify({"error": "Rule not found"}), 404


@intel_bp.route("/api/alerts/filter-stats", methods=["GET"])
@require_roles("analyst")
def api_alert_filter_stats():
    import web_app.app as _m
    stats = _m.alert_filter.get_rule_stats()
    return jsonify(stats)


@intel_bp.route("/api/models/registry", methods=["GET"])
@require_roles("analyst")
def api_model_registry():
    import web_app.app as _m
    limit = request.args.get("limit", default=50, type=int)
    entries = _m.model_registry.list_entries(limit=max(1, min(limit, 200)))
    return jsonify({"count": len(entries), "models": entries})


@intel_bp.route("/api/models", methods=["GET"])
@require_roles("analyst")
def api_models_catalog():
    import web_app.app as _m
    registry_entries = _m.model_registry.list_entries(limit=100)
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
        for name in sorted(_m.all_models.keys() or []):
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


@intel_bp.route("/api/threat-intelligence", methods=["GET"])
@require_roles("analyst")
def api_threat_intelligence_catalog():
    import web_app.app as _m
    stats = _m.ti_manager.stats()
    feeds = _m.ti_manager.feed_summary()
    indicators = []
    if hasattr(_m.ti_manager.cache, "all_indicators"):
        indicators = [item.to_dict() for item in _m.ti_manager.cache.all_indicators()]
    return jsonify({
        "sources": feeds,
        "indicators": indicators,
        "stats": stats,
        "threats": [],
    })


@intel_bp.route("/api/detections/history", methods=["GET"])
@require_roles("analyst")
def api_detections_history():
    import web_app.app as _m
    limit = request.args.get("limit", default=50, type=int)
    limit = max(1, min(limit, 200))
    audits = _m.ops_store._fetchall(
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
