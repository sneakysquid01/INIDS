"""System operations: anomaly, escalation, FP, investigations, playbooks, captures — Step 39."""
import json
import logging
from datetime import datetime, timezone
from flask import Blueprint, current_app, jsonify, request

from src.auth.decorators import require_roles

logger = logging.getLogger(__name__)
system_bp = Blueprint("system", __name__)


def _default_playbooks() -> list:
    return [
        {
            "id": "pb_block_source",
            "name": "Block Source IP",
            "type": "containment",
            "description": "Create a prevention action for a malicious source.",
            "enabled": True,
            "action_count": 1,
            "execution_count": 0,
            "created_date": "2026-01-01T00:00:00+00:00",
            "last_run_date": None,
        },
        {
            "id": "pb_collect_context",
            "name": "Collect Investigation Context",
            "type": "investigation",
            "description": "Review recent alerts, entity enrichment, and risk history.",
            "enabled": True,
            "action_count": 3,
            "execution_count": 0,
            "created_date": "2026-01-01T00:00:00+00:00",
            "last_run_date": None,
        },
    ]


@system_bp.route("/api/anomaly/status", methods=["GET"])
@require_roles("analyst")
def api_anomaly_status():
    import web_app.app as _m
    buf = _m.anomaly_engine.buffer_status()
    buf["engine_id"] = _m.anomaly_engine.engine_id
    buf["enabled"] = _m.engine_registry.is_enabled(_m.anomaly_engine.engine_id)
    buf["ready"] = _m.anomaly_engine.is_ready()
    return jsonify(buf)


@system_bp.route("/api/escalation/summary", methods=["GET"])
@require_roles("analyst")
def api_escalation_summary():
    import web_app.app as _m
    summary = _m.escalation_tracker.summary()
    return jsonify({
        "tracked_count": len(summary),
        "escalation": summary,
    })


@system_bp.route("/api/escalation/evict", methods=["POST"])
@require_roles("admin")
def api_escalation_evict():
    import web_app.app as _m
    removed = _m.escalation_tracker.evict_stale()
    summary = _m.escalation_tracker.summary()
    _m.ops_store.add_audit(
        event_type="escalation_evict",
        message=f"evicted={removed}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"evicted": removed, "remaining": len(summary)})


@system_bp.route("/api/fp-stats", methods=["GET"])
@require_roles("analyst")
def api_fp_stats():
    import web_app.app as _m
    return jsonify({"stats": _m.fp_manager.stats()})


@system_bp.route("/api/investigations", methods=["GET"])
@require_roles("analyst")
def api_investigations():
    import web_app.app as _m
    import uuid
    alerts = _m.ops_store.list_alerts(limit=100)
    investigations = []
    for alert in alerts:
        alert_id = str(alert.get("id", ""))
        investigations.append({
            "id": f"inv_{alert_id[-8:] or uuid.uuid4().hex[:8]}",
            "title": f"Investigation for {alert.get('prediction', 'alert')}",
            "description": alert.get("reason") or "Generated from alert activity",
            "severity": str(alert.get("severity", "low")).lower(),
            "status": "closed" if str(alert.get("status", "")).lower() == "closed" else "open",
            "investigator": alert.get("assignee") or "Unassigned",
            "created_date": alert.get("timestamp"),
            "evidence_count": 1,
            "alert_id": alert_id,
        })
    return jsonify({"count": len(investigations), "investigations": investigations})


@system_bp.route("/api/playbooks", methods=["GET"])
@require_roles("analyst")
def api_playbooks():
    pb = _default_playbooks()
    return jsonify({"playbooks": pb, "count": len(pb)})


@system_bp.route("/api/playbooks/<playbook_id>/execute", methods=["POST"])
@require_roles("analyst")
def api_playbook_execute(playbook_id: str):
    import web_app.app as _m
    _m.ops_store.add_audit(
        event_type="playbook_execute",
        message=json.dumps({"playbook_id": playbook_id}, separators=(",", ":")),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"ok": True, "playbook_id": playbook_id, "status": "queued"}), 202


@system_bp.route("/api/capture/start", methods=["POST"])
@require_roles("analyst")
def api_capture_start():
    payload = request.get_json(silent=True) or {}
    app = current_app._get_current_object()
    app.config["CAPTURE_RUNNING"] = True
    app.config["CAPTURE_CONFIG"] = {
        "interface": payload.get("interface", "eth0"),
        "bpf_filter": payload.get("bpf_filter", ""),
        "packet_limit": int(payload.get("packet_limit") or 1000),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify({"ok": True, "status": "capturing", **app.config["CAPTURE_CONFIG"]}), 202


@system_bp.route("/api/capture/stop", methods=["POST"])
@require_roles("analyst")
def api_capture_stop():
    app = current_app._get_current_object()
    app.config["CAPTURE_RUNNING"] = False
    return jsonify({"ok": True, "status": "stopped"}), 200
