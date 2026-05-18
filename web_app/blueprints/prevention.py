"""Policy, actions, allowlist, and honeypot management endpoints — Step 39."""
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta

from flask import Blueprint, jsonify, request

from src.auth.decorators import require_roles
from src.input_sanitizer import SanitizationError, sanitize_string

logger = logging.getLogger(__name__)
prevention_bp = Blueprint("prevention", __name__)


@prevention_bp.route("/api/policy", methods=["GET", "POST"])
@require_roles("admin")
def api_policy():
    import web_app.app as _m
    if request.method == "GET":
        return jsonify(_m.prevention_service.policy.to_dict())

    payload = request.get_json(silent=True) or {}
    try:
        policy = _m.prevention_service.set_policy(
            mode=payload.get("mode"),
            block_ttl_seconds=payload.get("block_ttl_seconds"),
            confidence_block_threshold=payload.get("confidence_block_threshold"),
            risk_alert_threshold=payload.get("risk_alert_threshold", payload.get("alert_threshold")),
            risk_rate_limit_threshold=payload.get("risk_rate_limit_threshold", payload.get("rate_limit_threshold")),
            risk_temp_block_threshold=payload.get("risk_temp_block_threshold", payload.get("temp_block_threshold")),
            risk_block_threshold=payload.get("risk_block_threshold", payload.get("block_threshold")),
            block_requires_approval=payload.get("block_requires_approval"),
            risk_weight_confidence=payload.get("risk_weight_confidence"),
            risk_weight_severity=payload.get("risk_weight_severity"),
            risk_weight_frequency=payload.get("risk_weight_frequency"),
            dry_run=payload.get("dry_run"),
        )
        try:
            changed_by = sanitize_string(
                payload.get("changed_by", "admin_api") or "admin_api",
                max_length=100, allow_special_chars=True, allow_spaces=True,
            ) or "admin_api"
            reason = sanitize_string(
                payload.get("reason", "policy_update") or "policy_update",
                max_length=500, allow_special_chars=True, allow_spaces=True,
            ) or "policy_update"
        except SanitizationError:
            changed_by, reason = "admin_api", "policy_update"
        pv = _m.policy_store.update(policy.to_dict(), changed_by=changed_by, reason=reason)
        logger.info("policy_runtime_reloaded version=%s", pv.version)
        _m.ops_store.add_audit(
            event_type="policy_update",
            message=(
                f"mode={policy.mode}, ttl={policy.block_ttl_seconds}, "
                f"threshold={policy.confidence_block_threshold}, dry_run={policy.dry_run}, v={pv.version}"
            ),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        _m.metrics_service.inc("policy_updates_total")
        return jsonify(policy.to_dict())
    except Exception:
        return jsonify({"error": "invalid_request"}), 400


@prevention_bp.route("/api/policy/history", methods=["GET"])
@require_roles("admin")
def api_policy_history():
    import web_app.app as _m
    limit = request.args.get("limit", default=50, type=int)
    history = _m.policy_store.history(limit=max(1, min(limit, 500)))
    return jsonify({"count": len(history), "history": history})


@prevention_bp.route("/api/policy/rollback", methods=["POST"])
@require_roles("admin")
def api_policy_rollback():
    import web_app.app as _m
    payload = request.get_json(silent=True) or {}
    to_version = payload.get("to_version")
    if to_version is None:
        return jsonify({"error": "'to_version' is required"}), 400

    try:
        changed_by = sanitize_string(
            payload.get("changed_by", "admin_api") or "admin_api",
            max_length=100, allow_special_chars=True, allow_spaces=True,
        ) or "admin_api"
    except SanitizationError:
        changed_by = "admin_api"
    pv = _m.policy_store.rollback(int(to_version), changed_by=changed_by)
    if pv is None:
        return jsonify({"error": "version_not_found"}), 404

    config = pv.config
    policy = _m.prevention_service.set_policy(
        mode=config.get("mode"),
        block_ttl_seconds=config.get("block_ttl_seconds"),
        confidence_block_threshold=config.get("confidence_block_threshold"),
        risk_alert_threshold=config.get("risk_alert_threshold", config.get("alert_threshold")),
        risk_rate_limit_threshold=config.get("risk_rate_limit_threshold", config.get("rate_limit_threshold")),
        risk_temp_block_threshold=config.get("risk_temp_block_threshold", config.get("temp_block_threshold")),
        risk_block_threshold=config.get("risk_block_threshold", config.get("block_threshold")),
        block_requires_approval=config.get("block_requires_approval"),
        risk_weight_confidence=config.get("risk_weight_confidence"),
        risk_weight_severity=config.get("risk_weight_severity"),
        risk_weight_frequency=config.get("risk_weight_frequency"),
        dry_run=config.get("dry_run"),
    )
    logger.info("policy_rolled_back to version=%s", pv.version)
    _m.ops_store.add_audit(
        event_type="policy_rollback",
        message=f"rollback_to={to_version} new_version={pv.version}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"policy": policy.to_dict(), "version": pv.to_dict()})


@prevention_bp.route("/api/honeypot/config", methods=["GET"])
@require_roles("analyst")
def api_honeypot_config_get():
    import web_app.app as _m
    ips = _m.honeypot_engine.get_config()["honeypot_ips"]
    ports = _m.honeypot_engine.get_config()["honeypot_ports"]
    return jsonify({
        "honeypot_ips": ips,
        "honeypot_ports": ports,
        "enabled": _m.honeypot_engine.is_ready(),
        "engine_id": _m.honeypot_engine.engine_id,
    })


@prevention_bp.route("/api/honeypot/config", methods=["POST", "PATCH"])
@require_roles("admin")
def api_honeypot_config_update():
    import web_app.app as _m
    payload = request.get_json(silent=True) or {}

    if "honeypot_ips" in payload:
        new_ips = payload.get("honeypot_ips", [])
        if not isinstance(new_ips, list):
            return jsonify({"error": "'honeypot_ips' must be a list"}), 400
        _m.honeypot_engine.update_honeypot_ips(new_ips)
        _m.ops_store.add_audit(
            event_type="honeypot_config_update",
            message=f"honeypot_ips updated: {new_ips}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        _m.metrics_service.inc("honeypot_config_updates_total")

    if "honeypot_ports" in payload:
        new_ports = payload.get("honeypot_ports", [])
        if not isinstance(new_ports, list):
            return jsonify({"error": "'honeypot_ports' must be a list"}), 400
        try:
            new_ports = [int(p) for p in new_ports]
        except (ValueError, TypeError):
            return jsonify({"error": "honeypot_ports must be integers"}), 400
        _m.honeypot_engine.update_honeypot_ports(new_ports)
        _m.ops_store.add_audit(
            event_type="honeypot_config_update",
            message=f"honeypot_ports updated: {new_ports}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        _m.metrics_service.inc("honeypot_config_updates_total")

    config = _m.honeypot_engine.get_config()
    return jsonify({
        "honeypot_ips": config["honeypot_ips"],
        "honeypot_ports": config["honeypot_ports"],
        "enabled": config["enabled"],
        "updated": True,
    }), 200


@prevention_bp.route("/api/actions", methods=["GET", "POST"])
@require_roles("analyst")
def api_actions():
    import web_app.app as _m
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        action_type = str(payload.get("type") or payload.get("action_type") or payload.get("action") or "block").strip()
        target = str(payload.get("target_ip") or payload.get("target") or payload.get("ip") or "").strip()
        if not target:
            return jsonify({"error": "'target_ip' or 'target' is required"}), 400
        ttl = int(payload.get("ttl_seconds") or _m.prevention_service.policy.block_ttl_seconds)
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl)).isoformat() if ttl > 0 else None
        dry_run = bool(_m.prevention_service.policy.dry_run)
        executed = False
        status = "DRY_RUN" if dry_run else "ACTIVE"
        if not dry_run and action_type in {"block", "temp_block", "rate_limit"}:
            if action_type == "rate_limit":
                executed, status_text = _m.action_executor.rate_limit(target, ttl)
            else:
                executed, status_text = _m.action_executor.block_ip(target, ttl)
            status = "ACTIVE" if executed else status_text.upper()
        action_id = f"act_{uuid.uuid4().hex[:16]}"
        row = {
            "action_id": action_id,
            "action": action_type,
            "action_type": action_type,
            "target": target,
            "ip": target,
            "reason": payload.get("reason") or f"manual_{action_type}",
            "expires_at": expires_at,
            "created_at": now.isoformat(),
            "executed_at": now.isoformat() if executed or dry_run else None,
            "adapter": _m.SETTINGS.firewall_adapter,
            "dry_run": dry_run,
            "executed": executed or dry_run,
            "status": status,
        }
        _m.ops_store.save_action(row)
        _m.ops_store.add_audit(
            event_type="manual_action",
            message=json.dumps({"action_id": action_id, "type": action_type, "target": target}, separators=(",", ":")),
            created_at=now.isoformat(),
        )
        normalized = _m._normalize_action_payload(row)
        return jsonify({"ok": True, "status": 201, "data": normalized, "action": normalized}), 201

    limit = request.args.get("limit", default=50, type=int)
    actions = [_m._normalize_action_payload(action) for action in _m.ops_store.list_actions(limit=max(1, min(limit, 200)))]
    return jsonify({"count": len(actions), "actions": actions})


@prevention_bp.route("/api/actions/pending", methods=["GET"])
@require_roles("analyst")
def api_actions_pending():
    import web_app.app as _m
    rows = _m.ops_store.list_pending_actions(limit=200)
    actions = [_m._normalize_action_payload(row) for row in rows]
    return jsonify({"count": len(actions), "actions": actions})


@prevention_bp.route("/api/actions/<action_id>/approve", methods=["POST"])
@require_roles("admin")
def api_action_approve(action_id: str):
    import web_app.app as _m
    body = request.get_json(silent=True) or {}
    ttl_seconds = body.get("ttl_seconds")
    result = _m.action_executor.approve_pending_block(
        action_id,
        _m.prevention_service.policy,
        ttl_seconds=int(ttl_seconds) if ttl_seconds is not None else None,
    )
    if not result.get("ok") and result.get("error") == "not_found":
        return jsonify({"error": "pending action not found"}), 404
    _m.ops_store.add_audit(
        event_type="action_approved",
        message=json.dumps(result, separators=(",", ":")),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify(result), (200 if result.get("ok") else 500)


@prevention_bp.route("/api/actions/cleanup", methods=["POST"])
@require_roles("admin")
def api_actions_cleanup():
    import web_app.app as _m
    payload = request.get_json(silent=True) or {}
    now_iso = payload.get("now")
    removed = _m.action_executor.cleanup_expired_actions(now_iso=now_iso)
    if removed:
        _m.ops_store.add_audit(
            event_type="actions_cleanup",
            message=f"removed_expired_actions={removed}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        _m.metrics_service.inc("expired_actions_cleaned_total", amount=removed)
    return jsonify({"removed": removed})


@prevention_bp.route("/api/allowlist", methods=["GET"])
@require_roles("analyst")
def api_allowlist_get():
    import web_app.app as _m
    entries = _m.allowlist.list_entries()
    formatted = [_m._format_allowlist_entry(entry) for entry in entries]
    return jsonify({"entries": entries, "allowlist": formatted}), 200


@prevention_bp.route("/api/allowlist", methods=["POST"])
@require_roles("admin")
def api_allowlist_add():
    import web_app.app as _m
    body = request.get_json(silent=True) or {}
    entry = str(body.get("entry", "")).strip()
    if not entry:
        return jsonify({"error": "'entry' is required"}), 400
    try:
        reason = sanitize_string(
            body.get("reason", "") or "", max_length=500, allow_special_chars=True, allow_spaces=True
        )
    except SanitizationError as exc:
        return jsonify({"error": f"invalid_input: {exc}"}), 400
    added = _m.allowlist.add(entry, reason=reason)
    _m.ops_store.add_audit(
        event_type="allowlist_add",
        message=f"entry={entry} reason={reason} new={added}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"entry": entry, "added": added}), (201 if added else 200)


@prevention_bp.route("/api/allowlist/<path:entry>", methods=["DELETE"])
@require_roles("admin")
def api_allowlist_remove(entry: str):
    import web_app.app as _m
    removed = _m.allowlist.remove(entry)
    if not removed:
        return jsonify({"error": "entry not found"}), 404
    _m.ops_store.add_audit(
        event_type="allowlist_remove",
        message=f"entry={entry}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return jsonify({"entry": entry, "removed": True}), 200


@prevention_bp.route("/api/honeypots", methods=["GET"])
@require_roles("analyst")
def api_honeypots():
    import web_app.app as _m
    config = _m.honeypot_engine.get_config()
    honeypots = []
    for ip in config.get("honeypot_ips", []):
        honeypots.append({
            "id": ip,
            "ip": ip,
            "name": f"Honeypot {ip}",
            "ports": config.get("honeypot_ports", []),
            "enabled": bool(config.get("enabled")),
            "status": "active" if config.get("enabled") else "inactive",
        })
    return jsonify({"honeypots": honeypots, "count": len(honeypots), "config": config})


@prevention_bp.route("/api/honeypots/<honeypot_id>/toggle", methods=["POST"])
@require_roles("admin")
def api_honeypot_toggle(honeypot_id: str):
    import web_app.app as _m
    body = request.get_json(silent=True) or {}
    enabled = bool(body.get("enabled", True))
    _m.engine_registry.set_enabled(_m.honeypot_engine.engine_id, enabled)
    return jsonify({"id": honeypot_id, "enabled": enabled})
