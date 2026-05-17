"""Auth, audit, and admin endpoints — Step 39, second blueprint extraction."""
import logging
from datetime import datetime, timezone

from flask import Blueprint, g, jsonify, request

from src.auth.auth_service import UnifiedAuthService
from src.auth.decorators import require_roles
from src.auth.jwt_manager import get_jwt_manager

logger = logging.getLogger(__name__)
auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/api/auth/login", methods=["POST"])
def api_auth_login():
    import web_app.app as _m
    api_key = request.headers.get("X-API-Key", "").strip()
    if not api_key:
        logger.warning("api_auth_login: rejected — no X-API-Key provided")
        return jsonify({"error": "X-API-Key header is required"}), 401

    auth_ctx = UnifiedAuthService(_m.ops_store).authenticate_api_key(api_key)
    if auth_ctx is None:
        logger.warning("api_auth_login: rejected — invalid API key")
        return jsonify({"error": "Invalid API key"}), 401

    roles = sorted(auth_ctx.roles)
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", auth_ctx.username)

    try:
        token = get_jwt_manager().create_token(
            user_id=auth_ctx.user_id,
            username=username,
            roles=roles,
        )
        logger.info("api_auth_login: JWT issued user=%s roles=%s", username, roles)
        return jsonify({
            "token": token,
            "expires_in": 3600,
            "user": username,
            "roles": roles,
        }), 200
    except Exception:
        logger.exception("api_auth_login: token creation failed")
        return jsonify({"error": "Authentication failed"}), 401


@auth_bp.route("/api/auth/refresh", methods=["POST"])
def api_auth_refresh():
    REFRESH_GRACE_SECONDS = 300
    import time

    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not token:
        return jsonify({"error": "Missing token"}), 401

    try:
        _jm = get_jwt_manager()
        is_valid, payload, error = _jm.verify_token(token)

        if not is_valid:
            ok_exp, payload, _ = _jm.decode_ignoring_expiry(token)
            if not ok_exp or payload is None:
                return jsonify({"error": f"Invalid token: {error}"}), 401
            seconds_overdue = time.time() - payload["exp"]
            if seconds_overdue > REFRESH_GRACE_SECONDS:
                logger.warning(
                    "api_auth_refresh: rejected — token expired %ds ago (grace=%ds) user=%s",
                    int(seconds_overdue), REFRESH_GRACE_SECONDS, payload.get("sub"),
                )
                return jsonify({"error": "Token expired — please re-authenticate"}), 401

        if not payload:
            return jsonify({"error": "Could not parse token"}), 401

        new_token = _jm.create_token(
            user_id=payload["user_id"],
            username=payload["sub"],
            roles=payload.get("roles", []),
        )
        logger.info("api_auth_refresh: token refreshed user=%s", payload.get("sub"))
        return jsonify({"token": new_token, "expires_in": 3600, "user": payload.get("sub")}), 200

    except Exception:
        logger.exception("api_auth_refresh: failed")
        return jsonify({"error": "Token refresh failed"}), 401


@auth_bp.route("/api/auth/validate", methods=["GET"])
def api_auth_validate():
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()

    if not token:
        return jsonify({"valid": False, "error": "No token provided"}), 401

    is_valid, payload, error = get_jwt_manager().verify_token(token)

    if not is_valid:
        return jsonify({"valid": False, "error": error}), 401

    exp_dt = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)

    return jsonify({
        "valid": True,
        "user": payload.get("sub"),
        "user_id": payload.get("user_id"),
        "roles": payload.get("roles", []),
        "expires_at": exp_dt.isoformat(),
    }), 200


@auth_bp.route("/api/auth/runas", methods=["POST"])
@require_roles("admin")
def api_auth_runas():
    body = request.get_json(silent=True) or {}
    target_user = body.get("target_user", "")
    target_roles = body.get("target_roles", ["analyst"])

    if not target_user:
        return jsonify({"error": "target_user is required"}), 400

    try:
        admin_username = g.auth.username
        token = get_jwt_manager().create_token(
            user_id=target_user,
            username=target_user,
            roles=target_roles,
        )
        logger.info("api_auth_runas: delegated token issued admin=%s target=%s", admin_username, target_user)
        return jsonify({
            "token": token,
            "expires_in": 3600,
            "admin": admin_username,
            "target_user": target_user,
        }), 200
    except Exception:
        logger.exception("api_auth_runas: token creation failed")
        return jsonify({"error": "Failed to create delegated token"}), 500


@auth_bp.route("/api/audit/logs", methods=["GET"])
@require_roles("viewer")
def api_audit_logs():
    from flask import current_app
    limit = min(int(request.args.get("limit", 100)), 500)
    user_filter = request.args.get("user")

    try:
        logs = current_app.audit_log.get_logs(limit=limit)
        if user_filter:
            logs = [log for log in logs if log["user"] == user_filter]
        return jsonify(logs), 200
    except Exception as e:
        logger.error("Failed to retrieve audit logs: %s", e)
        return jsonify({"error": "Failed to retrieve logs"}), 500


@auth_bp.route("/api/audit/user-activity", methods=["GET"])
@require_roles("admin")
def api_audit_user_activity():
    from flask import current_app
    user = request.args.get("user")
    hours = int(request.args.get("hours", 24))

    if not user:
        return jsonify({"error": "user parameter is required"}), 400

    try:
        activity = current_app.audit_log.get_user_activity(user, hours=hours)
        return jsonify(activity), 200
    except Exception as e:
        logger.error("Failed to retrieve user activity: %s", e)
        return jsonify({"error": "Failed to retrieve activity"}), 500


@auth_bp.route("/api/auth/status", methods=["GET"])
def api_auth_status():
    return jsonify({
        "auth_enabled": True,
        "jwt_algorithm": "RS256",
        "rate_limiting_enabled": True,
        "ip_blocking_enabled": True,
        "audit_logging_enabled": True,
        "middleware_stack": [
            "rate_limiting",
            "ip_blocking",
            "security_headers",
            "audit_logging",
        ],
    }), 200


@auth_bp.route("/api/auth/revoke", methods=["POST"])
def api_auth_revoke():
    import web_app.app as _m
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not token:
        return jsonify({"error": "Authorization: Bearer <token> header required"}), 400

    try:
        _rs256 = get_jwt_manager()
        ok, payload, error = _rs256.decode_ignoring_expiry(token)
        if not ok or payload is None:
            return jsonify({"error": "Invalid token", "reason": error}), 401

        jti = payload.get("jti", "")
        if not jti:
            return jsonify({"error": "Token has no jti claim — cannot revoke"}), 400

        user_id = str(payload.get("user_id", payload.get("sub", "")))
        exp = payload.get("exp", 0)
        expires_at = (
            datetime.fromtimestamp(exp, tz=timezone.utc).isoformat() if exp else ""
        )
        revoked_at = datetime.now(timezone.utc).isoformat()

        _m.ops_store.add_revoked_token(
            jti=jti,
            user_id=user_id,
            expires_at=expires_at,
            revoked_at=revoked_at,
        )
        logger.info("api_auth_revoke: jti=%s user_id=%s", jti, user_id)
        return jsonify({"revoked": True, "jti": jti}), 200
    except Exception:
        logger.exception("api_auth_revoke: failed")
        return jsonify({"error": "Revocation failed"}), 500


@auth_bp.route("/api/admin/cleanup-tokens", methods=["POST"])
@require_roles("admin")
def api_admin_cleanup_tokens():
    import web_app.app as _m
    try:
        deleted = _m.ops_store.cleanup_revoked_tokens()
        logger.info("api_admin_cleanup_tokens: deleted=%d expired revocation records", deleted)
        return jsonify({"deleted": deleted}), 200
    except Exception:
        logger.exception("api_admin_cleanup_tokens: failed")
        return jsonify({"error": "Cleanup failed"}), 500


@auth_bp.route("/api/admin/alert-retention", methods=["POST"])
@require_roles("admin")
def api_admin_alert_retention():
    import web_app.app as _m
    try:
        deleted = _m._run_alert_retention()
        return jsonify({"deleted": deleted}), 200
    except Exception:
        logger.exception("api_admin_alert_retention: failed")
        return jsonify({"error": "Retention job failed"}), 500
