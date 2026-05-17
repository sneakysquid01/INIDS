from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from functools import wraps
from typing import Callable

from flask import current_app, g, jsonify, request

from src.auth.models import AuthContext
from src.auth.auth_service import UnifiedAuthService

logger = logging.getLogger(__name__)

# Routes that require no authentication.
PUBLIC_ROUTES: frozenset[str] = frozenset(
    {
        "/health",
        "/api/health",
        "/api/health/live",
        "/api/health/ready",
        "/api/auth/login",
        "/api/auth/refresh",
        "/api/auth/revoke",
        "/api/auth/validate",
        "/api/auth/status",
    }
)


def _compat_enabled() -> bool:
    return os.environ.get("INIDS_AUTH_COMPAT", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_ops_store():
    return getattr(current_app, "ops_store", None)


def _try_unified_auth(ops_store) -> AuthContext | None:
    """Attempt auth via UnifiedAuthService (RS256 JWT then X-API-Key)."""
    if ops_store is None:
        return None
    svc = UnifiedAuthService(ops_store)

    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.startswith("Bearer "):
        ctx = svc.authenticate_jwt(auth_header[7:])
        if ctx is not None:
            return ctx

    api_key = request.headers.get("X-API-Key", "").strip()
    if api_key:
        ctx = svc.authenticate_api_key(api_key)
        if ctx is not None:
            return ctx

    return None


def _try_legacy_auth(required_roles: frozenset[str]) -> AuthContext | None:
    """Fallback to legacy AuthService (env-var API keys). Logs WARNING when used."""
    try:
        from src.auth_service import _auth_service, ROLE_RANK, SENSOR_ALLOWED_ENDPOINTS

        token = request.headers.get("X-API-Key", "").strip()
        if not token:
            return None
        principal = _auth_service.principals.get(token)
        if principal is None:
            return None

        if principal.role == "sensor":
            if request.path not in SENSOR_ALLOWED_ENDPOINTS:
                return None
        else:
            min_required_rank = min(
                (ROLE_RANK.get(r, 999) for r in required_roles), default=0
            )
            if ROLE_RANK.get(principal.role, -1) < min_required_rank:
                return None

        return AuthContext(
            user_id=f"legacy:{principal.role}",
            username=principal.role,
            roles=frozenset({principal.role}),
        )
    except Exception as exc:
        logger.error("Legacy auth fallback error: %s", exc)
        return None


def _log_auth_audit(ops_store, event_type: str, ctx: AuthContext, path: str) -> None:
    if ops_store is None:
        return
    try:
        ops_store.add_audit(
            event_type,
            f"user={ctx.username} roles={','.join(sorted(ctx.roles))} path={path}",
            datetime.now(timezone.utc).isoformat(),
        )
    except Exception:
        pass


def public_route(func: Callable) -> Callable:
    """Mark a route handler as publicly accessible (no auth required)."""
    func._is_public_route = True
    return func


def require_roles(*roles: str) -> Callable:
    """Decorator: require the caller to hold at least one of the given roles.

    Auth flow:
    1. Try UnifiedAuthService (RS256 JWT → X-API-Key OpsStore lookup).
    2. If INIDS_AUTH_COMPAT=true and step 1 fails, fall back to legacy AuthService.
       Divergence (new rejects / old accepts) is logged at WARNING.
    3. Role intersection check — 403 if caller holds none of the required roles.
    4. Sets flask.g.auth = AuthContext for downstream handlers.
    5. Logs auth_success / authz_denied to OpsStore.audits.
    """

    def decorator(func: Callable) -> Callable:
        func._required_roles = set(roles)

        @wraps(func)
        def wrapper(*args, **kwargs):
            ops_store = _get_ops_store()
            ctx = _try_unified_auth(ops_store)

            if ctx is None and _compat_enabled():
                role_set = frozenset(roles)
                ctx = _try_legacy_auth(role_set)
                if ctx is not None:
                    logger.warning(
                        "INIDS_AUTH_COMPAT: legacy auth accepted — "
                        "user=%s endpoint=%s required_roles=%s",
                        ctx.username,
                        request.path,
                        sorted(roles),
                    )

            if ctx is None:
                return (
                    jsonify({"error": "unauthorized", "reason": "access_denied"}),
                    401,
                )

            if roles and not (ctx.roles & frozenset(roles)):
                _log_auth_audit(ops_store, "authz_denied", ctx, request.path)
                return (
                    jsonify({"error": "forbidden", "reason": "insufficient_role"}),
                    403,
                )

            g.auth = ctx
            _log_auth_audit(ops_store, "auth_success", ctx, request.path)

            # Tier 2: per-user rate limit (200 req/min) via UnifiedRateLimiter.
            try:
                from src.rate_limiter import get_unified_rate_limiter as _get_rl
                _rl = _get_rl()
                if _rl is not None:
                    _allowed, _retry = _rl.check_user(ctx.user_id)
                    if not _allowed:
                        return (
                            jsonify({"error": "rate_limited", "retry_after_seconds": _retry}),
                            429,
                        )
            except Exception as _rl_exc:
                logger.error("require_roles: per-user rate check error: %s", _rl_exc)

            return func(*args, **kwargs)

        return wrapper

    return decorator
