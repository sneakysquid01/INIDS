from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import wraps
from typing import Callable

from flask import current_app, g, jsonify, request

from src.auth.models import AuthContext
from src.auth.auth_service import UnifiedAuthService

logger = logging.getLogger(__name__)

# Role hierarchy: admin can access analyst and viewer resources; analyst can access viewer resources.
_ROLE_HIERARCHY: dict[str, frozenset[str]] = {
    "admin": frozenset({"admin", "analyst", "viewer"}),
    "analyst": frozenset({"analyst", "viewer"}),
    "viewer": frozenset({"viewer"}),
    "sensor": frozenset({"sensor"}),
}


def _effective_roles(roles: frozenset[str]) -> frozenset[str]:
    effective: set[str] = set()
    for role in roles:
        effective |= _ROLE_HIERARCHY.get(role, {role})
    return frozenset(effective)


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


class AuthStoreUnboundError(RuntimeError):
    """Raised when ops_store is not attached to current_app."""


def _get_ops_store():
    store = getattr(current_app, "ops_store", None)
    if store is None:
        logger.error("auth.ops_store_unbound")
        raise AuthStoreUnboundError("ops_store not attached to current_app")
    return store


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

    Auth flow (F-AUTH-REMOVE: legacy AuthService removed):
    1. Try UnifiedAuthService (RS256 JWT → X-API-Key OpsStore lookup).
    2. Role intersection check — 403 if caller holds none of the required roles.
    3. Sets flask.g.auth = AuthContext for downstream handlers.
    4. Logs auth_success / authz_denied to OpsStore.audits.
    """

    def decorator(func: Callable) -> Callable:
        func._required_roles = set(roles)

        @wraps(func)
        def wrapper(*args, **kwargs):
            ops_store = _get_ops_store()
            ctx = _try_unified_auth(ops_store)

            if ctx is None:
                return (
                    jsonify({"error": "unauthorized", "reason": "access_denied"}),
                    401,
                )

            if roles and not (_effective_roles(ctx.roles) & frozenset(roles)):
                _log_auth_audit(ops_store, "authz_denied", ctx, request.path)
                return (
                    jsonify({"error": "unauthorized", "reason": "insufficient_role"}),
                    401,
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
