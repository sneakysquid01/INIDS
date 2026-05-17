from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import os
from typing import Callable

from flask import jsonify, request


@dataclass(frozen=True)
class Principal:
    role: str
    token: str


ROLE_RANK = {
    "viewer": 1,
    "analyst": 2,
    "admin": 3,
    "sensor": 0,  # endpoint-gated, not hierarchical — see SENSOR_ALLOWED_ENDPOINTS
}

SENSOR_ALLOWED_ENDPOINTS: frozenset[str] = frozenset({"/api/detect", "/api/stream"})


def _env_bool(*keys: str, default: str = "0") -> bool:
    for key in keys:
        raw = os.getenv(key)
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
    return default.strip().lower() in {"1", "true", "yes", "on"}


class AuthService:
    def __init__(self):
        self.principals: dict[str, Principal] = {}
        self.require_api_keys = _env_bool("INIDS_REQUIRE_API_KEYS")
        self._load_from_env()

    def _load_from_env(self) -> None:
        admin = os.getenv("INIDS_ADMIN_API_KEY", "").strip()
        analyst = os.getenv("INIDS_ANALYST_API_KEY", "").strip()
        sensor = os.getenv("INIDS_SENSOR_API_KEY", "").strip()
        viewer = os.getenv("INIDS_VIEWER_API_KEY", "").strip()

        if admin:
            self.principals[admin] = Principal(role="admin", token=admin)
        if analyst:
            self.principals[analyst] = Principal(role="analyst", token=analyst)
        if sensor:
            self.principals[sensor] = Principal(role="sensor", token=sensor)
        if viewer:
            self.principals[viewer] = Principal(role="viewer", token=viewer)

    @property
    def enabled(self) -> bool:
        return self.require_api_keys or len(self.principals) > 0

    def authorize(self, required_role: str) -> tuple[bool, str]:
        if required_role not in ROLE_RANK:
            return False, "unknown_role"

        if not self.enabled:
            return True, "auth_disabled"

        if self.require_api_keys and not self.principals:
            return False, "auth_not_configured"

        token = request.headers.get("X-API-Key", "").strip()
        if not token:
            return False, "missing_api_key"

        principal = self.principals.get(token)
        if principal is None:
            return False, "invalid_api_key"

        if principal.role == "sensor":
            if request.path in SENSOR_ALLOWED_ENDPOINTS:
                return True, "sensor"
            return False, "insufficient_role"

        if ROLE_RANK[principal.role] < ROLE_RANK[required_role]:
            return False, "insufficient_role"
        return True, principal.role


_auth_service = AuthService()


def require_role(required_role: str) -> Callable:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ok, reason = _auth_service.authorize(required_role)
            if not ok:
                return jsonify({"error": "unauthorized", "reason": "access_denied"}), 401
            return func(*args, **kwargs)

        return wrapper

    return decorator


def auth_status() -> dict[str, str | bool]:
    return {
        "enabled": _auth_service.enabled,
        "configured_roles": sorted({p.role for p in _auth_service.principals.values()}),
    }
