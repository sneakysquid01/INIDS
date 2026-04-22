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
}


class AuthService:
    def __init__(self):
        self.principals: dict[str, Principal] = {}
        # Default: ON (require API keys unless explicitly disabled)
        self.require_api_keys = os.getenv("INIDS_REQUIRE_API_KEYS", "1") == "1"
        self.allow_unauthenticated = os.getenv("INIDS_ALLOW_UNAUTHENTICATED", "0") == "1"
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
            self.principals[sensor] = Principal(role="analyst", token=sensor)
        if viewer:
            self.principals[viewer] = Principal(role="viewer", token=viewer)

    @property
    def enabled(self) -> bool:
        if self.allow_unauthenticated:
            return False
        return self.require_api_keys or len(self.principals) > 0

    def authorize(self, required_role: str) -> tuple[bool, str]:
        import sys
        print(f"DEBUG authorize: required_role={required_role}, allow_unauthenticated={self.allow_unauthenticated}, enabled={self.enabled}", file=sys.stderr)
        
        if required_role not in ROLE_RANK:
            return False, "unknown_role"
        
        if self.allow_unauthenticated:
            print(f"DEBUG authorize: RETURNING TRUE due to allow_unauthenticated", file=sys.stderr)
            return True, "unauthenticated_allowed"
        
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

        if ROLE_RANK[principal.role] < ROLE_RANK[required_role]:
            return False, "insufficient_role"
        return True, principal.role


_auth_service = AuthService()


def require_role(required_role: str) -> Callable:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # TEMPORARY DEBUG: Force bypass for all requests
            print(f"FORCE BYPASS: allow_unauthenticated={_auth_service.allow_unauthenticated}")
            if _auth_service.allow_unauthenticated:
                return func(*args, **kwargs)
            
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
