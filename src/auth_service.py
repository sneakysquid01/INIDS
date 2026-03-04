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
        self._load_from_env()

    def _load_from_env(self) -> None:
        admin = os.getenv("INIDS_ADMIN_API_KEY", "").strip()
        analyst = os.getenv("INIDS_ANALYST_API_KEY", "").strip()
        viewer = os.getenv("INIDS_VIEWER_API_KEY", "").strip()

        if admin:
            self.principals[admin] = Principal(role="admin", token=admin)
        if analyst:
            self.principals[analyst] = Principal(role="analyst", token=analyst)
        if viewer:
            self.principals[viewer] = Principal(role="viewer", token=viewer)

    @property
    def enabled(self) -> bool:
        return len(self.principals) > 0

    def authorize(self, required_role: str) -> tuple[bool, str]:
        if required_role not in ROLE_RANK:
            return False, "unknown_role"
        if not self.enabled:
            return True, "auth_disabled"

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
            ok, reason = _auth_service.authorize(required_role)
            if not ok:
                return jsonify({"error": "unauthorized", "reason": reason}), 401
            return func(*args, **kwargs)

        return wrapper

    return decorator


def auth_status() -> dict[str, str | bool]:
    return {
        "enabled": _auth_service.enabled,
        "configured_roles": sorted({p.role for p in _auth_service.principals.values()}),
    }
