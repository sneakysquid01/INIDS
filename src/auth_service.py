from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
=======
import os
>>>>>>> theirs
=======
import os
>>>>>>> theirs
=======
import os
>>>>>>> theirs
from typing import Callable

from flask import jsonify, request


@dataclass(frozen=True)
class Principal:
    role: str
    token: str


ROLE_RANK = {
    "viewer": 1,
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    "sensor": 2,
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    "analyst": 2,
    "admin": 3,
}


class AuthService:
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    def __init__(self, principals: dict[str, Principal]):
        self.principals: dict[str, Principal] = {}
        for token, principal in principals.items():
            if principal.role not in ROLE_RANK:
                raise RuntimeError(f"Invalid configured role: {principal.role}")
            self.principals[token] = principal
        if not self.principals:
            raise RuntimeError("AuthService requires at least one configured principal")
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
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
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs

    @property
    def enabled(self) -> bool:
        return len(self.principals) > 0

    def authorize(self, required_role: str) -> tuple[bool, str]:
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        normalized_role = "sensor" if required_role == "analyst" else required_role
        if normalized_role not in ROLE_RANK:
            return False, "unknown_role"
        if not self.enabled:
            return False, "auth_unconfigured"
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
        if required_role not in ROLE_RANK:
            return False, "unknown_role"
        if not self.enabled:
            return True, "auth_disabled"
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs

        token = request.headers.get("X-API-Key", "").strip()
        if not token:
            return False, "missing_api_key"

        principal = self.principals.get(token)
        if principal is None:
            return False, "invalid_api_key"

<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        if ROLE_RANK[principal.role] < ROLE_RANK[normalized_role]:
=======
        if ROLE_RANK[principal.role] < ROLE_RANK[required_role]:
>>>>>>> theirs
=======
        if ROLE_RANK[principal.role] < ROLE_RANK[required_role]:
>>>>>>> theirs
=======
        if ROLE_RANK[principal.role] < ROLE_RANK[required_role]:
>>>>>>> theirs
            return False, "insufficient_role"
        return True, principal.role


<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
_auth_service: AuthService | None = None


def configure_auth(
    *,
    admin_api_key: str,
    sensor_api_key: str,
    viewer_api_key: str,
) -> AuthService:
    global _auth_service
    principals = {
        admin_api_key: Principal(role="admin", token=admin_api_key),
        sensor_api_key: Principal(role="sensor", token=sensor_api_key),
        viewer_api_key: Principal(role="viewer", token=viewer_api_key),
    }
    _auth_service = AuthService(principals=principals)
    return _auth_service


def _get_auth_service() -> AuthService:
    if _auth_service is None:
        raise RuntimeError("Auth service is not configured")
    return _auth_service


def authorize_request(required_role: str) -> tuple[bool, str]:
    try:
        svc = _get_auth_service()
    except RuntimeError:
        return False, "auth_unconfigured"
    return svc.authorize(required_role)
=======
_auth_service = AuthService()
>>>>>>> theirs
=======
_auth_service = AuthService()
>>>>>>> theirs
=======
_auth_service = AuthService()
>>>>>>> theirs


def require_role(required_role: str) -> Callable:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
            ok, reason = authorize_request(required_role)
            if not ok:
                status = 500 if reason == "auth_unconfigured" else 401
                return jsonify({"error": "unauthorized", "reason": reason}), status
=======
            ok, reason = _auth_service.authorize(required_role)
            if not ok:
                return jsonify({"error": "unauthorized", "reason": reason}), 401
>>>>>>> theirs
=======
            ok, reason = _auth_service.authorize(required_role)
            if not ok:
                return jsonify({"error": "unauthorized", "reason": reason}), 401
>>>>>>> theirs
=======
            ok, reason = _auth_service.authorize(required_role)
            if not ok:
                return jsonify({"error": "unauthorized", "reason": reason}), 401
>>>>>>> theirs
            return func(*args, **kwargs)

        return wrapper

    return decorator


def auth_status() -> dict[str, str | bool]:
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    if _auth_service is None:
        return {
            "enabled": False,
            "configured_roles": [],
            "required_roles": ["admin", "sensor", "viewer"],
        }
    return {
        "enabled": _auth_service.enabled,
        "configured_roles": sorted({p.role for p in _auth_service.principals.values()}),
        "required_roles": ["admin", "sensor", "viewer"],
=======
    return {
        "enabled": _auth_service.enabled,
        "configured_roles": sorted({p.role for p in _auth_service.principals.values()}),
>>>>>>> theirs
=======
    return {
        "enabled": _auth_service.enabled,
        "configured_roles": sorted({p.role for p in _auth_service.principals.values()}),
>>>>>>> theirs
=======
    return {
        "enabled": _auth_service.enabled,
        "configured_roles": sorted({p.role for p in _auth_service.principals.values()}),
>>>>>>> theirs
    }
