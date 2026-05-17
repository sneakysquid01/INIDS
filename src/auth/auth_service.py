from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.auth.models import AuthContext
from src.auth.jwt_manager import get_jwt_manager

if TYPE_CHECKING:
    from src.ops_store import OpsStore

logger = logging.getLogger(__name__)


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class UnifiedAuthService:
    """Primary authentication service backed by OpsStore (users/api_keys/revoked_tokens).

    authenticate_api_key  — SHA-256 hash lookup against api_keys table
    authenticate_jwt      — RS256 verify + jti revocation check
    create_token          — issue a 1-hour RS256 JWT
    revoke_token          — write jti to revoked_tokens table
    """

    def __init__(self, ops_store: OpsStore) -> None:
        self._store = ops_store
        self._jwt = get_jwt_manager()

    def authenticate_api_key(self, key: str) -> AuthContext | None:
        key_hash = _sha256_hex(key)
        row = self._store.get_user_by_key_hash(key_hash)
        if row is None:
            return None
        roles: frozenset[str] = frozenset(
            r.strip() for r in row["roles"].split(",") if r.strip()
        )
        return AuthContext(
            user_id=row["user_id"],
            username=row["username"],
            roles=roles,
        )

    def authenticate_jwt(self, token: str) -> AuthContext | None:
        ok, payload, reason = self._jwt.verify_token(token)
        if not ok or payload is None:
            return None
        jti = payload.get("jti", "")
        if jti and self._store.is_token_revoked(jti):
            logger.warning("Rejected revoked token jti=%s", jti)
            return None
        roles: frozenset[str] = frozenset(payload.get("roles", []))
        return AuthContext(
            user_id=payload.get("user_id", ""),
            username=payload.get("sub", ""),
            roles=roles,
        )

    def create_token(self, user_id: str, username: str, roles: list[str]) -> str:
        return self._jwt.create_token(user_id, username, roles)

    def revoke_token(self, jti: str, user_id: str, expires_at: str) -> None:
        self._store.add_revoked_token(
            jti=jti,
            user_id=user_id,
            expires_at=expires_at,
            revoked_at=datetime.now(timezone.utc).isoformat(),
        )
