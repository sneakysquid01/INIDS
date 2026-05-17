from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

logger = logging.getLogger(__name__)

_ALGORITHM = "RS256"
_AUDIENCE = "INIDS-API"
_TOKEN_EXPIRY_SECONDS = 3600  # 1 hour — non-negotiable per PLAN.md


def _pem_bytes(value: str) -> bytes:
    """Normalize env-var PEM: expand literal \\n back to newlines."""
    return value.replace("\\n", "\n").encode()


class RS256JWTManager:
    """RS256 JWT sign/verify. Uses INIDS_JWT_PRIVATE_KEY / INIDS_JWT_PUBLIC_KEY.

    If INIDS_JWT_PRIVATE_KEY is absent, an ephemeral keypair is generated and
    a WARNING is logged. Ephemeral keys are invalidated on restart — only
    acceptable during the INIDS_AUTH_COMPAT transition window.
    """

    def __init__(self) -> None:
        self._private_key, self._public_key = self._load_keys()

    def _load_keys(self) -> tuple[Any, Any]:
        priv_pem = os.environ.get("INIDS_JWT_PRIVATE_KEY", "").strip()
        pub_pem = os.environ.get("INIDS_JWT_PUBLIC_KEY", "").strip()

        if not priv_pem:
            logger.warning(
                "INIDS_JWT_PRIVATE_KEY not set — generating ephemeral RSA-2048 keypair. "
                "Tokens issued now will be invalid after restart. "
                "Set INIDS_JWT_PRIVATE_KEY and INIDS_JWT_PUBLIC_KEY for production."
            )
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend(),
            )
            public_key = private_key.public_key()
            return private_key, public_key

        private_key = serialization.load_pem_private_key(
            _pem_bytes(priv_pem), password=None, backend=default_backend()
        )
        if pub_pem:
            public_key = serialization.load_pem_public_key(
                _pem_bytes(pub_pem), backend=default_backend()
            )
        else:
            # Derive public key from the private key (acceptable during compat window)
            logger.warning(
                "INIDS_JWT_PUBLIC_KEY not set — deriving public key from private key. "
                "Set INIDS_JWT_PUBLIC_KEY before setting INIDS_AUTH_COMPAT=false."
            )
            public_key = private_key.public_key()

        return private_key, public_key

    def create_token(self, user_id: str, username: str, roles: list[str]) -> str:
        now = datetime.now(timezone.utc)
        jti = uuid.uuid4().hex
        payload = {
            "sub": username,
            "user_id": user_id,
            "roles": sorted(roles),
            "jti": jti,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=_TOKEN_EXPIRY_SECONDS)).timestamp()),
            "aud": _AUDIENCE,
        }
        return jwt.encode(payload, self._private_key, algorithm=_ALGORITHM)

    def verify_token(
        self, token: str
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        """Verify RS256 token. Returns (ok, payload_dict, error_reason)."""
        try:
            payload = jwt.decode(
                token,
                self._public_key,
                algorithms=[_ALGORITHM],
                audience=_AUDIENCE,
            )
            return True, payload, None
        except jwt.ExpiredSignatureError:
            return False, None, "token_expired"
        except jwt.InvalidTokenError as exc:
            return False, None, f"invalid_token:{exc}"
        except Exception as exc:
            logger.error("JWT verification error: %s", exc)
            return False, None, "verification_error"

    def decode_ignoring_expiry(
        self, token: str
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        """Verify RS256 signature and decode payload, ignoring token expiry.

        C-04 (Step 18): used by the revoke endpoint to accept recently-expired
        tokens so clients can revoke tokens from a compromised session even if
        the 1-hour window has passed. The RS256 signature is always enforced —
        forged or HS256 tokens are rejected.

        Returns (ok, payload_dict, error_reason).
        """
        try:
            payload = jwt.decode(
                token,
                self._public_key,
                algorithms=[_ALGORITHM],
                audience=_AUDIENCE,
                options={"verify_exp": False},
            )
            return True, payload, None
        except jwt.InvalidTokenError as exc:
            return False, None, f"invalid_token:{exc}"
        except Exception as exc:
            logger.error("JWT decode_ignoring_expiry error: %s", exc)
            return False, None, "verification_error"


# Module-level singleton — created once per process
_manager: RS256JWTManager | None = None


def get_jwt_manager() -> RS256JWTManager:
    global _manager
    if _manager is None:
        _manager = RS256JWTManager()
    return _manager


def reset_jwt_manager() -> None:
    """Force re-initialization (used in tests)."""
    global _manager
    _manager = None
