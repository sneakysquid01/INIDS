"""C-04 security regression tests — RS256 + Token Revocation (Step 18).

Validation checkpoints per PLAN.md C-04 spec:
1. decode_ignoring_expiry() accepts a valid RS256 token → returns payload
2. decode_ignoring_expiry() accepts an expired RS256 token (sig valid) → returns payload
3. decode_ignoring_expiry() rejects forged / tampered tokens → returns error
4. decode_ignoring_expiry() rejects HS256 tokens (wrong algorithm) → returns error
5. After add_revoked_token(), is_token_revoked(jti) → True
6. cleanup_revoked_tokens() deletes rows whose expires_at < cutoff
7. cleanup_revoked_tokens() does NOT delete rows whose expires_at >= cutoff
8. cleanup_revoked_tokens() returns correct deleted row count
9. Revoke endpoint (POST /api/auth/revoke): valid RS256 token → 200 + revoked jti
10. Revoke endpoint: missing Authorization header → 400
11. Revoke endpoint: tampered token → 401
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import jwt as _jwt
import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa

from src.auth.jwt_manager import RS256JWTManager, get_jwt_manager, reset_jwt_manager
from src.ops_store import OpsStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ephemeral_manager():
    """RS256JWTManager with an ephemeral RSA keypair (no env vars needed)."""
    reset_jwt_manager()
    mgr = RS256JWTManager()
    yield mgr
    reset_jwt_manager()


@pytest.fixture
def fresh_store(tmp_path: Path) -> OpsStore:
    return OpsStore(str(tmp_path / "test.db"))


def _make_expired_token(mgr: RS256JWTManager) -> str:
    """Issue a token with exp already in the past."""
    import uuid
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "test_user",
        "user_id": "u-test",
        "roles": ["analyst"],
        "jti": uuid.uuid4().hex,
        "iat": int((now - timedelta(hours=2)).timestamp()),
        "exp": int((now - timedelta(hours=1)).timestamp()),  # expired 1 hour ago
        "aud": "INIDS-API",
    }
    import jwt as _jwt_mod
    return _jwt_mod.encode(payload, mgr._private_key, algorithm="RS256")


# ---------------------------------------------------------------------------
# Checkpoint 1–4: decode_ignoring_expiry()
# ---------------------------------------------------------------------------

class TestDecodeIgnoringExpiry:
    def test_valid_rs256_token_returns_payload(self, ephemeral_manager):
        """Checkpoint 1: valid, non-expired RS256 token → ok=True, payload returned."""
        token = ephemeral_manager.create_token("u-1", "alice", ["admin"])
        ok, payload, error = ephemeral_manager.decode_ignoring_expiry(token)
        assert ok is True
        assert payload is not None
        assert payload["sub"] == "alice"
        assert error is None

    def test_expired_rs256_token_returns_payload(self, ephemeral_manager):
        """Checkpoint 2: RS256 token expired 1 hour ago → ok=True (sig still valid)."""
        token = _make_expired_token(ephemeral_manager)
        ok, payload, error = ephemeral_manager.decode_ignoring_expiry(token)
        assert ok is True
        assert payload is not None
        assert payload["sub"] == "test_user"

    def test_tampered_token_returns_error(self, ephemeral_manager):
        """Checkpoint 3: tampered payload → ok=False."""
        token = ephemeral_manager.create_token("u-2", "bob", ["analyst"])
        # Corrupt the payload section (middle part)
        parts = token.split(".")
        corrupted = parts[0] + ".AAAAAAAAAAAAAAAAAAAAAAAA." + parts[2]
        ok, payload, error = ephemeral_manager.decode_ignoring_expiry(corrupted)
        assert ok is False
        assert payload is None
        assert error is not None

    def test_hs256_token_rejected(self, ephemeral_manager):
        """Checkpoint 4: HS256 token is rejected (wrong algorithm)."""
        import uuid
        hs256_payload = {
            "sub": "attacker",
            "user_id": "attacker",
            "roles": ["admin"],
            "jti": uuid.uuid4().hex,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "aud": "INIDS-API",
        }
        hs256_token = _jwt.encode(hs256_payload, "secret-key", algorithm="HS256")
        ok, payload, error = ephemeral_manager.decode_ignoring_expiry(hs256_token)
        assert ok is False
        assert payload is None

    def test_completely_invalid_token_returns_error(self, ephemeral_manager):
        """Random string → ok=False, no exception."""
        ok, payload, error = ephemeral_manager.decode_ignoring_expiry("not.a.jwt")
        assert ok is False
        assert payload is None
        assert error is not None

    def test_jti_present_in_decoded_payload(self, ephemeral_manager):
        """jti claim must be in the decoded payload."""
        token = ephemeral_manager.create_token("u-3", "carol", ["viewer"])
        ok, payload, _ = ephemeral_manager.decode_ignoring_expiry(token)
        assert ok is True
        assert "jti" in (payload or {})
        assert len(payload["jti"]) > 0

    def test_verify_token_still_rejects_expired_normally(self, ephemeral_manager):
        """verify_token() must still reject expired tokens (not affected by C-04 change)."""
        token = _make_expired_token(ephemeral_manager)
        ok, payload, error = ephemeral_manager.verify_token(token)
        assert ok is False
        assert error == "token_expired"


# ---------------------------------------------------------------------------
# Checkpoint 5: Revocation round-trip (add → is_revoked)
# ---------------------------------------------------------------------------

class TestRevocationRoundTrip:
    def test_is_token_revoked_true_after_add(self, ephemeral_manager, fresh_store):
        """Checkpoint 5: add_revoked_token → is_token_revoked returns True."""
        token = ephemeral_manager.create_token("u-rt", "rt_user", ["analyst"])
        ok, payload, _ = ephemeral_manager.verify_token(token)
        assert ok and payload is not None
        jti = payload["jti"]

        fresh_store.add_revoked_token(
            jti=jti,
            user_id="u-rt",
            expires_at=datetime.now(timezone.utc).isoformat(),
            revoked_at=datetime.now(timezone.utc).isoformat(),
        )
        assert fresh_store.is_token_revoked(jti) is True

    def test_is_token_revoked_false_for_unrevoced_jti(self, fresh_store):
        """Not-yet-revoked jti → is_token_revoked returns False."""
        assert fresh_store.is_token_revoked("not-in-table-jti") is False

    def test_add_revoked_token_idempotent(self, ephemeral_manager, fresh_store):
        """Duplicate add_revoked_token calls must not raise (INSERT OR IGNORE)."""
        token = ephemeral_manager.create_token("u-dup", "dup_user", ["admin"])
        _, payload, _ = ephemeral_manager.verify_token(token)
        jti = payload["jti"]
        now = datetime.now(timezone.utc).isoformat()

        fresh_store.add_revoked_token(jti=jti, user_id="u-dup", expires_at=now, revoked_at=now)
        try:
            fresh_store.add_revoked_token(jti=jti, user_id="u-dup", expires_at=now, revoked_at=now)
        except Exception as exc:
            pytest.fail(f"Duplicate add_revoked_token raised: {exc}")


# ---------------------------------------------------------------------------
# Checkpoints 6–8: cleanup_revoked_tokens()
# ---------------------------------------------------------------------------

class TestCleanupRevokedTokens:
    def _add_token(self, store: OpsStore, jti: str, expires_at: str) -> None:
        store.add_revoked_token(
            jti=jti,
            user_id="cleanup-test",
            expires_at=expires_at,
            revoked_at=datetime.now(timezone.utc).isoformat(),
        )

    def test_deletes_rows_before_cutoff(self, fresh_store):
        """Checkpoint 6: rows with expires_at < cutoff are deleted."""
        past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        self._add_token(fresh_store, "jti-old", past)

        deleted = fresh_store.cleanup_revoked_tokens()
        assert deleted >= 1
        assert fresh_store.is_token_revoked("jti-old") is False

    def test_does_not_delete_rows_after_cutoff(self, fresh_store):
        """Checkpoint 7: rows with expires_at >= cutoff are preserved."""
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        self._add_token(fresh_store, "jti-future", future)

        deleted = fresh_store.cleanup_revoked_tokens()
        assert deleted == 0
        assert fresh_store.is_token_revoked("jti-future") is True

    def test_returns_correct_count(self, fresh_store):
        """Checkpoint 8: return value equals number of deleted rows."""
        past = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        self._add_token(fresh_store, "jti-count-1", past)
        self._add_token(fresh_store, "jti-count-2", past)
        self._add_token(fresh_store, "jti-count-3", past)

        deleted = fresh_store.cleanup_revoked_tokens()
        assert deleted == 3

    def test_cleanup_with_explicit_cutoff(self, fresh_store):
        """Custom before_iso cutoff is respected."""
        cutoff = datetime.now(timezone.utc).isoformat()
        past = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        self._add_token(fresh_store, "jti-before", past)

        deleted = fresh_store.cleanup_revoked_tokens(before_iso=cutoff)
        assert deleted >= 1

    def test_cleanup_empty_table_returns_zero(self, fresh_store):
        """No rows → cleanup returns 0."""
        deleted = fresh_store.cleanup_revoked_tokens()
        assert deleted == 0

    def test_cleanup_mixed_preserves_future_rows(self, fresh_store):
        """Mixed old and future rows: only old ones deleted."""
        past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        self._add_token(fresh_store, "jti-old-mix", past)
        self._add_token(fresh_store, "jti-new-mix", future)

        deleted = fresh_store.cleanup_revoked_tokens()
        assert deleted == 1
        assert fresh_store.is_token_revoked("jti-old-mix") is False
        assert fresh_store.is_token_revoked("jti-new-mix") is True


# ---------------------------------------------------------------------------
# Checkpoints 9–11: Revoke endpoint (Flask integration)
# ---------------------------------------------------------------------------

class TestRevokeEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Patch the app's ops_store and share a single JWT manager per test.

        reset_jwt_manager() + get_jwt_manager() is called ONCE in the fixture
        to create the ephemeral keypair that both the test and the endpoint
        handler will use via the module-level singleton.  A second reset after
        the request would create a different keypair and break signature checks.
        """
        import web_app.app as _app_module
        self._store = OpsStore(str(tmp_path / "ep_test.db"))

        # Establish a known singleton; endpoint calls get_jwt_manager() too.
        reset_jwt_manager()
        self._mgr = get_jwt_manager()

        with patch.object(_app_module, "ops_store", self._store):
            self._client = _app_module.app.test_client()
            yield

        reset_jwt_manager()  # Clean up after test

    def test_valid_rs256_token_returns_200_and_revoked(self):
        """Checkpoint 9: valid RS256 token → 200, {"revoked": true, "jti": ...}."""
        token = self._mgr.create_token("u-ep", "ep_user", ["analyst"])
        resp = self._client.post(
            "/api/auth/revoke",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["revoked"] is True
        assert "jti" in data
        assert len(data["jti"]) > 0

    def test_missing_authorization_returns_400(self):
        """Checkpoint 10: no Authorization header → 400."""
        resp = self._client.post("/api/auth/revoke")
        assert resp.status_code == 400

    def test_tampered_token_returns_401(self):
        """Checkpoint 11: tampered token → 401."""
        token = self._mgr.create_token("u-ep2", "ep_user2", ["analyst"])
        parts = token.split(".")
        corrupted = parts[0] + ".AAAAAAAAAAAAAAAAAAAAAA." + parts[2]
        resp = self._client.post(
            "/api/auth/revoke",
            headers={"Authorization": f"Bearer {corrupted}"},
        )
        assert resp.status_code == 401

    def test_hs256_token_returns_401(self):
        """HS256 token must be rejected by revoke endpoint (RS256-only)."""
        import uuid
        hs256_payload = {
            "sub": "attacker",
            "user_id": "attacker",
            "roles": ["admin"],
            "jti": uuid.uuid4().hex,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "aud": "INIDS-API",
        }
        hs256_token = _jwt.encode(hs256_payload, "secret-key", algorithm="HS256")
        resp = self._client.post(
            "/api/auth/revoke",
            headers={"Authorization": f"Bearer {hs256_token}"},
        )
        assert resp.status_code == 401

    def test_revoked_jti_recorded_in_ops_store(self):
        """After successful revoke, jti is in revoked_tokens table."""
        token = self._mgr.create_token("u-ep3", "ep_user3", ["analyst"])
        _, payload, _ = self._mgr.verify_token(token)
        jti = payload["jti"]

        self._client.post(
            "/api/auth/revoke",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert self._store.is_token_revoked(jti) is True
