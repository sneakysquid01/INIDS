"""C-01 security regression tests — Unified Authentication System (Step 16).

Validation checkpoints per PLAN.md C-01 spec:
1. POST /api/auth/login with valid credential → RS256 JWT issued
2. RS256 JWT accepted on @require_roles("admin") endpoint → 200
3. Forged HS256 JWT → 401 (RS256-only enforcement)
4. Revoked jti → 401
5. INIDS_AUTH_COMPAT=true: old API key accepted as fallback
6. All existing integration tests pass under compat mode (covered by full suite)

Additional coverage:
- OpsStore schema v3: users/api_keys/revoked_tokens tables + indexes
- UnifiedAuthService.authenticate_api_key: valid/invalid key
- UnifiedAuthService.authenticate_jwt: RS256 valid, expired, wrong algorithm
- validate_config_at_startup: missing secrets, placeholder values, compat vs non-compat
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_rsa_keypair():
    """Generate a throwaway RSA-2048 keypair for tests."""
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv_pem, pub_pem, private_key


def _make_store(tmp_path: Path, env_keys: dict | None = None) -> "OpsStore":
    from src.auth.jwt_manager import reset_jwt_manager
    reset_jwt_manager()

    db_path = str(tmp_path / "test.db")
    env = env_keys or {}
    with patch.dict(os.environ, env, clear=False):
        from src.ops_store import OpsStore
        store = OpsStore(db_path)
    return store


def _seed_user(store, user_id: str, username: str, role: str, api_key: str):
    """Directly insert a user + api_key into an OpsStore (bypasses env-var seeding)."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    now = store._utc_now_iso()
    store._execute(
        "INSERT OR IGNORE INTO users (user_id, username, roles, created_at) "
        "VALUES (:uid, :uname, :roles, :now)",
        {"uid": user_id, "uname": username, "roles": role, "now": now},
    )
    store._execute(
        "INSERT OR IGNORE INTO api_keys (key_id, user_id, key_hash, label, created_at) "
        "VALUES (:kid, :uid, :khash, :label, :now)",
        {
            "kid": f"key-{user_id}",
            "uid": user_id,
            "khash": key_hash,
            "label": "test",
            "now": now,
        },
    )


# ---------------------------------------------------------------------------
# Schema v3 tests
# ---------------------------------------------------------------------------

class TestSchemaV3:
    def test_fresh_db_has_users_table(self, tmp_path):
        store = _make_store(tmp_path)
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "users" in tables
        assert "api_keys" in tables
        assert "revoked_tokens" in tables

    def test_idx_api_keys_hash_exists(self, tmp_path):
        _make_store(tmp_path)
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()
        assert "idx_api_keys_hash" in indexes

    def test_idx_revoked_jti_exists(self, tmp_path):
        # G-AUTH-1: prevent full table scan on revocation check
        _make_store(tmp_path)
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()
        assert "idx_revoked_jti" in indexes

    def test_schema_version_at_least_3(self, tmp_path):
        """C-01 added migration v3; subsequent C phases may bump further."""
        store = _make_store(tmp_path)
        assert store.SCHEMA_VERSION >= 3

    def test_env_key_seeded_into_api_keys(self, tmp_path):
        from src.auth.jwt_manager import reset_jwt_manager
        reset_jwt_manager()
        api_key = "test-admin-key-for-seeding"
        # OpsStore must be created inside the env patch so migration v3 seeds
        # from our test key, not from any real key previously loaded into os.environ
        # by other tests importing the Flask app (which calls _load_dotenv()).
        with patch.dict(
            os.environ,
            {"INIDS_ADMIN_API_KEY": api_key},
            clear=False,
        ):
            from src.ops_store import OpsStore
            store = OpsStore(str(tmp_path / "seed.db"))
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        row = store.get_user_by_key_hash(key_hash)
        assert row is not None
        assert row["roles"] == "admin"
        assert row["username"] == "svc_admin"

    def test_seed_is_idempotent(self, tmp_path):
        api_key = "idempotent-key-unique-xyz"
        # Create OpsStore inside the env patch so migration v3 seeds our test key.
        with patch.dict(os.environ, {"INIDS_ADMIN_API_KEY": api_key}, clear=False):
            from src.ops_store import OpsStore
            store = OpsStore(str(tmp_path / "idem.db"))
            # Call seed a second time — must not raise or duplicate.
            store._seed_service_accounts()
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        rows = store._fetchall(
            "SELECT * FROM api_keys WHERE key_hash = :h", {"h": key_hash}
        )
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# RS256JWTManager tests
# ---------------------------------------------------------------------------

class TestRS256JWTManager:
    def _make_manager(self, tmp_path) -> "RS256JWTManager":
        from src.auth.jwt_manager import reset_jwt_manager, RS256JWTManager
        reset_jwt_manager()
        priv_pem, pub_pem, _ = _make_rsa_keypair()
        with patch.dict(
            os.environ,
            {"INIDS_JWT_PRIVATE_KEY": priv_pem, "INIDS_JWT_PUBLIC_KEY": pub_pem},
            clear=False,
        ):
            mgr = RS256JWTManager()
        return mgr

    def test_create_token_returns_string(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        token = mgr.create_token("uid-1", "alice", ["admin"])
        assert isinstance(token, str)
        assert len(token) > 50

    def test_verify_valid_rs256_token(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        token = mgr.create_token("uid-1", "alice", ["admin"])
        ok, payload, err = mgr.verify_token(token)
        assert ok is True
        assert payload is not None
        assert payload["sub"] == "alice"
        assert payload["user_id"] == "uid-1"
        assert "admin" in payload["roles"]
        assert "jti" in payload

    def test_token_contains_1hr_expiry(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        token = mgr.create_token("uid-1", "alice", ["admin"])
        _, payload, _ = mgr.verify_token(token)
        exp_delta = payload["exp"] - payload["iat"]
        # Allow ±5s for clock drift during test execution
        assert abs(exp_delta - 3600) < 5

    def test_hs256_token_rejected_after_rs256_migration(self, tmp_path):
        """Checkpoint 3: forged HS256 JWT → 401 (RS256-only service rejects it)."""
        mgr = self._make_manager(tmp_path)
        hs256_token = jwt.encode(
            {
                "sub": "attacker",
                "user_id": "evil",
                "roles": ["admin"],
                "jti": uuid.uuid4().hex,
                "iat": int(time.time()),
                "exp": int(time.time()) + 3600,
                "aud": "INIDS-API",
            },
            "any-secret",
            algorithm="HS256",
        )
        ok, payload, err = mgr.verify_token(hs256_token)
        assert ok is False
        assert payload is None

    def test_expired_token_rejected(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        priv_pem, _, priv_key_obj = _make_rsa_keypair()
        expired_token = jwt.encode(
            {
                "sub": "alice",
                "user_id": "uid-1",
                "roles": ["admin"],
                "jti": uuid.uuid4().hex,
                "iat": int(time.time()) - 7200,
                "exp": int(time.time()) - 3600,
                "aud": "INIDS-API",
            },
            mgr._private_key,
            algorithm="RS256",
        )
        ok, payload, err = mgr.verify_token(expired_token)
        assert ok is False
        assert "token_expired" in (err or "")

    def test_tampered_token_rejected(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        token = mgr.create_token("uid-1", "alice", ["viewer"])
        # Flip a character in the signature (last segment)
        parts = token.split(".")
        sig = parts[-1]
        parts[-1] = sig[:-4] + ("XXXX" if not sig.endswith("XXXX") else "YYYY")
        tampered = ".".join(parts)
        ok, _, err = mgr.verify_token(tampered)
        assert ok is False

    def test_ephemeral_key_generated_when_env_absent(self, tmp_path):
        from src.auth.jwt_manager import reset_jwt_manager, RS256JWTManager
        reset_jwt_manager()
        with patch.dict(
            os.environ, {}, clear=False
        ), patch.dict(os.environ, {"INIDS_JWT_PRIVATE_KEY": "", "INIDS_JWT_PUBLIC_KEY": ""}):
            mgr = RS256JWTManager()
        token = mgr.create_token("uid-1", "alice", ["admin"])
        ok, payload, _ = mgr.verify_token(token)
        assert ok is True
        assert payload["sub"] == "alice"


# ---------------------------------------------------------------------------
# UnifiedAuthService tests
# ---------------------------------------------------------------------------

class TestUnifiedAuthService:
    def _make_svc(self, store):
        from src.auth.jwt_manager import reset_jwt_manager, RS256JWTManager
        reset_jwt_manager()
        priv_pem, pub_pem, _ = _make_rsa_keypair()
        with patch.dict(
            os.environ,
            {"INIDS_JWT_PRIVATE_KEY": priv_pem, "INIDS_JWT_PUBLIC_KEY": pub_pem},
            clear=False,
        ):
            mgr = RS256JWTManager()

        from src.auth import UnifiedAuthService
        from src.auth import jwt_manager as jm_module
        jm_module._manager = mgr
        return UnifiedAuthService(store)

    def test_authenticate_valid_api_key(self, tmp_path):
        store = _make_store(tmp_path)
        _seed_user(store, "uid-admin", "alice", "admin", "my-secret-key")
        svc = self._make_svc(store)

        ctx = svc.authenticate_api_key("my-secret-key")

        assert ctx is not None
        assert ctx.username == "alice"
        assert "admin" in ctx.roles

    def test_authenticate_invalid_api_key_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        svc = self._make_svc(store)
        assert svc.authenticate_api_key("wrong-key") is None

    def test_authenticate_valid_rs256_jwt(self, tmp_path):
        """Checkpoint 1+2: RS256 JWT issued then accepted."""
        store = _make_store(tmp_path)
        _seed_user(store, "uid-admin", "alice", "admin", "my-key")
        svc = self._make_svc(store)

        token = svc.create_token("uid-admin", "alice", ["admin"])
        ctx = svc.authenticate_jwt(token)

        assert ctx is not None
        assert ctx.username == "alice"
        assert "admin" in ctx.roles

    def test_authenticate_revoked_token_returns_none(self, tmp_path):
        """Checkpoint 4: revoked jti → rejected."""
        store = _make_store(tmp_path)
        svc = self._make_svc(store)

        token = svc.create_token("uid-admin", "alice", ["admin"])
        from src.auth.jwt_manager import _ALGORITHM, _AUDIENCE
        payload = jwt.decode(
            token,
            svc._jwt._public_key,
            algorithms=[_ALGORITHM],
            audience=_AUDIENCE,
        )
        jti = payload["jti"]
        exp_iso = __import__("datetime").datetime.utcfromtimestamp(payload["exp"]).isoformat()

        svc.revoke_token(jti, "uid-admin", exp_iso)
        ctx = svc.authenticate_jwt(token)

        assert ctx is None

    def test_revoke_is_idempotent(self, tmp_path):
        store = _make_store(tmp_path)
        svc = self._make_svc(store)
        token = svc.create_token("uid-1", "alice", ["admin"])
        from src.auth.jwt_manager import _ALGORITHM, _AUDIENCE
        payload = jwt.decode(
            token, svc._jwt._public_key, algorithms=[_ALGORITHM], audience=_AUDIENCE
        )
        jti = payload["jti"]
        exp_iso = __import__("datetime").datetime.utcfromtimestamp(payload["exp"]).isoformat()
        svc.revoke_token(jti, "uid-1", exp_iso)
        svc.revoke_token(jti, "uid-1", exp_iso)  # second call must not raise
        assert store.is_token_revoked(jti)

    def test_hs256_token_rejected_by_unified_service(self, tmp_path):
        """Checkpoint 3: HS256 token always returns None from unified auth."""
        store = _make_store(tmp_path)
        svc = self._make_svc(store)
        hs256_token = jwt.encode(
            {
                "sub": "attacker",
                "user_id": "evil",
                "roles": ["admin"],
                "jti": uuid.uuid4().hex,
                "iat": int(time.time()),
                "exp": int(time.time()) + 3600,
                "aud": "INIDS-API",
            },
            "any-secret",
            algorithm="HS256",
        )
        ctx = svc.authenticate_jwt(hs256_token)
        assert ctx is None


# ---------------------------------------------------------------------------
# validate_config_at_startup tests
# ---------------------------------------------------------------------------

class TestValidateConfigAtStartup:
    def _validate(self, env: dict):
        from src.auth.validators import validate_config_at_startup
        with patch.dict(os.environ, env, clear=True):
            validate_config_at_startup()

    def test_valid_secrets_pass(self):
        self._validate({
            "SECRET_KEY": "real-secret-32chars-AAAAAAAAAA",
            "INIDS_ADMIN_API_KEY": "real-admin-key-1234567890",
            "INIDS_AUTH_COMPAT": "true",
        })

    def test_missing_secret_key_raises(self):
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            self._validate({
                "INIDS_ADMIN_API_KEY": "real-admin-key",
                "INIDS_AUTH_COMPAT": "true",
            })

    def test_missing_admin_key_raises(self):
        with pytest.raises(RuntimeError, match="INIDS_ADMIN_API_KEY"):
            self._validate({
                "SECRET_KEY": "real-secret",
                "INIDS_AUTH_COMPAT": "true",
            })

    def test_placeholder_secret_key_raises(self):
        with pytest.raises(RuntimeError, match="(?i)placeholder"):
            self._validate({
                "SECRET_KEY": "changeme",
                "INIDS_ADMIN_API_KEY": "real-key",
                "INIDS_AUTH_COMPAT": "true",
            })

    def test_placeholder_admin_key_raises(self):
        with pytest.raises(RuntimeError, match="(?i)placeholder"):
            self._validate({
                "SECRET_KEY": "real-secret",
                "INIDS_ADMIN_API_KEY": "placeholder",
                "INIDS_AUTH_COMPAT": "true",
            })

    def test_rs256_mode_requires_jwt_public_key(self):
        """G-AUTH-3: INIDS_JWT_PUBLIC_KEY required when INIDS_AUTH_COMPAT=false."""
        with pytest.raises(RuntimeError, match="INIDS_JWT_PUBLIC_KEY"):
            self._validate({
                "SECRET_KEY": "real-secret",
                "INIDS_ADMIN_API_KEY": "real-key",
                "INIDS_AUTH_COMPAT": "false",
            })

    def test_compat_mode_allows_missing_jwt_key(self):
        """During compat window, missing JWT public key is only a warning."""
        # Should not raise
        self._validate({
            "SECRET_KEY": "real-secret",
            "INIDS_ADMIN_API_KEY": "real-key",
            "INIDS_AUTH_COMPAT": "true",
        })

    def test_rs256_mode_passes_with_jwt_key_set(self):
        _, pub_pem, _ = _make_rsa_keypair()
        self._validate({
            "SECRET_KEY": "real-secret",
            "INIDS_ADMIN_API_KEY": "real-key",
            "INIDS_AUTH_COMPAT": "false",
            "INIDS_JWT_PUBLIC_KEY": pub_pem,
        })


# ---------------------------------------------------------------------------
# require_roles decorator tests (Flask integration)
# ---------------------------------------------------------------------------

class TestRequireRolesDecorator:
    """Integration tests for require_roles using a minimal Flask app."""

    def _make_app(self, store, priv_pem: str, pub_pem: str):
        from flask import Flask, jsonify
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.ops_store = store

        from src.auth.jwt_manager import reset_jwt_manager, RS256JWTManager
        reset_jwt_manager()
        from src.auth import jwt_manager as jm_module
        with patch.dict(
            os.environ,
            {"INIDS_JWT_PRIVATE_KEY": priv_pem, "INIDS_JWT_PUBLIC_KEY": pub_pem},
            clear=False,
        ):
            jm_module._manager = RS256JWTManager()

        from src.auth.decorators import require_roles

        @app.route("/admin-only")
        @require_roles("admin")
        def admin_only():
            return jsonify({"ok": True})

        @app.route("/analyst-or-admin")
        @require_roles("analyst", "admin")
        def multi_role():
            return jsonify({"ok": True})

        return app

    def test_rs256_token_grants_access(self, tmp_path):
        """Checkpoint 2: RS256 JWT on @require_roles endpoint → 200."""
        store = _make_store(tmp_path)
        _seed_user(store, "uid-admin", "alice", "admin", "key-alice")
        priv_pem, pub_pem, _ = _make_rsa_keypair()
        app = self._make_app(store, priv_pem, pub_pem)

        from src.auth.auth_service import UnifiedAuthService
        svc = UnifiedAuthService(store)
        token = svc.create_token("uid-admin", "alice", ["admin"])

        with app.test_client() as client:
            resp = client.get("/admin-only", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_missing_token_returns_401(self, tmp_path):
        store = _make_store(tmp_path)
        priv_pem, pub_pem, _ = _make_rsa_keypair()
        with patch.dict(os.environ, {"INIDS_AUTH_COMPAT": "false"}, clear=False):
            app = self._make_app(store, priv_pem, pub_pem)
            with app.test_client() as client:
                resp = client.get("/admin-only")
            assert resp.status_code == 401

    def test_insufficient_role_returns_403(self, tmp_path):
        store = _make_store(tmp_path)
        _seed_user(store, "uid-viewer", "viewer_user", "viewer", "viewer-key")
        priv_pem, pub_pem, _ = _make_rsa_keypair()
        app = self._make_app(store, priv_pem, pub_pem)

        from src.auth.auth_service import UnifiedAuthService
        svc = UnifiedAuthService(store)
        token = svc.create_token("uid-viewer", "viewer_user", ["viewer"])

        with patch.dict(os.environ, {"INIDS_AUTH_COMPAT": "false"}, clear=False):
            with app.test_client() as client:
                resp = client.get(
                    "/admin-only", headers={"Authorization": f"Bearer {token}"}
                )
        assert resp.status_code == 403

    def test_hs256_token_returns_401(self, tmp_path):
        """Checkpoint 3: forged HS256 JWT on require_roles endpoint → 401."""
        store = _make_store(tmp_path)
        priv_pem, pub_pem, _ = _make_rsa_keypair()
        with patch.dict(os.environ, {"INIDS_AUTH_COMPAT": "false"}, clear=False):
            app = self._make_app(store, priv_pem, pub_pem)
            hs256_token = jwt.encode(
                {
                    "sub": "attacker",
                    "user_id": "evil",
                    "roles": ["admin"],
                    "jti": uuid.uuid4().hex,
                    "iat": int(time.time()),
                    "exp": int(time.time()) + 3600,
                    "aud": "INIDS-API",
                },
                "any-secret",
                algorithm="HS256",
            )
            with app.test_client() as client:
                resp = client.get(
                    "/admin-only", headers={"Authorization": f"Bearer {hs256_token}"}
                )
        assert resp.status_code == 401

    def test_revoked_token_returns_401(self, tmp_path):
        """Checkpoint 4: revoked jti → 401."""
        store = _make_store(tmp_path)
        _seed_user(store, "uid-admin", "alice", "admin", "key-alice")
        priv_pem, pub_pem, _ = _make_rsa_keypair()
        app = self._make_app(store, priv_pem, pub_pem)

        from src.auth.auth_service import UnifiedAuthService
        from src.auth.jwt_manager import _ALGORITHM, _AUDIENCE
        from src.auth import jwt_manager as jm_module

        svc = UnifiedAuthService(store)
        token = svc.create_token("uid-admin", "alice", ["admin"])
        payload = jwt.decode(
            token,
            jm_module._manager._public_key,
            algorithms=[_ALGORITHM],
            audience=_AUDIENCE,
        )
        jti = payload["jti"]
        import datetime as dt
        exp_iso = dt.datetime.utcfromtimestamp(payload["exp"]).isoformat()
        svc.revoke_token(jti, "uid-admin", exp_iso)

        with patch.dict(os.environ, {"INIDS_AUTH_COMPAT": "false"}, clear=False):
            with app.test_client() as client:
                resp = client.get(
                    "/admin-only", headers={"Authorization": f"Bearer {token}"}
                )
        assert resp.status_code == 401

    def test_compat_mode_legacy_api_key_accepted(self, tmp_path):
        """Checkpoint 5: INIDS_AUTH_COMPAT=true — old env-var API key accepted."""
        store = _make_store(tmp_path)
        priv_pem, pub_pem, _ = _make_rsa_keypair()
        app = self._make_app(store, priv_pem, pub_pem)

        legacy_key = "legacy-admin-api-key"
        mock_principal = MagicMock()
        mock_principal.role = "admin"

        mock_auth_svc = MagicMock()
        mock_auth_svc.principals = {legacy_key: mock_principal}

        with patch.dict(os.environ, {"INIDS_AUTH_COMPAT": "true"}, clear=False):
            with patch("src.auth.decorators._auth_service", mock_auth_svc, create=True):
                with patch(
                    "src.auth.decorators._try_legacy_auth",
                    return_value=__import__("src.auth.models", fromlist=["AuthContext"]).AuthContext(
                        user_id="legacy:admin",
                        username="admin",
                        roles=frozenset({"admin"}),
                    ),
                ):
                    with app.test_client() as client:
                        resp = client.get(
                            "/admin-only", headers={"X-API-Key": legacy_key}
                        )
        assert resp.status_code == 200

    def test_api_key_in_db_grants_access(self, tmp_path):
        """Direct API key path (no JWT): X-API-Key looked up via OpsStore."""
        store = _make_store(tmp_path)
        _seed_user(store, "uid-admin", "alice", "admin", "direct-admin-key")
        priv_pem, pub_pem, _ = _make_rsa_keypair()
        app = self._make_app(store, priv_pem, pub_pem)

        with patch.dict(os.environ, {"INIDS_AUTH_COMPAT": "false"}, clear=False):
            with app.test_client() as client:
                resp = client.get(
                    "/admin-only", headers={"X-API-Key": "direct-admin-key"}
                )
        assert resp.status_code == 200
