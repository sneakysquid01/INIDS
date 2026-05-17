"""Canonical attack-chain regression tests.

AC-1: Credential-bypass / expired-token replay
AC-2: Model substitution — covered in test_model_load_rejects_bad_checksum.py
AC-3: Supply-chain compromise (pinned hashes + container isolation)
AC-4: Role-confusion via FP-suppression — covered in test_all_routes_require_auth.py

This file provides the exact test names referenced in PLAN.md line 2646
so the go/no-go checklist (B-GATE-1) can be verified by name.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# AC-1: Credential bypass / expired-token replay
# ---------------------------------------------------------------------------

class TestAC1CredentialBypass:
    def test_login_requires_valid_credentials(self, tmp_path):
        """POST /api/auth/login without a valid X-API-Key must return 401."""
        import os
        os.environ.setdefault("OPS_DB_PATH", str(tmp_path / "ops.db"))

        from web_app.app import app
        client = app.test_client()

        # No X-API-Key header
        resp = client.post("/api/auth/login")
        assert resp.status_code == 401, (
            "AC-1: login endpoint must reject requests with no X-API-Key"
        )

        # Wrong key
        resp = client.post("/api/auth/login", headers={"X-API-Key": "totally-wrong"})
        assert resp.status_code == 401, (
            "AC-1: login endpoint must reject invalid API keys"
        )

    def test_expired_token_not_refreshable(self, tmp_path):
        """Expired tokens outside the 300 s grace window must be rejected."""
        import time
        import jwt
        import os

        os.environ.setdefault("OPS_DB_PATH", str(tmp_path / "ops.db"))

        from web_app.app import app

        # Grab the secret key
        secret = app.config.get("SECRET_KEY", "test-secret")

        # Build a token that expired 400 s ago (outside the 300 s grace window)
        now = int(time.time())
        payload = {
            "sub": "testuser",
            "role": "viewer",
            "exp": now - 400,
            "iat": now - 3600,
        }
        stale_token = jwt.encode(payload, secret, algorithm="HS256")

        client = app.test_client()
        resp = client.post(
            "/api/auth/refresh",
            headers={"Authorization": f"Bearer {stale_token}"},
        )
        assert resp.status_code in (401, 422), (
            f"AC-1: token expired 400 s ago (outside 300 s grace) must be rejected, got {resp.status_code}"
        )

    def test_allow_unauthenticated_env_is_fail_closed(self):
        """ALLOW_UNAUTHENTICATED=true must cause a RuntimeError at startup."""
        import importlib
        import os
        import sys

        # Temporarily set the env var
        old = os.environ.pop("ALLOW_UNAUTHENTICATED", None)
        os.environ["ALLOW_UNAUTHENTICATED"] = "true"
        try:
            # Force reload of settings module
            if "src.settings" in sys.modules:
                del sys.modules["src.settings"]
            import src.settings as settings_mod
            with pytest.raises(RuntimeError, match="ALLOW_UNAUTHENTICATED"):
                settings_mod.load_settings()
        finally:
            if old is not None:
                os.environ["ALLOW_UNAUTHENTICATED"] = old
            else:
                os.environ.pop("ALLOW_UNAUTHENTICATED", None)
            if "src.settings" in sys.modules:
                del sys.modules["src.settings"]


# ---------------------------------------------------------------------------
# AC-3: Supply-chain compromise
# ---------------------------------------------------------------------------

class TestAC3SupplyChain:
    def test_requirements_are_pinned_with_hashes(self):
        """Every entry in requirements.txt must have at least one --hash=sha256 pin."""
        req_path = ROOT / "requirements.txt"
        assert req_path.exists(), "requirements.txt must exist"

        content = req_path.read_text(encoding="utf-8")
        assert "--hash=sha256:" in content, (
            "AC-3: requirements.txt must contain SHA-256 hashes (run scripts/_gen_hashed_requirements.py)"
        )

        # Count package entries vs hash entries — every package must be hashed
        pkg_lines = [
            line for line in content.splitlines()
            if line and not line.startswith("#") and not line.startswith(" ") and "==" in line
        ]
        hash_lines = [line for line in content.splitlines() if "--hash=sha256:" in line]

        assert len(pkg_lines) > 0, "requirements.txt has no package entries"
        assert len(hash_lines) >= len(pkg_lines), (
            f"AC-3: {len(pkg_lines)} packages but only {len(hash_lines)} hash entries — "
            "some packages are unpinned"
        )

    def test_container_does_not_mount_source_tree(self):
        """docker-compose.yml must not bind-mount the repo root into the container."""
        compose_path = ROOT / "deploy" / "compose" / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml must exist"

        content = compose_path.read_text(encoding="utf-8")

        # Check only non-comment lines for dangerous bind-mount patterns
        non_comment_lines = "\n".join(
            line for line in content.splitlines() if not line.strip().startswith("#")
        )

        # The old vulnerability: '../../:/app' or similar repo-root bind mounts in volumes
        # Note: build.context: ../.. is legitimate; only volumes: entries are dangerous
        volume_section = re.search(r"volumes:\s*\n((?:\s+-.+\n?)*)", non_comment_lines)
        if volume_section:
            volume_entries = volume_section.group(0)
            assert "../../" not in volume_entries, (
                "AC-3: docker-compose.yml volumes must not bind-mount the repo root (../../)"
            )

        # Must use named volumes, not host-path mounts pointing to repo root
        assert "inids-data:" in content or "inids-models:" in content, (
            "AC-3: docker-compose.yml must use named volumes for data and models"
        )

    def test_dockerfile_uses_require_hashes(self):
        """Dockerfile pip install must use --require-hashes."""
        dockerfile_path = ROOT / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile must exist"

        content = dockerfile_path.read_text(encoding="utf-8")
        assert "--require-hashes" in content, (
            "AC-3: Dockerfile must pass --require-hashes to pip install (A-07)"
        )

    def test_dockerignore_excludes_env_files(self):
        """.dockerignore must exclude .env files to prevent secrets leaking into image."""
        di_path = ROOT / ".dockerignore"
        assert di_path.exists(), ".dockerignore must exist"

        content = di_path.read_text(encoding="utf-8")
        assert ".env" in content, "AC-3: .dockerignore must exclude .env files"
        assert "*.key" in content or "*.pem" in content, (
            "AC-3: .dockerignore must exclude private key files"
        )
