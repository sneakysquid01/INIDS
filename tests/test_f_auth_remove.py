"""F-AUTH-REMOVE security regression tests (Phase F).

Validates:
1. src/auth_service.py deleted — no importable legacy AuthService
2. src/auth_jwt.py deleted — no importable legacy JWTAuthManager/RunAsManager
3. INIDS_AUTH_COMPAT removed from decorators.py and validators.py
4. require_roles() in app.py replaces all @require_role / @require_auth
5. All routes have _required_roles attribute (set by require_roles decorator)
6. UnifiedAuthService is the sole authentication path
"""
from __future__ import annotations

import importlib.util
import ast
import os
import pathlib

import pytest

SRC = pathlib.Path(__file__).parent.parent / "src"
WEB = pathlib.Path(__file__).parent.parent / "web_app"


# ---------------------------------------------------------------------------
# 1. Legacy modules deleted
# ---------------------------------------------------------------------------


def test_legacy_auth_service_deleted():
    """src/auth_service.py must not exist after F-AUTH-REMOVE."""
    spec = importlib.util.find_spec("src.auth_service")
    assert spec is None, "src/auth_service.py found — must be deleted by F-AUTH-REMOVE"


def test_legacy_auth_jwt_deleted():
    """src/auth_jwt.py must not exist after F-AUTH-REMOVE."""
    spec = importlib.util.find_spec("src.auth_jwt")
    assert spec is None, "src/auth_jwt.py found — must be deleted by F-AUTH-REMOVE"


# ---------------------------------------------------------------------------
# 2. INIDS_AUTH_COMPAT removed from source code
# ---------------------------------------------------------------------------


def _grep_source(pattern: str) -> list[str]:
    """Return file paths under src/ and web_app/ containing pattern."""
    hits = []
    for root_dir in (SRC, WEB):
        for path in root_dir.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                if pattern in text:
                    hits.append(str(path))
            except Exception:
                pass
    return hits


def test_inids_auth_compat_removed_from_decorators():
    """src/auth/decorators.py must not reference INIDS_AUTH_COMPAT."""
    path = SRC / "auth" / "decorators.py"
    assert path.exists(), "src/auth/decorators.py not found"
    assert "INIDS_AUTH_COMPAT" not in path.read_text(), (
        "INIDS_AUTH_COMPAT still present in src/auth/decorators.py"
    )


def test_inids_auth_compat_removed_from_validators():
    """src/auth/validators.py must not reference INIDS_AUTH_COMPAT."""
    path = SRC / "auth" / "validators.py"
    assert path.exists(), "src/auth/validators.py not found"
    assert "INIDS_AUTH_COMPAT" not in path.read_text(), (
        "INIDS_AUTH_COMPAT still present in src/auth/validators.py"
    )


def test_no_legacy_auth_import_in_app():
    """web_app/app.py must not import from src.auth_service or src.auth_jwt."""
    app_text = (WEB / "app.py").read_text(encoding="utf-8")
    assert "from src.auth_service" not in app_text, (
        "app.py still imports from legacy src.auth_service"
    )
    assert "from src.auth_jwt" not in app_text, (
        "app.py still imports from legacy src.auth_jwt"
    )


def test_no_legacy_auth_import_in_decorators():
    """src/auth/decorators.py must not import from src.auth_service."""
    text = (SRC / "auth" / "decorators.py").read_text(encoding="utf-8")
    assert "from src.auth_service" not in text


# ---------------------------------------------------------------------------
# 3. All app routes protected by require_roles (via _required_roles attribute)
# ---------------------------------------------------------------------------


def test_all_routes_have_required_roles_attribute():
    """Every non-public route must have _required_roles (set by @require_roles)."""
    import web_app.app as app_module
    PUBLIC = app_module.PUBLIC_ROUTES | {"/static/<path:filename>"}

    uncovered = []
    for rule in app_module.app.url_map.iter_rules():
        if rule.rule in PUBLIC or rule.rule.startswith("/static/"):
            continue
        fn = app_module.app.view_functions.get(rule.endpoint)
        if fn is None:
            continue
        if not hasattr(fn, "_required_roles"):
            uncovered.append(f"{rule.endpoint} ({rule.rule})")

    assert not uncovered, (
        f"Routes missing @require_roles decorator: {uncovered}"
    )


# ---------------------------------------------------------------------------
# 4. require_roles is the sole auth decorator in app.py
# ---------------------------------------------------------------------------


def test_app_has_no_require_auth_decorator():
    """app.py must not use the old @require_auth decorator."""
    app_text = (WEB / "app.py").read_text(encoding="utf-8")
    assert "@require_auth" not in app_text, (
        "app.py still uses legacy @require_auth — replace with @require_roles"
    )


def test_app_has_no_jwt_require_role_decorator():
    """app.py must not use @jwt_require_role."""
    app_text = (WEB / "app.py").read_text(encoding="utf-8")
    assert "@jwt_require_role" not in app_text


def test_app_has_no_legacy_require_role():
    """app.py must not import or use the legacy @require_role (without 's')."""
    app_text = (WEB / "app.py").read_text(encoding="utf-8")
    # Allow 'require_roles' but not 'require_role(' (without the 's')
    import re
    # Match @require_role( but not @require_roles(
    legacy_uses = re.findall(r"@require_role\(", app_text)
    assert not legacy_uses, (
        f"Found {len(legacy_uses)} uses of legacy @require_role() in app.py"
    )


# ---------------------------------------------------------------------------
# 5. UnifiedAuthService authentication round-trip
# ---------------------------------------------------------------------------


def test_unified_auth_rejects_unknown_key(tmp_path):
    """authenticate_api_key() returns None for a key not in OpsStore."""
    from src.ops_store import OpsStore
    from src.auth.auth_service import UnifiedAuthService
    store = OpsStore(str(tmp_path / "ops.db"))
    svc = UnifiedAuthService(store)
    assert svc.authenticate_api_key("ghost-key-xyz") is None


def test_unified_auth_accepts_seeded_key(tmp_path, monkeypatch):
    """authenticate_api_key() returns AuthContext for a key seeded via env var."""
    monkeypatch.setenv("INIDS_ADMIN_API_KEY", "test-admin-key-f-remove")
    from src.ops_store import OpsStore
    from src.auth.auth_service import UnifiedAuthService
    store = OpsStore(str(tmp_path / "ops.db"))
    svc = UnifiedAuthService(store)
    ctx = svc.authenticate_api_key("test-admin-key-f-remove")
    assert ctx is not None
    assert "admin" in ctx.roles


def test_unified_auth_jwt_roundtrip(tmp_path, monkeypatch):
    """create_token() → authenticate_jwt() round-trip succeeds."""
    monkeypatch.setenv("INIDS_ADMIN_API_KEY", "jwt-roundtrip-key")
    from src.ops_store import OpsStore
    from src.auth.auth_service import UnifiedAuthService
    from src.auth.jwt_manager import reset_jwt_manager
    reset_jwt_manager()
    try:
        store = OpsStore(str(tmp_path / "ops.db"))
        svc = UnifiedAuthService(store)
        token = svc.create_token("u1", "admin_user", ["admin"])
        ctx = svc.authenticate_jwt(token)
        assert ctx is not None
        assert "admin" in ctx.roles
    finally:
        reset_jwt_manager()


def test_rs256_rejects_hs256_token():
    """RS256JWTManager must reject tokens signed with HS256."""
    import jwt as pyjwt
    from src.auth.jwt_manager import RS256JWTManager, reset_jwt_manager
    reset_jwt_manager()
    try:
        hs256_token = pyjwt.encode(
            {"sub": "attacker", "roles": ["admin"], "aud": "INIDS-API"},
            "secret",
            algorithm="HS256",
        )
        mgr = RS256JWTManager()
        ok, _, reason = mgr.verify_token(hs256_token)
        assert ok is False, "RS256JWTManager accepted an HS256 token — algorithm confusion attack"
    finally:
        reset_jwt_manager()
