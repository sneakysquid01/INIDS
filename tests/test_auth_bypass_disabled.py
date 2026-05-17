"""
Regression tests for PLAN.md Phase A Step 1 (A-01): Auth Bypass Removal.
F-AUTH-REMOVE: tests that imported legacy AuthService (auth_service.py) are
replaced with equivalent tests on settings.py (still authoritative for bypass
prevention) and checks that the legacy module is gone.
"""
import os
import sys
import importlib
import importlib.util
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_load_settings_raises_on_allow_unauthenticated_true(monkeypatch):
    """load_settings() must raise RuntimeError when ALLOW_UNAUTHENTICATED=true."""
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "true")
    monkeypatch.setenv("SECRET_KEY", "a" * 32)
    import src.settings as settings_mod
    importlib.reload(settings_mod)
    with pytest.raises(RuntimeError, match="ALLOW_UNAUTHENTICATED"):
        settings_mod.load_settings()


@pytest.mark.parametrize("value", ["1", "yes", "on", "TRUE", "Yes"])
def test_load_settings_raises_on_all_truthy_variants(monkeypatch, value):
    """load_settings() must reject all truthy variants of the bypass flag."""
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", value)
    monkeypatch.setenv("SECRET_KEY", "b" * 32)
    import src.settings as settings_mod
    importlib.reload(settings_mod)
    with pytest.raises(RuntimeError, match="ALLOW_UNAUTHENTICATED"):
        settings_mod.load_settings()


def test_legacy_auth_service_deleted():
    """F-AUTH-REMOVE: src/auth_service.py must not be importable."""
    spec = importlib.util.find_spec("src.auth_service")
    assert spec is None, (
        "src/auth_service.py still exists — must be deleted by F-AUTH-REMOVE"
    )


def test_unified_auth_service_has_no_bypass():
    """UnifiedAuthService must not have a bypass-enabling method or attribute."""
    from src.auth.auth_service import UnifiedAuthService
    assert not hasattr(UnifiedAuthService, "_bypass_enabled")
    assert not hasattr(UnifiedAuthService, "allow_unauthenticated")


def test_unified_auth_rejects_missing_api_key(tmp_path):
    """UnifiedAuthService.authenticate_api_key() returns None for empty key."""
    from src.ops_store import OpsStore
    from src.auth.auth_service import UnifiedAuthService
    store = OpsStore(str(tmp_path / "ops.db"))
    svc = UnifiedAuthService(store)
    assert svc.authenticate_api_key("") is None
    assert svc.authenticate_api_key("nonexistent-key-xyz") is None


def test_unified_auth_accepts_seeded_api_key(tmp_path, monkeypatch):
    """UnifiedAuthService.authenticate_api_key() returns AuthContext for a seeded key."""
    monkeypatch.setenv("INIDS_ADMIN_API_KEY", "valid-admin-key-abc123")
    from src.ops_store import OpsStore
    from src.auth.auth_service import UnifiedAuthService
    store = OpsStore(str(tmp_path / "ops.db"))
    svc = UnifiedAuthService(store)
    ctx = svc.authenticate_api_key("valid-admin-key-abc123")
    assert ctx is not None
    assert "admin" in ctx.roles
