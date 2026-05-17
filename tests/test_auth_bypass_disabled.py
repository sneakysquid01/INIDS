"""
Regression tests for PLAN.md Phase A Step 1 (A-01): Auth Bypass Removal.
These tests verify that the ALLOW_UNAUTHENTICATED bypass is permanently disabled.
"""
import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_load_settings_raises_on_allow_unauthenticated_true(monkeypatch):
    """load_settings() must raise RuntimeError when ALLOW_UNAUTHENTICATED=true."""
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "true")
    monkeypatch.setenv("SECRET_KEY", "a" * 32)
    # Re-import to bypass cached module state
    import importlib
    import src.settings as settings_mod
    importlib.reload(settings_mod)
    with pytest.raises(RuntimeError, match="ALLOW_UNAUTHENTICATED"):
        settings_mod.load_settings()


@pytest.mark.parametrize("value", ["1", "yes", "on", "TRUE", "Yes"])
def test_load_settings_raises_on_all_truthy_variants(monkeypatch, value):
    """load_settings() must reject all truthy variants of the bypass flag."""
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", value)
    monkeypatch.setenv("SECRET_KEY", "b" * 32)
    import importlib
    import src.settings as settings_mod
    importlib.reload(settings_mod)
    with pytest.raises(RuntimeError, match="ALLOW_UNAUTHENTICATED"):
        settings_mod.load_settings()


def test_auth_service_has_no_bypass_enabled_method():
    """AuthService must not have a _bypass_enabled method (it was the bypass gate)."""
    from src.auth_service import AuthService
    svc = AuthService()
    assert not hasattr(svc, "_bypass_enabled"), (
        "_bypass_enabled() must be removed from AuthService — it is the auth bypass"
    )


def test_auth_service_has_no_allow_unauthenticated_attribute():
    """AuthService must not store allow_unauthenticated as an attribute."""
    from src.auth_service import AuthService
    svc = AuthService()
    assert not hasattr(svc, "allow_unauthenticated"), (
        "allow_unauthenticated attribute must be removed from AuthService"
    )


def test_auth_service_requires_api_key_when_principals_configured(monkeypatch):
    """When API keys are configured, AuthService.authorize() must reject missing key."""
    monkeypatch.setenv("INIDS_ADMIN_API_KEY", "test-admin-key-xyz")
    monkeypatch.delenv("INIDS_SENSOR_API_KEY", raising=False)
    monkeypatch.delenv("INIDS_VIEWER_API_KEY", raising=False)
    monkeypatch.delenv("INIDS_ANALYST_API_KEY", raising=False)

    import importlib
    import src.auth_service as auth_mod
    importlib.reload(auth_mod)

    from flask import Flask
    app = Flask(__name__)
    with app.test_request_context("/", headers={}):
        svc = auth_mod.AuthService()
        ok, reason = svc.authorize("viewer")
        assert not ok, f"Expected authorization to fail without API key, got ok=True reason={reason}"
        assert reason == "missing_api_key"


def test_auth_service_accepts_valid_api_key(monkeypatch):
    """AuthService.authorize() must succeed with a correct API key."""
    monkeypatch.setenv("INIDS_ADMIN_API_KEY", "valid-admin-key-abc123")
    monkeypatch.delenv("INIDS_SENSOR_API_KEY", raising=False)
    monkeypatch.delenv("INIDS_VIEWER_API_KEY", raising=False)
    monkeypatch.delenv("INIDS_ANALYST_API_KEY", raising=False)

    import importlib
    import src.auth_service as auth_mod
    importlib.reload(auth_mod)

    from flask import Flask
    app = Flask(__name__)
    with app.test_request_context("/", headers={"X-API-Key": "valid-admin-key-abc123"}):
        svc = auth_mod.AuthService()
        ok, reason = svc.authorize("admin")
        assert ok, f"Expected authorization to succeed, got reason={reason}"
        assert reason == "admin"
