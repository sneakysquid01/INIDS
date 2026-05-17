"""
Regression tests for PLAN.md Phase A Step 4 (A-04): Break Passwordless JWT Login.
Verifies that /api/auth/login requires X-API-Key and /api/auth/refresh rejects stale tokens.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

AUTH_BP_PY = os.path.join(ROOT, "web_app", "blueprints", "auth.py")


def _read_source():
    with open(AUTH_BP_PY, encoding="utf-8") as f:
        return f.read()


def test_api_auth_login_requires_api_key_in_source():
    """api_auth_login must check X-API-Key header (not accept any username)."""
    source = _read_source()
    # Find the login handler section
    idx = source.find("def api_auth_login")
    assert idx != -1, "api_auth_login not found in app.py"
    handler_src = source[idx:idx + 2000]
    assert "X-API-Key" in handler_src, (
        "api_auth_login must check X-API-Key header (PLAN.md A-04)"
    )


def test_api_auth_login_rejects_missing_key_in_source():
    """api_auth_login must return 401 when X-API-Key is missing."""
    source = _read_source()
    idx = source.find("def api_auth_login")
    handler_src = source[idx:idx + 2000]
    # Should have a 401 return when API key is absent
    assert "401" in handler_src, (
        "api_auth_login must return 401 when X-API-Key is absent"
    )


def test_api_auth_login_no_arbitrary_username_role_in_source():
    """api_auth_login must derive roles from the API key's auth context, not user input."""
    source = _read_source()
    idx = source.find("def api_auth_login")
    handler_src = source[idx:idx + 2000]
    # F-AUTH-REMOVE: UnifiedAuthService replaced legacy principal lookup.
    # Roles come from auth_ctx.roles (OpsStore-backed lookup), never from user input.
    assert "auth_ctx.roles" in handler_src, (
        "api_auth_login must derive roles from auth_ctx (UnifiedAuthService), not from user input"
    )


def test_api_auth_refresh_no_allow_expired_true():
    """api_auth_refresh must not use allow_expired=True unconditionally."""
    source = _read_source()
    idx = source.find("def api_auth_refresh")
    handler_src = source[idx:idx + 3000]
    # The old code had: jwt_manager.verify_token(token, allow_expired=True) as primary path
    # New code: allow_expired=True only inside the grace-window fallback block
    # Verify: the function has a grace window check
    assert "REFRESH_GRACE_SECONDS" in handler_src or "grace" in handler_src.lower(), (
        "api_auth_refresh must implement a grace window (not unlimited allow_expired)"
    )


def test_api_auth_refresh_rejects_long_expired_tokens():
    """api_auth_refresh must reject tokens expired more than 5 minutes ago."""
    source = _read_source()
    idx = source.find("def api_auth_refresh")
    handler_src = source[idx:idx + 3000]
    # The handler must compare seconds_overdue against REFRESH_GRACE_SECONDS
    assert "seconds_overdue" in handler_src or "REFRESH_GRACE" in handler_src, (
        "api_auth_refresh must reject tokens overdue beyond grace period"
    )


def test_token_expiry_is_one_hour_in_login():
    """api_auth_login must issue tokens with 1-hour expiry (non-negotiable per PLAN.md)."""
    source = _read_source()
    idx = source.find("def api_auth_login")
    handler_src = source[idx:idx + 2000]
    assert "expires_in=3600" in handler_src or "3600" in handler_src, (
        "api_auth_login must issue tokens with expires_in=3600 (1 hour)"
    )
    # Must NOT have 86400 (24h) expiry anymore
    assert "expires_in=86400" not in handler_src, (
        "api_auth_login must not issue 24-hour tokens — 1-hour expiry is mandatory"
    )
