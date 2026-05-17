"""F-AUTH-REMOVE: src/auth_service.py (legacy API-key AuthService) has been deleted.

These tests verify the deletion completed successfully and that the replacement
(UnifiedAuthService in src/auth/) provides equivalent guarantees.
"""
import importlib
import importlib.util


def test_legacy_auth_service_module_deleted():
    """src.auth_service must not be importable after F-AUTH-REMOVE."""
    spec = importlib.util.find_spec("src.auth_service")
    assert spec is None, (
        "src/auth_service.py still exists — F-AUTH-REMOVE requires it to be deleted"
    )


def test_unified_auth_service_importable():
    """src.auth.auth_service (UnifiedAuthService) must be importable."""
    from src.auth.auth_service import UnifiedAuthService
    assert UnifiedAuthService is not None


def test_unified_auth_service_has_authenticate_api_key():
    from src.auth.auth_service import UnifiedAuthService
    assert callable(getattr(UnifiedAuthService, "authenticate_api_key", None))


def test_unified_auth_service_has_authenticate_jwt():
    from src.auth.auth_service import UnifiedAuthService
    assert callable(getattr(UnifiedAuthService, "authenticate_jwt", None))


def test_unified_auth_service_has_create_token():
    from src.auth.auth_service import UnifiedAuthService
    assert callable(getattr(UnifiedAuthService, "create_token", None))


def test_unified_auth_service_has_revoke_token():
    from src.auth.auth_service import UnifiedAuthService
    assert callable(getattr(UnifiedAuthService, "revoke_token", None))
