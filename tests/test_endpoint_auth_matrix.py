"""Comprehensive RBAC matrix tests for every protected endpoint.

Verifies that:
- Unauthenticated requests return 401 on protected endpoints when auth is on.
- viewer role cannot access analyst-only or admin-only endpoints.
- analyst role can access analyst endpoints but not admin-only ones.
- admin role can access all protected endpoints.
- Open endpoints (health, live, ready, /api/predict) are accessible without keys.

Matrix rows  = (endpoint, method, min_required_role)
Matrix cols  = (no-key, viewer, analyst, admin)
Expected     = 401 | 401 | 200/other | 200/other
"""
from __future__ import annotations

import pytest
import web_app.app as app_module
from src.ops_store import OpsStore
import src.auth_service as auth_module

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

TOKENS = {
    "viewer": "viewer-tok",
    "analyst": "analyst-tok",
    "admin": "admin-tok",
}

PRINCIPALS = {
    TOKENS["viewer"]: auth_module.Principal(role="viewer", token=TOKENS["viewer"]),
    TOKENS["analyst"]: auth_module.Principal(role="analyst", token=TOKENS["analyst"]),
    TOKENS["admin"]: auth_module.Principal(role="admin", token=TOKENS["admin"]),
}


def _setup_client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "detection_service", None)
    monkeypatch.setattr(app_module, "all_models", {})
    monkeypatch.setattr(app_module, "load_models", lambda: None)
    monkeypatch.setattr(app_module, "ops_store", OpsStore(str(tmp_path / "ops.db")))
    monkeypatch.setattr(auth_module._auth_service, "principals", PRINCIPALS)
    return app_module.app.test_client()


def _get(client, path, role=None):
    headers = {"X-API-Key": TOKENS[role]} if role else {}
    return client.get(path, headers=headers)


def _post(client, path, role=None, body=None):
    headers = {"X-API-Key": TOKENS[role]} if role else {}
    return client.post(path, json=body or {}, headers=headers)


def _delete(client, path, role=None):
    headers = {"X-API-Key": TOKENS[role]} if role else {}
    return client.delete(path, headers=headers)


# ---------------------------------------------------------------------------
# Open (no-auth) endpoints
# ---------------------------------------------------------------------------


class TestOpenEndpoints:
    def test_health_no_key(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        resp = _get(client, "/api/health")
        assert resp.status_code == 200

    def test_health_live_no_key(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _get(client, "/api/health/live").status_code == 200

    def test_health_ready_no_key(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        # /ready returns 200 when fully ready, 503 when not — both are valid and not 401
        assert _get(client, "/api/health/ready").status_code in (200, 503)

    def test_predict_no_key_still_200(self, monkeypatch, tmp_path):
        """POST /api/predict has no role guard — should function without a key."""
        client = _setup_client(monkeypatch, tmp_path)
        resp = _post(client, "/api/predict", body={"features": {"duration": 0}})
        # Model not loaded → 200 or 500, but NOT 401
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# analyst-required endpoints
# ---------------------------------------------------------------------------

ANALYST_ENDPOINTS = [
    ("GET",  "/api/alerts"),
    ("GET",  "/api/engines"),
    ("GET",  "/api/allowlist"),
    ("GET",  "/api/threat-intel/stats"),
    ("GET",  "/api/metrics"),
    ("GET",  "/api/models/registry"),
]


class TestAnalystEndpoints:
    @pytest.mark.parametrize("method,path", ANALYST_ENDPOINTS)
    def test_no_key_returns_401(self, monkeypatch, tmp_path, method, path):
        client = _setup_client(monkeypatch, tmp_path)
        resp = _get(client, path) if method == "GET" else _post(client, path)
        assert resp.status_code == 401, f"{method} {path}: expected 401, got {resp.status_code}"

    @pytest.mark.parametrize("method,path", ANALYST_ENDPOINTS)
    def test_viewer_returns_401(self, monkeypatch, tmp_path, method, path):
        client = _setup_client(monkeypatch, tmp_path)
        resp = _get(client, path, role="viewer") if method == "GET" else _post(client, path, role="viewer")
        assert resp.status_code == 401, f"{method} {path} as viewer: expected 401"

    @pytest.mark.parametrize("method,path", ANALYST_ENDPOINTS)
    def test_analyst_is_allowed(self, monkeypatch, tmp_path, method, path):
        client = _setup_client(monkeypatch, tmp_path)
        resp = _get(client, path, role="analyst") if method == "GET" else _post(client, path, role="analyst")
        assert resp.status_code != 401, f"{method} {path} as analyst: got 401"

    @pytest.mark.parametrize("method,path", ANALYST_ENDPOINTS)
    def test_admin_is_allowed(self, monkeypatch, tmp_path, method, path):
        client = _setup_client(monkeypatch, tmp_path)
        resp = _get(client, path, role="admin") if method == "GET" else _post(client, path, role="admin")
        assert resp.status_code != 401, f"{method} {path} as admin: got 401"


# ---------------------------------------------------------------------------
# admin-required endpoints
# ---------------------------------------------------------------------------

ADMIN_ENDPOINTS_GET = [
    "/api/policy",
    "/api/policy/history",
    "/api/audit",
    "/api/siem/flush",
]

ADMIN_ENDPOINTS_POST = [
    "/api/policy",
    "/api/policy/rollback",
    "/api/actions/cleanup",
    "/api/allowlist",
]


class TestAdminGetEndpoints:
    @pytest.mark.parametrize("path", ADMIN_ENDPOINTS_GET)
    def test_no_key_returns_401(self, monkeypatch, tmp_path, path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _get(client, path).status_code == 401

    @pytest.mark.parametrize("path", ADMIN_ENDPOINTS_GET)
    def test_viewer_returns_401(self, monkeypatch, tmp_path, path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _get(client, path, role="viewer").status_code == 401

    @pytest.mark.parametrize("path", ADMIN_ENDPOINTS_GET)
    def test_analyst_returns_401(self, monkeypatch, tmp_path, path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _get(client, path, role="analyst").status_code == 401

    @pytest.mark.parametrize("path", ADMIN_ENDPOINTS_GET)
    def test_admin_allowed(self, monkeypatch, tmp_path, path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _get(client, path, role="admin").status_code != 401


class TestAdminPostEndpoints:
    @pytest.mark.parametrize("path", ADMIN_ENDPOINTS_POST)
    def test_no_key_returns_401(self, monkeypatch, tmp_path, path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _post(client, path).status_code == 401

    @pytest.mark.parametrize("path", ADMIN_ENDPOINTS_POST)
    def test_viewer_returns_401(self, monkeypatch, tmp_path, path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _post(client, path, role="viewer").status_code == 401

    @pytest.mark.parametrize("path", ADMIN_ENDPOINTS_POST)
    def test_analyst_returns_401(self, monkeypatch, tmp_path, path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _post(client, path, role="analyst").status_code == 401

    @pytest.mark.parametrize("path", ADMIN_ENDPOINTS_POST)
    def test_admin_allowed(self, monkeypatch, tmp_path, path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _post(client, path, role="admin").status_code != 401


class TestEngineToggleAdminOnly:
    """Engine toggle requires admin; listing requires analyst."""

    def test_toggle_no_key_returns_401(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _post(client, "/api/engines/ml/toggle").status_code == 401

    def test_toggle_analyst_returns_401(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _post(client, "/api/engines/ml/toggle", role="analyst").status_code == 401

    def test_toggle_admin_allowed(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        resp = _post(client, "/api/engines/ml/toggle", role="admin")
        assert resp.status_code != 401


class TestAlertFeedbackAnalystOnly:
    def test_feedback_no_key_returns_401(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _post(client, "/api/alerts/abc123/feedback").status_code == 401

    def test_feedback_viewer_returns_401(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _post(client, "/api/alerts/abc123/feedback", role="viewer").status_code == 401

    def test_feedback_analyst_allowed(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        resp = _post(client, "/api/alerts/abc123/feedback", role="analyst", body={"correct": True})
        assert resp.status_code != 401


class TestIngestEndpoints:
    def test_ingest_no_key_returns_401(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _post(client, "/api/ingest", body={"records": []}).status_code == 401

    def test_ingest_analyst_allowed(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        resp = _post(client, "/api/ingest", role="analyst", body={"records": []})
        assert resp.status_code != 401

    def test_ingest_process_no_key_returns_401(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _post(client, "/api/ingest/process").status_code == 401

    def test_ingest_process_analyst_allowed(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        resp = _post(client, "/api/ingest/process", role="analyst")
        assert resp.status_code != 401


class TestAllowlistDeleteAdminOnly:
    def test_delete_no_key_returns_401(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _delete(client, "/api/allowlist/10.0.0.1").status_code == 401

    def test_delete_analyst_returns_401(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        assert _delete(client, "/api/allowlist/10.0.0.1", role="analyst").status_code == 401

    def test_delete_admin_allowed(self, monkeypatch, tmp_path):
        client = _setup_client(monkeypatch, tmp_path)
        resp = _delete(client, "/api/allowlist/10.0.0.1", role="admin")
        assert resp.status_code != 401
