"""E-04: GET /api/health must NOT expose the OPS_DB file path in the response."""
import pytest
from unittest.mock import patch


def _make_client():
    from web_app.app import app
    app.config["TESTING"] = True
    return app.test_client()


def _health_json():
    """Return health JSON with model/pipeline loading mocked out."""
    client = _make_client()
    with patch("web_app.app.ensure_detection_service", return_value=False), \
         patch("web_app.app._ensure_pipeline_started"):
        resp = client.get("/api/health")
    return resp, resp.get_json() or {}


class TestHealthNoDB:
    def test_health_top_level_has_no_ops_db_key(self):
        """ops_db must NOT appear as a top-level key (it exposed the file path)."""
        resp, data = _health_json()
        assert resp.status_code == 200
        assert "ops_db" not in data, (
            f"Health endpoint still has top-level 'ops_db' key: {data.get('ops_db')}"
        )

    def test_health_top_level_has_no_db_path_key(self):
        resp, data = _health_json()
        assert resp.status_code == 200
        top_level_keys = list(data.keys())
        db_path_keys = [k for k in top_level_keys if "db_path" in k.lower()]
        assert db_path_keys == [], f"Found DB path keys: {db_path_keys}"

    def test_health_response_does_not_contain_ops_db_path_value(self):
        """The actual filesystem path string must not appear in the response body."""
        from web_app.app import OPS_DB_PATH
        client = _make_client()
        with patch("web_app.app.ensure_detection_service", return_value=False), \
             patch("web_app.app._ensure_pipeline_started"):
            resp = client.get("/api/health")
        body = resp.data.decode("utf-8", errors="replace")
        assert OPS_DB_PATH not in body, (
            f"OPS_DB_PATH value {OPS_DB_PATH!r} found in health response body"
        )

    def test_health_returns_status_ok(self):
        resp, data = _health_json()
        assert resp.status_code == 200
        assert data.get("status") == "ok"

    def test_health_has_required_safe_fields(self):
        """Health response must still have non-sensitive operational fields."""
        resp, data = _health_json()
        assert resp.status_code == 200
        assert "status" in data
        assert "model_loaded" in data
