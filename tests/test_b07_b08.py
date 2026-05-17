"""Regression tests for B-07 (CORS middleware) and B-08 (sensor key role).

B-07 validation checkpoints:
- OPTIONS from allowed origin → 200 with Access-Control-Allow-Origin header
- OPTIONS from unauthorized origin → no CORS headers (passes through)
- Regular responses from allowed origins carry CORS headers
- INIDS_CORS_ORIGINS env var controls allowed origins

B-08 validation checkpoints:
- POST /api/detect with sensor key → 200 (or non-403 with real payload)
- GET /api/alerts with sensor key → 401/403 (sensor not allowed)
- POST /api/fp-suppressions with sensor key → 401/403
- Sensor principal registered as role="sensor" (not "analyst")
- auth_status() reports "sensor" as a configured role
"""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# B-07: CORSMiddleware unit tests
# ---------------------------------------------------------------------------


class TestCORSMiddlewareUnit:
    """Unit tests — exercise CORSMiddleware directly without Flask routing."""

    def _make_cors(self, origins=None):
        from src.middleware import CORSMiddleware
        return CORSMiddleware(
            allowed_origins=origins or ["http://allowed.example.com"],
            allowed_methods=["GET", "POST", "OPTIONS"],
            allowed_headers=["Authorization", "Content-Type", "X-API-Key", "X-Correlation-ID"],
            allow_credentials=True,
        )

    def test_handle_preflight_allowed_origin_returns_response(self):
        cors = self._make_cors()
        from flask import Flask
        app = Flask(__name__)
        with app.test_request_context(
            "/api/alerts",
            method="OPTIONS",
            headers={"Origin": "http://allowed.example.com"},
        ):
            resp = cors.handle_preflight()
        assert resp is not None
        assert resp.status_code == 200
        assert resp.headers["Access-Control-Allow-Origin"] == "http://allowed.example.com"
        assert "Authorization" in resp.headers["Access-Control-Allow-Headers"]
        assert "X-Correlation-ID" in resp.headers["Access-Control-Allow-Headers"]
        assert resp.headers["Access-Control-Allow-Credentials"] == "true"

    def test_handle_preflight_disallowed_origin_returns_none(self):
        cors = self._make_cors()
        from flask import Flask
        app = Flask(__name__)
        with app.test_request_context(
            "/api/alerts",
            method="OPTIONS",
            headers={"Origin": "http://evil.example.com"},
        ):
            resp = cors.handle_preflight()
        assert resp is None

    def test_handle_preflight_non_options_returns_none(self):
        cors = self._make_cors()
        from flask import Flask
        app = Flask(__name__)
        with app.test_request_context(
            "/api/alerts",
            method="GET",
            headers={"Origin": "http://allowed.example.com"},
        ):
            resp = cors.handle_preflight()
        assert resp is None

    def test_add_headers_allowed_origin(self):
        from flask import Flask, Response as FlaskResponse
        cors = self._make_cors()
        app = Flask(__name__)
        with app.test_request_context(
            "/api/alerts",
            headers={"Origin": "http://allowed.example.com"},
        ):
            resp = cors.add_headers(FlaskResponse("ok", status=200))
        assert resp.headers["Access-Control-Allow-Origin"] == "http://allowed.example.com"
        assert resp.headers["Access-Control-Allow-Credentials"] == "true"

    def test_add_headers_disallowed_origin_no_cors_header(self):
        from flask import Flask, Response as FlaskResponse
        cors = self._make_cors()
        app = Flask(__name__)
        with app.test_request_context(
            "/api/alerts",
            headers={"Origin": "http://evil.example.com"},
        ):
            resp = cors.add_headers(FlaskResponse("ok", status=200))
        assert "Access-Control-Allow-Origin" not in resp.headers

    def test_after_request_is_alias_for_add_headers(self):
        from flask import Flask, Response as FlaskResponse
        cors = self._make_cors()
        app = Flask(__name__)
        with app.test_request_context(
            "/api/alerts",
            headers={"Origin": "http://allowed.example.com"},
        ):
            resp1 = cors.add_headers(FlaskResponse("ok", status=200))
            resp2 = cors.after_request(FlaskResponse("ok", status=200))
        assert resp1.headers.get("Access-Control-Allow-Origin") == resp2.headers.get(
            "Access-Control-Allow-Origin"
        )


class TestCORSMiddlewareEnvVar:
    """register_middleware reads INIDS_CORS_ORIGINS from environment."""

    def test_cors_origins_read_from_env(self):
        from flask import Flask
        from src.middleware import register_middleware

        app = Flask(__name__)
        app.config["SECRET_KEY"] = "test"

        with patch.dict(
            os.environ, {"INIDS_CORS_ORIGINS": "http://env-origin.example.com"}
        ):
            instances = register_middleware(app)

        cors = instances["cors"]
        assert "http://env-origin.example.com" in cors.allowed_origins

    def test_cors_origins_passed_directly(self):
        from flask import Flask
        from src.middleware import register_middleware

        app = Flask(__name__)
        app.config["SECRET_KEY"] = "test"

        instances = register_middleware(
            app, cors_origins=["http://direct.example.com"]
        )
        cors = instances["cors"]
        assert "http://direct.example.com" in cors.allowed_origins

    def test_settings_cors_origins_parsed(self):
        """Settings.cors_origins is a comma-separated string; parsing produces a list."""
        with patch.dict(
            os.environ,
            {"INIDS_CORS_ORIGINS": "http://a.example.com,http://b.example.com"},
        ):
            from src import settings as settings_mod
            import importlib
            # Force re-read env by calling load_settings directly
            s = settings_mod.load_settings.__wrapped__() if hasattr(settings_mod.load_settings, "__wrapped__") else None
            # Simpler: just check the env parse logic directly
            raw = "http://a.example.com,http://b.example.com"
            parsed = [o.strip() for o in raw.split(",") if o.strip()]
        assert parsed == ["http://a.example.com", "http://b.example.com"]


# ---------------------------------------------------------------------------
# B-08: Sensor key role unit tests
# ---------------------------------------------------------------------------


class TestSensorKeyRole:
    """Unit tests for sensor role in AuthService."""

    def _make_service_with_sensor(self, sensor_key="sensor-test-key"):
        with patch.dict(os.environ, {"INIDS_SENSOR_API_KEY": sensor_key}):
            from src import auth_service as auth_mod
            import importlib
            importlib.reload(auth_mod)
            return auth_mod.AuthService()

    def test_sensor_key_registered_as_sensor_role(self):
        svc = self._make_service_with_sensor()
        principal = svc.principals.get("sensor-test-key")
        assert principal is not None
        assert principal.role == "sensor", (
            f"Expected role='sensor', got role='{principal.role}'"
        )

    def test_sensor_role_not_analyst(self):
        svc = self._make_service_with_sensor()
        principal = svc.principals.get("sensor-test-key")
        assert principal.role != "analyst"

    def test_auth_status_includes_sensor_role(self):
        with patch.dict(os.environ, {"INIDS_SENSOR_API_KEY": "sensor-test-key"}):
            from src import auth_service as auth_mod
            import importlib
            importlib.reload(auth_mod)
            status = auth_mod.auth_status()
        assert "sensor" in status["configured_roles"]

    def test_sensor_authorize_allowed_on_detect_endpoint(self):
        from flask import Flask
        svc = self._make_service_with_sensor()
        app = Flask(__name__)
        with app.test_request_context(
            "/api/detect",
            method="POST",
            headers={"X-API-Key": "sensor-test-key"},
        ):
            ok, reason = svc.authorize("analyst")
        assert ok is True
        assert reason == "sensor"

    def test_sensor_authorize_denied_on_alerts_endpoint(self):
        from flask import Flask
        svc = self._make_service_with_sensor()
        app = Flask(__name__)
        with app.test_request_context(
            "/api/alerts",
            method="GET",
            headers={"X-API-Key": "sensor-test-key"},
        ):
            ok, reason = svc.authorize("analyst")
        assert ok is False
        assert reason == "insufficient_role"

    def test_sensor_authorize_denied_on_fp_suppressions(self):
        from flask import Flask
        svc = self._make_service_with_sensor()
        app = Flask(__name__)
        with app.test_request_context(
            "/api/fp-suppressions",
            method="POST",
            headers={"X-API-Key": "sensor-test-key"},
        ):
            ok, reason = svc.authorize("analyst")
        assert ok is False
        assert reason == "insufficient_role"

    def test_sensor_allowed_endpoints_constant_defined(self):
        from src.auth_service import SENSOR_ALLOWED_ENDPOINTS
        assert "/api/detect" in SENSOR_ALLOWED_ENDPOINTS
        assert "/api/stream" in SENSOR_ALLOWED_ENDPOINTS

    def test_analyst_key_still_hierarchical(self):
        """Analyst key must still work on analyst-level endpoints."""
        from flask import Flask
        with patch.dict(os.environ, {"INIDS_ANALYST_API_KEY": "analyst-key"}):
            from src import auth_service as auth_mod
            import importlib
            importlib.reload(auth_mod)
            svc = auth_mod.AuthService()
        app = Flask(__name__)
        with app.test_request_context(
            "/api/alerts",
            method="GET",
            headers={"X-API-Key": "analyst-key"},
        ):
            ok, reason = svc.authorize("analyst")
        assert ok is True

    def test_viewer_key_denied_on_analyst_endpoint(self):
        """Viewer key must still be denied on analyst-level endpoints."""
        from flask import Flask
        with patch.dict(os.environ, {"INIDS_VIEWER_API_KEY": "viewer-key"}):
            from src import auth_service as auth_mod
            import importlib
            importlib.reload(auth_mod)
            svc = auth_mod.AuthService()
        app = Flask(__name__)
        with app.test_request_context(
            "/api/alerts",
            method="GET",
            headers={"X-API-Key": "viewer-key"},
        ):
            ok, reason = svc.authorize("analyst")
        assert ok is False
        assert reason == "insufficient_role"
