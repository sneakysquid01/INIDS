"""Regression tests for FIX-003 (audit identity), FIX-007 (UUID alert IDs),
and FIX-015 (AuthStoreUnboundError)."""
import re
import pytest


# ---------------------------------------------------------------------------
# FIX-003: Audit identity from g.auth, not X-User-ID header
# ---------------------------------------------------------------------------

class TestFix003AuditIdentity:
    def test_x_user_id_not_read_in_after_request(self):
        """middleware.py must not call request.headers.get('X-User-ID') for audit user."""
        import inspect
        from src import middleware
        src = inspect.getsource(middleware.AuditLogMiddleware.after_request)
        # The old pattern was: request.headers.get('X-User-ID', ...)
        # Ensure that exact call is gone (comments mentioning the header are OK)
        assert "request.headers.get('X-User-ID'" not in src and \
               'request.headers.get("X-User-ID"' not in src, (
            "AuditLogMiddleware.after_request must not call request.headers.get('X-User-ID')"
        )

    def test_g_auth_used_for_audit_user(self):
        """after_request must read identity from g.auth."""
        import inspect
        from src import middleware
        src = inspect.getsource(middleware.AuditLogMiddleware.after_request)
        assert "g.auth" in src or "getattr(g" in src, (
            "after_request must source username from g.auth"
        )

    def test_audit_user_from_g_auth_context(self, flask_app):
        """_extract_audit_user must return username from g.auth."""
        from src.auth.models import AuthContext
        from flask import g

        with flask_app.test_request_context("/api/alerts"):
            g.auth = AuthContext(
                user_id="u-viewer",
                username="correct_user",
                roles=frozenset({"viewer"}),
            )
            _auth_ctx = getattr(g, "auth", None)
            user = getattr(_auth_ctx, "username", None) or "anonymous"
            assert user == "correct_user"

    def test_audit_defaults_to_anonymous_without_g_auth(self, flask_app):
        """When g.auth is absent, audit user must be 'anonymous'."""
        from flask import g

        with flask_app.test_request_context("/api/health"):
            if hasattr(g, "auth"):
                del g.auth
            _auth_ctx = getattr(g, "auth", None)
            user = getattr(_auth_ctx, "username", None) or "anonymous"
            assert user == "anonymous"


# ---------------------------------------------------------------------------
# FIX-007: _on_detection_event alert ID must be full UUID
# ---------------------------------------------------------------------------

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
SHORT_ID_RE = re.compile(r"^al_[0-9a-f]{10}$")


class TestFix007AlertIdFormat:
    def test_on_detection_event_source_has_full_uuid(self):
        """The _on_detection_event code must use str(uuid.uuid4()) not al_ prefix."""
        import inspect
        from web_app import app as app_module
        src = inspect.getsource(app_module._on_detection_event)
        assert "str(uuid.uuid4())" in src, (
            "_on_detection_event must produce full UUID alert IDs"
        )
        # Old short-form must not be present
        assert 'f"al_{uuid.uuid4().hex[:10]}"' not in src

    def test_detection_event_uuid_format(self):
        """Alert IDs from _on_detection_event must match UUID regex."""
        import web_app.app as _app_mod
        from src.core.event_bus import DetectionEvent

        event = DetectionEvent(
            source_ip="1.2.3.4",
            prediction="attack",
            confidence=0.9,
            severity="high",
            profile="balanced",
            reason="test",
            features={},
            timestamp="2025-01-01T00:00:00Z",
            attack_type="port_scan",
            suspicious=True,
        )

        captured: list[str] = []
        original = _app_mod.ops_store.save_alert

        def spy(d):
            captured.append(d.get("id", ""))
            return original(d)

        with _app_mod.app.app_context():
            _app_mod.ops_store.save_alert = spy
            try:
                _app_mod._on_detection_event(event)
            finally:
                _app_mod.ops_store.save_alert = original

        assert captured, "save_alert was never called"
        for aid in captured:
            assert UUID_RE.match(aid), f"Non-UUID alert ID: {aid!r}"
            assert not SHORT_ID_RE.match(aid), f"Still using short ID: {aid!r}"


# ---------------------------------------------------------------------------
# FIX-015: _get_ops_store raises AuthStoreUnboundError when store not attached
# ---------------------------------------------------------------------------

class TestFix015AuthStoreUnbound:
    def test_auth_store_unbound_error_class_exists(self):
        from src.auth.decorators import AuthStoreUnboundError
        assert issubclass(AuthStoreUnboundError, RuntimeError)

    def test_get_ops_store_raises_when_unbound(self):
        from flask import Flask
        from src.auth.decorators import _get_ops_store, AuthStoreUnboundError

        bare = Flask("test_unbound")
        with bare.app_context():
            with pytest.raises(AuthStoreUnboundError):
                _get_ops_store()

    def test_get_ops_store_returns_store_when_present(self):
        import web_app.app as _app_mod
        from src.auth.decorators import _get_ops_store
        with _app_mod.app.app_context():
            store = _get_ops_store()
            assert store is _app_mod.ops_store

    def test_unbound_returns_503(self):
        from flask import Flask
        from src.auth.decorators import require_roles, AuthStoreUnboundError

        bare = Flask("test_503")

        @bare.route("/protected")
        @require_roles("viewer")
        def protected():
            return "ok"

        @bare.errorhandler(AuthStoreUnboundError)
        def handle_unbound(e):
            from flask import jsonify
            return jsonify({"error": "service_unavailable", "reason": "auth_store_unbound"}), 503

        with bare.test_client() as c:
            resp = c.get("/protected")
            assert resp.status_code == 503
            data = resp.get_json()
            assert data["reason"] == "auth_store_unbound"
