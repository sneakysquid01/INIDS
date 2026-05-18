"""
FIX-005 regression: /events namespace requires a valid JWT to connect.
"""
import pytest
from unittest.mock import MagicMock, patch
from web_app.app import app, socketio


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    return socketio.test_client(app, namespace="/events")


@pytest.fixture()
def flask_client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_valid_ctx():
    from src.auth.models import AuthContext
    return AuthContext(user_id="u1", username="test-user", roles=["viewer"])


class TestWSAuthGating:
    def test_connect_without_token_rejected(self):
        """No auth dict → server must return False (reject)."""
        with app.test_request_context():
            from web_app.app import handle_events_connect
            result = handle_events_connect(auth=None)
        assert result is False

    def test_connect_with_empty_auth_dict_rejected(self):
        with app.test_request_context():
            from web_app.app import handle_events_connect
            result = handle_events_connect(auth={})
        assert result is False

    def test_connect_with_invalid_token_rejected(self):
        """Malformed / tampered JWT → rejected."""
        with app.test_request_context():
            from web_app.app import handle_events_connect
            with patch("web_app.app.UnifiedAuthService") as MockAS:
                MockAS.return_value.authenticate_jwt.return_value = None
                result = handle_events_connect(auth={"token": "bad.token.here"})
        assert result is False

    def test_connect_with_valid_token_accepted(self):
        """Valid JWT verified by UnifiedAuthService → accepted (returns None / truthy)."""
        ctx = _make_valid_ctx()
        with app.test_request_context():
            from web_app.app import handle_events_connect
            with patch("web_app.app.UnifiedAuthService") as MockAS, \
                 patch("web_app.app._start_module_update_broadcaster"), \
                 patch("web_app.app.emit"), \
                 patch("web_app.app._build_realtime_state", return_value={}):
                MockAS.return_value.authenticate_jwt.return_value = ctx
                result = handle_events_connect(auth={"token": "valid.jwt.token"})
        assert result is not False

    def test_connect_auth_ctx_stored_in_environ(self):
        """On success, auth context written to request.environ."""
        ctx = _make_valid_ctx()
        with app.test_request_context() as rc:
            from web_app.app import handle_events_connect
            with patch("web_app.app.UnifiedAuthService") as MockAS, \
                 patch("web_app.app._start_module_update_broadcaster"), \
                 patch("web_app.app.emit"), \
                 patch("web_app.app._build_realtime_state", return_value={}):
                MockAS.return_value.authenticate_jwt.return_value = ctx
                handle_events_connect(auth={"token": "valid.jwt.token"})
            from flask import request as _req
            assert _req.environ.get("inids_auth_ctx") is ctx

    def test_connect_auth_exception_rejected(self):
        """Exception in UnifiedAuthService → rejected, not 500."""
        with app.test_request_context():
            from web_app.app import handle_events_connect
            with patch("web_app.app.UnifiedAuthService") as MockAS:
                MockAS.return_value.authenticate_jwt.side_effect = RuntimeError("db down")
                result = handle_events_connect(auth={"token": "some.jwt"})
        assert result is False

    def test_bearer_header_fallback(self):
        """Token supplied via Authorization: Bearer header (not auth dict) should also work."""
        ctx = _make_valid_ctx()
        env = {"HTTP_AUTHORIZATION": "Bearer valid.jwt.token"}
        with app.test_request_context(environ_base=env):
            from web_app.app import handle_events_connect
            with patch("web_app.app.UnifiedAuthService") as MockAS, \
                 patch("web_app.app._start_module_update_broadcaster"), \
                 patch("web_app.app.emit"), \
                 patch("web_app.app._build_realtime_state", return_value={}):
                MockAS.return_value.authenticate_jwt.return_value = ctx
                result = handle_events_connect(auth=None)
        assert result is not False

    def test_socket_manager_js_passes_auth_token(self):
        """socket-manager.js must pass auth: { token } in io() call."""
        sm = (
            (
                __import__("pathlib").Path(__file__).parent.parent
                / "web_app" / "static" / "js" / "core" / "socket-manager.js"
            ).read_text(encoding="utf-8")
        )
        assert "auth:" in sm and "token" in sm, \
            "socket-manager.js must pass auth:{token} to io()"

    def test_socket_manager_get_token_reads_global_state(self):
        """_getToken() in socket-manager.js reads GlobalState data."""
        sm = (
            __import__("pathlib").Path(__file__).parent.parent
            / "web_app" / "static" / "js" / "core" / "socket-manager.js"
        ).read_text(encoding="utf-8")
        assert "auth.token" in sm
