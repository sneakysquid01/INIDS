from flask import Flask

import src.auth_service as auth_module


def test_auth_disabled_allows_access(monkeypatch):
    monkeypatch.setattr(auth_module._auth_service, "principals", {})
    ok, reason = auth_module._auth_service.authorize("admin")
    assert ok is True
    assert reason == "auth_disabled"


def test_auth_enabled_enforces_role(monkeypatch):
    monkeypatch.setattr(
        auth_module._auth_service,
        "principals",
        {
            "viewer-token": auth_module.Principal(role="viewer", token="viewer-token"),
            "admin-token": auth_module.Principal(role="admin", token="admin-token"),
        },
    )

    app = Flask(__name__)

    with app.test_request_context(headers={"X-API-Key": "viewer-token"}):
        ok, reason = auth_module._auth_service.authorize("admin")
        assert ok is False
        assert reason == "insufficient_role"

    with app.test_request_context(headers={"X-API-Key": "admin-token"}):
        ok, reason = auth_module._auth_service.authorize("admin")
        assert ok is True
        assert reason == "admin"
