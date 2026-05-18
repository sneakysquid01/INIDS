"""FIX-021 regression: CSRF middleware must short-circuit for /api/* paths."""
import inspect
import pytest
from flask import Flask


def test_csrf_middleware_skips_api_paths_in_source():
    """csrf_protect_middleware must contain the /api/ short-circuit."""
    from src.csrf_protection import csrf_protect_middleware
    src = inspect.getsource(csrf_protect_middleware)
    assert "startswith('/api/')" in src or 'startswith("/api/")' in src, (
        "csrf_protect_middleware must skip /api/* paths"
    )


def test_csrf_middleware_returns_none_for_api_before_request():
    """before_request inside csrf_protect_middleware must return None for /api/ paths."""
    from src.csrf_protection import csrf_protect_middleware
    app = Flask("test_csrf_api")
    app.secret_key = "test-secret"
    csrf_protect_middleware(app)

    @app.route("/api/test", methods=["POST"])
    def api_test():
        return "ok"

    with app.test_client() as c:
        resp = c.post("/api/test")
        # Must NOT be a 403 CSRF failure; route returns 200
        assert resp.status_code == 200


def test_csrf_validation_not_triggered_on_api_post():
    """POSTing to /api/* without a CSRF token must not produce a 403."""
    from src.csrf_protection import csrf_protect_middleware
    app = Flask("test_csrf_api2")
    app.secret_key = "test-secret"
    csrf_protect_middleware(app)

    @app.route("/api/data", methods=["POST"])
    def api_data():
        return {"status": "ok"}, 200

    with app.test_client() as c:
        resp = c.post("/api/data", json={"key": "value"})
        assert resp.status_code == 200, (
            f"Expected 200 on /api/* POST without CSRF token, got {resp.status_code}"
        )


def test_non_api_path_still_generates_token():
    """Non-API paths must still get a CSRF token in the session."""
    from src.csrf_protection import csrf_protect_middleware, CSRF_TOKEN_SESSION_KEY
    app = Flask("test_csrf_non_api")
    app.secret_key = "test-secret"
    csrf_protect_middleware(app)

    @app.route("/dashboard")
    def dashboard():
        from flask import session
        return session.get(CSRF_TOKEN_SESSION_KEY, "missing")

    with app.test_client() as c:
        resp = c.get("/dashboard")
        assert resp.data.decode() != "missing", (
            "Non-API GET must set a CSRF token in the session"
        )


def test_csrf_token_injected_in_template_context():
    """context_processor must inject csrf_token into templates."""
    from src.csrf_protection import csrf_protect_middleware
    app = Flask("test_csrf_ctx")
    app.secret_key = "test-secret"
    csrf_protect_middleware(app)

    @app.route("/form")
    def form():
        from flask import render_template_string
        return render_template_string("{{ csrf_token }}")

    with app.test_client() as c:
        resp = c.get("/form")
        assert len(resp.data) > 10, "csrf_token must be injected via context_processor"
