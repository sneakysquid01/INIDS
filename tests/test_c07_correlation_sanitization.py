"""C-07 security regression tests — Correlation ID Sanitization + Audit Timestamps (Step 22).

Validation checkpoints per PLAN.md C-07 spec:
1.  sanitize_correlation_id(None) returns a server-generated UUID
2.  sanitize_correlation_id('') returns a server-generated UUID
3.  sanitize_correlation_id with \\n in value → generate new ID (log injection rejected)
4.  sanitize_correlation_id with \\r in value → generate new ID (log injection rejected)
5.  sanitize_correlation_id with null byte in value → generate new ID (log injection rejected)
6.  sanitize_correlation_id strips non-printable control characters
7.  sanitize_correlation_id truncates to 64 characters
8.  sanitize_correlation_id passes clean short values through unchanged
9.  sanitize_correlation_id with only non-printable chars returns generated ID
10. correlation_id_middleware uses sanitized value from X-Correlation-ID header
11. correlation_id_middleware uses sanitized value from X-Request-ID header fallback
12. correlation_id_middleware generates ID when no header present
13. get_correlation_id() sanitizes header values before using them
14. AuditLogMiddleware.after_request uses timezone-aware timestamp (no utcnow)
15. AuditLogMiddleware.get_user_activity uses timezone-aware cutoff
16. RBACManager AuditLog.id uses uuid.uuid4().hex (no user input in PK)
17. Security regression: newline in header does not appear in response or logs
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import timezone
from unittest.mock import MagicMock, patch

import pytest

from src.correlation_tracing import (
    CORRELATION_ID_HEADER,
    REQUEST_ID_HEADER,
    generate_correlation_id,
    sanitize_correlation_id,
)


# ---------------------------------------------------------------------------
# 1–3 helpers
# ---------------------------------------------------------------------------

_UUID_PATTERN = re.compile(r'^req_[0-9a-f]{16}$')


def _is_generated(value: str) -> bool:
    return bool(_UUID_PATTERN.match(value))


# ---------------------------------------------------------------------------
# 1–2: None / empty input → generated ID
# ---------------------------------------------------------------------------

def test_sanitize_none_returns_generated():
    result = sanitize_correlation_id(None)
    assert _is_generated(result), f"Expected generated ID, got {result!r}"


def test_sanitize_empty_string_returns_generated():
    result = sanitize_correlation_id("")
    assert _is_generated(result)


# ---------------------------------------------------------------------------
# 3–5: Log injection characters → reject entire value
# ---------------------------------------------------------------------------

def test_sanitize_rejects_newline():
    result = sanitize_correlation_id("valid-id\nX-Header: injected")
    assert _is_generated(result), "Newline must cause ID rejection"


def test_sanitize_rejects_carriage_return():
    result = sanitize_correlation_id("valid-id\rX-Header: injected")
    assert _is_generated(result), "CR must cause ID rejection"


def test_sanitize_rejects_null_byte():
    result = sanitize_correlation_id("valid-id\x00payload")
    assert _is_generated(result), "Null byte must cause ID rejection"


def test_sanitize_rejects_logs_warning_on_injection(caplog):
    with caplog.at_level(logging.WARNING, logger="src.correlation_tracing"):
        sanitize_correlation_id("id\ninjection")
    assert any("log_injection" in r.message or "rejected" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 6: Non-printable characters stripped (but not full rejection)
# ---------------------------------------------------------------------------

def test_sanitize_strips_non_printable_control_chars():
    # \x01 is non-printable but not in the rejection set → should be stripped
    result = sanitize_correlation_id("abc\x01def")
    assert result == "abcdef"


def test_sanitize_strips_bell_char():
    result = sanitize_correlation_id("req\x07-id")
    assert result == "req-id"


# ---------------------------------------------------------------------------
# 7: Truncation to 64 characters
# ---------------------------------------------------------------------------

def test_sanitize_truncates_to_64_chars():
    long_id = "a" * 100
    result = sanitize_correlation_id(long_id)
    assert len(result) == 64


def test_sanitize_preserves_exactly_64_chars():
    exact = "b" * 64
    result = sanitize_correlation_id(exact)
    assert result == exact


def test_sanitize_preserves_shorter_than_64():
    short = "my-request-1234"
    result = sanitize_correlation_id(short)
    assert result == short


# ---------------------------------------------------------------------------
# 8: Clean values pass through unchanged
# ---------------------------------------------------------------------------

def test_sanitize_clean_uuid_passes_through():
    clean = str(uuid.uuid4())
    result = sanitize_correlation_id(clean)
    assert result == clean


def test_sanitize_alphanumeric_passes_through():
    result = sanitize_correlation_id("req_abc123DEF456")
    assert result == "req_abc123DEF456"


# ---------------------------------------------------------------------------
# 9: All non-printable → returns generated ID
# ---------------------------------------------------------------------------

def test_sanitize_all_non_printable_returns_generated():
    # \x01\x02\x03 are all non-printable but not in rejection set → stripped → empty → generated
    result = sanitize_correlation_id("\x01\x02\x03")
    assert _is_generated(result)


# ---------------------------------------------------------------------------
# 10–13: Middleware integration via Flask test client
# ---------------------------------------------------------------------------

def _build_flask_app():
    from flask import Flask, jsonify
    from src.correlation_tracing import correlation_id_middleware, get_correlation_id

    app = Flask(__name__)
    correlation_id_middleware(app)

    @app.route("/ping")
    def ping():
        return jsonify({"cid": get_correlation_id()})

    return app


def test_middleware_uses_sanitized_correlation_id_from_header():
    app = _build_flask_app()
    with app.test_client() as client:
        resp = client.get("/ping", headers={CORRELATION_ID_HEADER: "my-clean-id"})
        data = resp.get_json()
        assert data["cid"] == "my-clean-id"


def test_middleware_rejects_injection_in_header():
    # Use environ_base to bypass Werkzeug test client header validation —
    # this simulates a raw WSGI server that does not strip newlines before us.
    app = _build_flask_app()
    with app.test_request_context(
        "/ping",
        environ_base={"HTTP_X_CORRELATION_ID": "id\nEvil: header"},
    ):
        from flask import g as flask_g
        app.preprocess_request()
        result_cid = flask_g.correlation_id
    assert _is_generated(result_cid), f"Expected generated ID, got {result_cid!r}"


def test_middleware_uses_request_id_fallback():
    app = _build_flask_app()
    with app.test_client() as client:
        resp = client.get("/ping", headers={REQUEST_ID_HEADER: "fallback-id-42"})
        data = resp.get_json()
        assert data["cid"] == "fallback-id-42"


def test_middleware_generates_id_when_no_header():
    app = _build_flask_app()
    with app.test_client() as client:
        resp = client.get("/ping")
        data = resp.get_json()
        assert _is_generated(data["cid"])


# ---------------------------------------------------------------------------
# 14: AuditLogMiddleware uses timezone-aware timestamps
# ---------------------------------------------------------------------------

def test_audit_log_timestamp_is_timezone_aware():
    """AuditLogEntry timestamp must be UTC-aware (no naive datetime)."""
    from src.middleware import AuditLogMiddleware
    from flask import Flask

    app = Flask(__name__)
    audit = AuditLogMiddleware()

    with app.test_request_context("/test"):
        from flask import request as flask_request
        flask_request.start_time = __import__("time").time()

        from flask import Response
        resp = Response(response=b"{}", status=200, mimetype="application/json")
        audit.after_request(resp)

    logs = audit.get_logs(limit=1)
    assert logs, "Expected at least one audit log entry"
    ts = logs[0]["timestamp"]
    # Timezone-aware ISO format includes '+' or 'Z' offset
    assert "+" in ts or ts.endswith("Z"), \
        f"Timestamp must be timezone-aware, got {ts!r}"


def test_audit_log_no_utcnow_in_source():
    """Static check: AuditLogMiddleware must not call datetime.utcnow()."""
    import inspect
    from src.middleware import AuditLogMiddleware
    source = inspect.getsource(AuditLogMiddleware)
    assert "utcnow" not in source, \
        "AuditLogMiddleware must not use deprecated datetime.utcnow()"


# ---------------------------------------------------------------------------
# 15: get_user_activity uses timezone-aware cutoff (no mixing naive/aware)
# ---------------------------------------------------------------------------

def test_get_user_activity_uses_aware_cutoff():
    """get_user_activity must not raise when comparing aware vs naive datetimes."""
    from src.middleware import AuditLogMiddleware
    from flask import Flask

    app = Flask(__name__)
    audit = AuditLogMiddleware()

    with app.test_request_context("/test"):
        from flask import request as flask_request
        flask_request.start_time = __import__("time").time()
        from flask import Response
        resp = Response(response=b"{}", status=200, mimetype="application/json")
        audit.after_request(resp)

    # Should not raise TypeError about mixing aware/naive datetimes
    result = audit.get_user_activity("anonymous", hours=1)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 16: rbac_manager AuditLog.id uses uuid (no user input in PK)
# ---------------------------------------------------------------------------

def test_rbac_audit_log_id_is_uuid_hex():
    """AuditLog.id must be uuid.uuid4().hex — no user-controlled data in PK."""
    import inspect
    from src.rbac_manager import RBACManager
    source = inspect.getsource(RBACManager.check_permission)
    assert "uuid.uuid4().hex" in source, \
        "AuditLog.id must use uuid.uuid4().hex to avoid user-input in primary key"
    assert "user_id" not in source.split("id=")[1].split("\n")[0], \
        "AuditLog.id must not include user_id in primary key"


# ---------------------------------------------------------------------------
# 17: Security regression — injection char cannot survive end-to-end
# ---------------------------------------------------------------------------

def test_injection_cannot_appear_in_response_header():
    # Simulate raw WSGI environ with CRLF injection attempt — sanitizer must
    # catch this before the value ever reaches a response header.
    app = _build_flask_app()
    with app.test_request_context(
        "/ping",
        environ_base={"HTTP_X_CORRELATION_ID": "x\r\nX-Injected: evil"},
    ):
        from flask import g as flask_g
        app.preprocess_request()
        result_cid = flask_g.correlation_id
    assert "\r" not in result_cid
    assert "\n" not in result_cid
    assert "evil" not in result_cid
    assert _is_generated(result_cid)
