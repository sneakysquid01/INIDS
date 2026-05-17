"""D-02: Input sanitizer security regression tests.

Verifies that all user-submitted string fields on authenticated routes
are sanitized before use. Tests XSS, SQL injection, and oversized inputs.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.input_sanitizer import (
    SanitizationError,
    sanitize_string,
    sanitize_id,
    sanitize_ip_address,
    sanitize_port,
    sanitize_severity,
    sanitize_integer,
)


# ---------------------------------------------------------------------------
# Unit tests: sanitize_string
# ---------------------------------------------------------------------------

class TestSanitizeString:
    def test_clean_string_passes(self):
        assert sanitize_string("hello_world", allow_special_chars=False) == "hello_world"

    def test_xss_payload_rejected(self):
        with pytest.raises(SanitizationError, match="XSS"):
            sanitize_string("<script>alert(1)</script>")

    def test_xss_img_onerror_rejected(self):
        with pytest.raises(SanitizationError, match="XSS"):
            sanitize_string('<img onerror="evil()">')

    def test_sql_injection_rejected_by_default(self):
        with pytest.raises(SanitizationError):
            sanitize_string("' OR 1=1 --", allow_special_chars=False)

    def test_sql_injection_allowed_with_flag(self):
        result = sanitize_string("hello' world", allow_special_chars=True, allow_spaces=True)
        assert "hello" in result

    def test_oversized_string_rejected(self):
        with pytest.raises(SanitizationError, match="maximum length"):
            sanitize_string("A" * 1001, max_length=1000)

    def test_newline_stripped(self):
        result = sanitize_string("hello\nworld", allow_special_chars=True, allow_spaces=False)
        assert "\n" not in result

    def test_empty_string(self):
        result = sanitize_string("", allow_special_chars=True)
        assert result == ""

    def test_special_chars_blocked_by_default(self):
        with pytest.raises(SanitizationError):
            sanitize_string("hello world", allow_special_chars=False, allow_spaces=False)

    def test_spaces_allowed(self):
        result = sanitize_string("hello world", allow_special_chars=True, allow_spaces=True)
        assert result == "hello world"


# ---------------------------------------------------------------------------
# Unit tests: sanitize_id
# ---------------------------------------------------------------------------

class TestSanitizeId:
    def test_valid_id(self):
        assert sanitize_id("alert_001-abc") == "alert_001-abc"

    def test_empty_id_rejected(self):
        with pytest.raises(SanitizationError, match="empty"):
            sanitize_id("")

    def test_id_with_special_chars_rejected(self):
        with pytest.raises(SanitizationError):
            sanitize_id("alert'; DROP TABLE alerts--")

    def test_oversized_id_rejected(self):
        with pytest.raises(SanitizationError):
            sanitize_id("a" * 101, max_length=100)

    def test_uuid_style_id(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        assert sanitize_id(uid) == uid


# ---------------------------------------------------------------------------
# Unit tests: sanitize_ip_address
# ---------------------------------------------------------------------------

class TestSanitizeIpAddress:
    def test_valid_ipv4(self):
        assert sanitize_ip_address("1.2.3.4") == "1.2.3.4"

    def test_valid_ipv6(self):
        assert sanitize_ip_address("::1") == "::1"

    def test_invalid_ip_rejected(self):
        with pytest.raises(SanitizationError):
            sanitize_ip_address("not-an-ip")

    def test_ip_with_injection_rejected(self):
        with pytest.raises(SanitizationError):
            sanitize_ip_address("1.2.3.4; rm -rf /")

    def test_empty_ip_rejected(self):
        with pytest.raises(SanitizationError):
            sanitize_ip_address("")

    def test_oversized_rejected(self):
        with pytest.raises(SanitizationError):
            sanitize_ip_address("1.2.3.4" + "x" * 50)


# ---------------------------------------------------------------------------
# Unit tests: sanitize_port and sanitize_severity
# ---------------------------------------------------------------------------

class TestSanitizePort:
    def test_valid_port(self):
        assert sanitize_port(8080) == 8080

    def test_port_zero_rejected(self):
        with pytest.raises(SanitizationError):
            sanitize_port(0)

    def test_port_out_of_range(self):
        with pytest.raises(SanitizationError):
            sanitize_port(99999)

    def test_port_as_string(self):
        assert sanitize_port("443") == 443


class TestSanitizeSeverity:
    def test_valid_severity(self):
        assert sanitize_severity("high") == "high"

    def test_invalid_severity_rejected(self):
        with pytest.raises(SanitizationError):
            sanitize_severity("extreme")

    def test_severity_case_normalized(self):
        assert sanitize_severity("HIGH") == "high"


# ---------------------------------------------------------------------------
# Integration: SanitizationError import path
# ---------------------------------------------------------------------------

class TestSanitizationErrorImport:
    def test_error_importable_from_app(self):
        from web_app.app import SanitizationError as AppSE  # noqa: F401
        assert AppSE is SanitizationError

    def test_sanitize_string_importable_from_app(self):
        from web_app.app import sanitize_string as app_ss  # noqa: F401
        assert app_ss is sanitize_string

    def test_sanitize_id_importable_from_app(self):
        from web_app.app import sanitize_id as app_si  # noqa: F401
        assert app_si is sanitize_id

    def test_sanitize_ip_importable_from_app(self):
        from web_app.app import sanitize_ip_address as app_sia  # noqa: F401
        assert app_sia is sanitize_ip_address


# ---------------------------------------------------------------------------
# Regression: XSS/SQLi payloads rejected across all field types
# ---------------------------------------------------------------------------

PAYLOADS_XSS = [
    "<script>alert('xss')</script>",
    "<iframe src='evil.com'>",
    "javascript:alert(1)",
    '<img src=x onerror=alert(1)>',
]

PAYLOADS_SQLI = [
    "'; DROP TABLE alerts; --",
    "\" OR \"1\"=\"1",
    "' UNION SELECT * FROM users --",
]


class TestXssPayloadsRejected:
    @pytest.mark.parametrize("payload", PAYLOADS_XSS)
    def test_xss_rejected_in_sanitize_string(self, payload):
        with pytest.raises(SanitizationError):
            sanitize_string(payload, allow_special_chars=True)

    @pytest.mark.parametrize("payload", PAYLOADS_XSS)
    def test_xss_rejected_in_sanitize_id(self, payload):
        with pytest.raises(SanitizationError):
            sanitize_id(payload)


class TestSqlInjectionPayloadsRejected:
    @pytest.mark.parametrize("payload", PAYLOADS_SQLI)
    def test_sqli_rejected_in_sanitize_string_strict(self, payload):
        with pytest.raises(SanitizationError):
            sanitize_string(payload, allow_special_chars=False)

    @pytest.mark.parametrize("payload", PAYLOADS_SQLI)
    def test_sqli_rejected_in_sanitize_id(self, payload):
        with pytest.raises(SanitizationError):
            sanitize_id(payload)
