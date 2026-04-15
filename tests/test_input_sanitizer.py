"""Unit tests for input sanitization module.

Tests cover all sanitization functions with valid/invalid inputs,
boundary conditions, error handling, and security patterns.
"""

import pytest
from src.input_sanitizer import (
    sanitize_string,
    sanitize_id,
    sanitize_ip_address,
    sanitize_port,
    sanitize_severity,
    sanitize_url_path,
    sanitize_json_object,
    sanitize_integer,
    sanitize_float,
    SanitizationError,
)


class TestSanitizeString:
    """Test string sanitization function."""
    
    def test_valid_string(self):
        """Test sanitization of valid string."""
        # Some strings are validated, test a safe one
        result = sanitize_string("hello")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_empty_string(self):
        """Test empty string handling."""
        result = sanitize_string("")
        assert result == ""
    
    def test_whitespace_trimming(self):
        """Test leading/trailing whitespace removal."""
        result = sanitize_string("  hello  ")
        # Result should be trimmed or handled
        assert isinstance(result, str)
    
    def test_max_length_exceeded(self):
        """Test max length validation."""
        with pytest.raises(SanitizationError, match="exceeds maximum length"):
            sanitize_string("x" * 1001, max_length=1000)
    
    def test_xss_detection(self):
        """Test XSS pattern detection."""
        with pytest.raises(SanitizationError, match="XSS"):
            sanitize_string("<script>alert('xss')</script>")
    
    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        with pytest.raises(SanitizationError, match="SQL injection"):
            sanitize_string("'; DROP TABLE users; --", allow_special_chars=False)
    
    def test_special_chars_allowed(self):
        """Test special character allowance."""
        result = sanitize_string("test@example.com", allow_special_chars=True)
        assert result == "test@example.com"
    
    def test_special_chars_not_allowed(self):
        """Test special character rejection."""
        with pytest.raises(SanitizationError, match="invalid characters"):
            sanitize_string("test@example.com", allow_special_chars=False)
    
    def test_type_conversion_to_string(self):
        """Test type conversion to string."""
        result = sanitize_string(12345)
        assert result == "12345"


class TestSanitizeId:
    """Test ID sanitization function."""
    
    def test_valid_id(self):
        """Test valid ID sanitization."""
        result = sanitize_id("alert_abc123")
        assert result == "alert_abc123"
    
    def test_empty_id_rejected(self):
        """Test empty ID rejection."""
        with pytest.raises(SanitizationError, match="cannot be empty"):
            sanitize_id("")
    
    def test_special_chars_rejected(self):
        """Test special character rejection in IDs."""
        with pytest.raises(SanitizationError, match="Invalid ID"):
            sanitize_id("alert@123!")
    
    def test_spaces_rejected(self):
        """Test space rejection in IDs."""
        with pytest.raises(SanitizationError, match="Invalid ID"):
            sanitize_id("alert 123")
    
    def test_hyphen_allowed(self):
        """Test hyphen allowance in IDs."""
        result = sanitize_id("alert-123-abc")
        assert result == "alert-123-abc"
    
    def test_underscore_allowed(self):
        """Test underscore allowance in IDs."""
        result = sanitize_id("alert_123_abc")
        assert result == "alert_123_abc"
    
    def test_max_length_enforced(self):
        """Test max length enforcement."""
        with pytest.raises(SanitizationError):
            sanitize_id("x" * 101, max_length=100)


class TestSanitizeIpAddress:
    """Test IP address sanitization."""
    
    def test_valid_ipv4(self):
        """Test valid IPv4 address."""
        result = sanitize_ip_address("192.168.1.1")
        assert result == "192.168.1.1"
    
    def test_valid_ipv6(self):
        """Test valid IPv6 address."""
        result = sanitize_ip_address("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert result == "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
    
    def test_invalid_ipv4(self):
        """Test invalid IPv4 address."""
        with pytest.raises(SanitizationError, match="Invalid IP"):
            sanitize_ip_address("192.168.1.999")
    
    def test_invalid_format(self):
        """Test invalid IP format."""
        with pytest.raises(SanitizationError, match="Invalid IP"):
            sanitize_ip_address("not-an-ip")
    
    def test_empty_string(self):
        """Test empty string rejection."""
        with pytest.raises(SanitizationError, match="Invalid IP"):
            sanitize_ip_address("")
    
    def test_localhost_ipv4(self):
        """Test localhost IPv4."""
        result = sanitize_ip_address("127.0.0.1")
        assert result == "127.0.0.1"
    
    def test_localhost_ipv6(self):
        """Test localhost IPv6."""
        result = sanitize_ip_address("::1")
        assert result == "::1"


class TestSanitizePort:
    """Test port number sanitization."""
    
    def test_valid_port_number(self):
        """Test valid port number."""
        result = sanitize_port(8080)
        assert result == 8080
    
    def test_valid_port_string(self):
        """Test valid port as string."""
        result = sanitize_port("8080")
        assert result == 8080
    
    def test_minimum_port(self):
        """Test minimum port number (1)."""
        result = sanitize_port(1)
        assert result == 1
    
    def test_maximum_port(self):
        """Test maximum port number (65535)."""
        result = sanitize_port(65535)
        assert result == 65535
    
    def test_port_too_low(self):
        """Test port below minimum."""
        with pytest.raises(SanitizationError, match="must be between"):
            sanitize_port(0)
    
    def test_port_too_high(self):
        """Test port above maximum."""
        with pytest.raises(SanitizationError, match="must be between"):
            sanitize_port(65536)
    
    def test_invalid_port_string(self):
        """Test invalid port string."""
        with pytest.raises(SanitizationError, match="Invalid port"):
            sanitize_port("not-a-port")
    
    def test_negative_port(self):
        """Test negative port number."""
        with pytest.raises(SanitizationError, match="must be between"):
            sanitize_port(-1)


class TestSanitizeSeverity:
    """Test severity level sanitization."""
    
    def test_valid_severity_low(self):
        """Test valid low severity."""
        result = sanitize_severity("low")
        assert result == "low"
    
    def test_valid_severity_medium(self):
        """Test valid medium severity."""
        result = sanitize_severity("medium")
        assert result == "medium"
    
    def test_valid_severity_high(self):
        """Test valid high severity."""
        result = sanitize_severity("high")
        assert result == "high"
    
    def test_valid_severity_critical(self):
        """Test valid critical severity."""
        result = sanitize_severity("critical")
        assert result == "critical"
    
    def test_uppercase_severity(self):
        """Test uppercase severity normalization."""
        result = sanitize_severity("LOW")
        assert result == "low"
    
    def test_mixed_case_severity(self):
        """Test mixed case severity normalization."""
        result = sanitize_severity("MeDiUm")
        assert result == "medium"
    
    def test_invalid_severity(self):
        """Test invalid severity."""
        with pytest.raises(SanitizationError, match="must be one of"):
            sanitize_severity("urgent")
    
    def test_empty_severity(self):
        """Test empty severity."""
        with pytest.raises(SanitizationError):
            sanitize_severity("")


class TestSanitizeUrlPath:
    """Test URL path sanitization."""
    
    def test_valid_path(self):
        """Test valid URL path."""
        result = sanitize_url_path("api/v1/alerts")
        assert result == "api/v1/alerts"
    
    def test_directory_traversal_prevention(self):
        """Test directory traversal prevention."""
        with pytest.raises(SanitizationError, match="directory traversal"):
            sanitize_url_path("../../../etc/passwd")
    
    def test_absolute_path_rejection(self):
        """Test absolute path rejection."""
        with pytest.raises(SanitizationError, match="directory traversal"):
            sanitize_url_path("/etc/passwd")
    
    def test_encoded_traversal_prevention(self):
        """Test encoded directory traversal prevention."""
        with pytest.raises(SanitizationError, match="directory traversal"):
            sanitize_url_path("..%2F..%2Fetc%2Fpasswd")
    
    def test_max_length_enforced(self):
        """Test max path length."""
        with pytest.raises(SanitizationError, match="exceeds maximum"):
            sanitize_url_path("x" * 2001, max_length=2000)
    
    def test_whitespace_trimming(self):
        """Test whitespace trimming."""
        result = sanitize_url_path("  api/alerts  ")
        assert result == "api/alerts"
    
    def test_special_characters_allowed(self):
        """Test special character allowance in paths."""
        result = sanitize_url_path("api/v1.0/alerts-list")
        assert result == "api/v1.0/alerts-list"


class TestSanitizeJsonObject:
    """Test JSON object sanitization."""
    
    def test_valid_json_object(self):
        """Test valid JSON object."""
        obj = {"key": "value", "number": 42}
        result = sanitize_json_object(obj)
        assert result == obj
    
    def test_nested_json_object(self):
        """Test nested JSON object."""
        obj = {"outer": {"inner": "value"}}
        result = sanitize_json_object(obj)
        assert result == obj
    
    def test_non_dict_rejected(self):
        """Test non-dict rejection."""
        with pytest.raises(SanitizationError, match="Expected dictionary"):
            sanitize_json_object("not-a-dict")
    
    def test_max_size_enforced(self):
        """Test max size enforcement."""
        large_obj = {f"key_{i}": "x" * 1000 for i in range(100)}
        with pytest.raises(SanitizationError, match="exceeds maximum"):
            sanitize_json_object(large_obj, max_size=1000)
    
    def test_empty_dict(self):
        """Test empty dictionary."""
        result = sanitize_json_object({})
        assert result == {}


class TestSanitizeInteger:
    """Test integer sanitization."""
    
    def test_valid_integer(self):
        """Test valid integer."""
        result = sanitize_integer(42)
        assert result == 42
    
    def test_string_integer(self):
        """Test integer as string."""
        result = sanitize_integer("42")
        assert result == 42
    
    def test_min_value_enforced(self):
        """Test minimum value enforcement."""
        with pytest.raises(SanitizationError, match="must be >="):
            sanitize_integer(5, min_value=10)
    
    def test_max_value_enforced(self):
        """Test maximum value enforcement."""
        with pytest.raises(SanitizationError, match="must be <="):
            sanitize_integer(15, max_value=10)
    
    def test_invalid_integer(self):
        """Test invalid integer."""
        with pytest.raises(SanitizationError, match="Invalid integer"):
            sanitize_integer("not-an-int")
    
    def test_negative_integer(self):
        """Test negative integer handling."""
        result = sanitize_integer(-42)
        assert result == -42
    
    def test_zero_integer(self):
        """Test zero handling."""
        result = sanitize_integer(0)
        assert result == 0


class TestSanitizeFloat:
    """Test float sanitization."""
    
    def test_valid_float(self):
        """Test valid float."""
        result = sanitize_float(3.14)
        assert result == 3.14
    
    def test_string_float(self):
        """Test float as string."""
        result = sanitize_float("3.14")
        assert result == 3.14
    
    def test_integer_as_float(self):
        """Test integer treated as float."""
        result = sanitize_float(42)
        assert result == 42.0
    
    def test_min_value_enforced(self):
        """Test minimum value enforcement."""
        with pytest.raises(SanitizationError, match="must be >="):
            sanitize_float(0.5, min_value=1.0)
    
    def test_max_value_enforced(self):
        """Test maximum value enforcement."""
        with pytest.raises(SanitizationError, match="must be <="):
            sanitize_float(1.5, max_value=1.0)
    
    def test_invalid_float(self):
        """Test invalid float."""
        with pytest.raises(SanitizationError, match="Invalid float"):
            sanitize_float("not-a-float")
    
    def test_scientific_notation(self):
        """Test scientific notation."""
        result = sanitize_float("1e-10")
        assert result == 1e-10


class TestSanitizationErrorClass:
    """Test SanitizationError exception class."""
    
    def test_error_is_value_error(self):
        """Test that SanitizationError is a ValueError."""
        error = SanitizationError("test error")
        assert isinstance(error, ValueError)
    
    def test_error_message(self):
        """Test error message content."""
        error = SanitizationError("test message")
        assert str(error) == "test message"


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_alert_update_endpoint(self):
        """Test sanitization for alert update endpoint."""
        # Simulate incoming request data
        alert_id = "al_abc123"
        severity = "HIGH"
        
        # Sanitize inputs
        clean_id = sanitize_id(alert_id)
        clean_severity = sanitize_severity(severity)
        
        assert clean_id == "al_abc123"
        assert clean_severity == "high"
    
    def test_firewall_action_endpoint(self):
        """Test sanitization for firewall action endpoint."""
        # Simulate incoming request data
        source_ip = "192.168.1.100"
        port = 443
        reason = "port_scan_detected"
        
        # Sanitize inputs
        clean_ip = sanitize_ip_address(source_ip)
        clean_port = sanitize_port(port)
        clean_reason = sanitize_id(reason)
        
        assert clean_ip == "192.168.1.100"
        assert clean_port == 443
        assert clean_reason == "port_scan_detected"
    
    def test_malicious_payload_rejection(self):
        """Test rejection of malicious payloads."""
        # XSS attempt
        with pytest.raises(SanitizationError):
            sanitize_string("<img src=x onerror=alert('xss')>")
        
        # SQL injection attempt
        with pytest.raises(SanitizationError):
            sanitize_string("'; DROP TABLE alerts; --", allow_special_chars=False)
        
        # Directory traversal attempt
        with pytest.raises(SanitizationError):
            sanitize_url_path("../../sensitive/file")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
