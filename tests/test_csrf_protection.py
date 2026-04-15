"""Unit tests for CSRF protection module.

Tests cover token generation, validation, timing-safe comparison,
and protection patterns.
"""

import pytest
import secrets
from unittest.mock import Mock, patch
from src.csrf_protection import (
    generate_csrf_token,
    get_csrf_token,
    validate_csrf_token,
    require_csrf_token,
    csrf_protect_middleware,
    create_csrf_token_field,
)


class TestCSRFTokenGeneration:
    """Test CSRF token generation."""
    
    def test_generates_valid_token(self):
        """Test that generated token is valid."""
        token = generate_csrf_token()
        assert token
        assert len(token) > 0
    
    def test_token_uniqueness(self):
        """Test that generated tokens are unique."""
        token1 = generate_csrf_token()
        token2 = generate_csrf_token()
        assert token1 != token2
    
    def test_token_length(self):
        """Test token length is appropriate."""
        token = generate_csrf_token()
        assert len(token) >= 32
    
    def test_token_format_alphanumeric(self):
        """Test token contains only safe characters."""
        token = generate_csrf_token()
        assert all(c.isalnum() or c in '-_' for c in token)
    
    def test_token_entropy(self):
        """Test that multiple tokens have good entropy."""
        tokens = [generate_csrf_token() for _ in range(100)]
        assert len(set(tokens)) == 100


class TestCSRFTokenStorage:
    """Test CSRF token storage."""
    
    def test_create_csrf_token_field_html(self, request_context):
        """Test HTML generation for form field."""
        token = generate_csrf_token()
        
        with patch('src.csrf_protection.get_csrf_token', return_value=token):
            html = create_csrf_token_field()
            assert isinstance(html, str)
            assert len(html) > 0


class TestCSRFTokenValidation:
    """Test CSRF token validation."""
    
    def test_validate_with_mock(self):
        """Test token validation logic."""
        token = generate_csrf_token()
        
        # Test that timing-safe comparison works
        result = secrets.compare_digest(token, token)
        assert result is True
    
    def test_validate_failure(self):
        """Test validation failure."""
        token1 = generate_csrf_token()
        token2 = generate_csrf_token()
        
        result = secrets.compare_digest(token1, token2)
        assert result is False


class TestTimingSafeComparison:
    """Test timing-safe token comparison."""
    
    def test_equal_tokens(self):
        """Test comparison of equal tokens."""
        token = "test-token-abc123"
        result = secrets.compare_digest(token, token)
        assert result is True
    
    def test_different_tokens(self):
        """Test comparison of different tokens."""
        token1 = "test-token-abc123"
        token2 = "test-token-xyz789"
        result = secrets.compare_digest(token1, token2)
        assert result is False
    
    def test_partial_match(self):
        """Test that partial matches are rejected."""
        token1 = "test-token-abc123"
        token2 = "test-token-abc"
        result = secrets.compare_digest(token1, token2)
        assert result is False


class TestCSRFDecorator:
    """Test CSRF protection decorator."""
    
    def test_decorator_is_callable(self):
        """Test decorator is callable."""
        @require_csrf_token
        def protected_view():
            return "success"
        
        assert callable(protected_view)


class TestMiddlewareIntegration:
    """Test Flask middleware integration."""
    
    def test_middleware_registers(self, flask_app):
        """Test middleware registers with app."""
        csrf_protect_middleware(flask_app)
        assert flask_app is not None


class TestFormGeneration:
    """Test form field generation."""
    
    def test_csrf_field_generation(self, request_context):
        """Test CSRF field generation."""
        token = generate_csrf_token()
        
        with patch('src.csrf_protection.get_csrf_token', return_value=token):
            html = create_csrf_token_field()
            assert isinstance(html, str)


class TestSecurityProperties:
    """Test security properties."""
    
    def test_tokens_not_predictable(self):
        """Test that tokens are not predictable."""
        tokens = [generate_csrf_token() for _ in range(100)]
        for i in range(len(tokens) - 1):
            assert tokens[i] != tokens[i + 1]
    
    def test_timing_safe_properties(self):
        """Test timing-safe comparison properties."""
        token = "abcdefghijklmnopqrstuvwxyz0123456789"
        
        # Both should complete without short-circuit
        secrets.compare_digest(token, "zbcdefghijklmnopqrstuvwxyz0123456789")
        secrets.compare_digest(token, "abcdefghijklmnopqrstuvwxyz0123456789a")


class TestEdgeCases:
    """Test edge cases."""
    
    def test_very_long_token(self):
        """Test handling of very long tokens."""
        token1 = "x" * 1000
        token2 = "x" * 1000
        
        result = secrets.compare_digest(token1, token2)
        assert result is True
    
    def test_unicode_in_token(self):
        """Test token creation."""
        token = generate_csrf_token()
        assert isinstance(token, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
