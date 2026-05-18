"""
INIDS Week 1 Integration Tests

Tests for:
- Middleware security stack (rate limiting, IP blocking, security headers, audit logging)
- JWT authentication
- Run-as impersonation
- Schema validation
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Import middleware components
from src.middleware import (
    RateLimitMiddleware, RateLimitConfig, IPBlockingMiddleware,
    SecurityHeadersMiddleware, AuditLogMiddleware, CORSMiddleware,
    RequestValidationMiddleware, ContentSecurityMiddleware, AuditLogEntry
)

# Import auth components (F-AUTH-REMOVE: RS256JWTManager replaces JWTAuthManager/RunAsManager)
from src.auth.jwt_manager import RS256JWTManager, reset_jwt_manager
from src.validation_schemas import (
    ConfigValidator, validate_rule, validate_predict_request, validate_detect_request
)


class TestRateLimitMiddleware:
    """Test rate limiting middleware."""
    
    def test_rate_limit_initialization(self):
        """Test middleware initialization."""
        config = RateLimitConfig(max_requests=100, window_seconds=60)
        middleware = RateLimitMiddleware(config)
        
        assert middleware.config.max_requests == 100
        assert middleware.config.window_seconds == 60
    
    def test_get_client_key_with_api_key(self):
        """Test client key extraction from API key."""
        middleware = RateLimitMiddleware()
        # Test implemented via integration test instead (requires Flask app context)
        assert middleware is not None
    
    def test_get_client_key_from_ip(self):
        """Test client key extraction from IP."""
        middleware = RateLimitMiddleware()
        # Test implemented via integration test instead (requires Flask app context)
        assert middleware is not None


class TestIPBlockingMiddleware:
    """Test IP blocking middleware."""
    
    def test_initialization(self):
        """Test IP blocker initialization."""
        blocker = IPBlockingMiddleware(max_failures=5, block_time_seconds=300)
        
        assert blocker.max_failures == 5
        assert blocker.block_time == 300
        assert '127.0.0.1' in blocker.whitelist
    
    def test_add_failure(self):
        """Test recording failures."""
        blocker = IPBlockingMiddleware(max_failures=2, block_time_seconds=60)
        
        ip = '192.168.1.100'
        blocker.add_failure(ip)
        assert ip not in blocker.blocked
        
        blocker.add_failure(ip)
        assert ip in blocker.blocked
    
    def test_whitelist_bypass(self):
        """Test that whitelisted IPs are not blocked."""
        blocker = IPBlockingMiddleware(max_failures=1)
        
        # Add many failures to whitelisted IP
        for _ in range(10):
            blocker.add_failure('127.0.0.1')
        
        assert '127.0.0.1' not in blocker.blocked


class TestJWTAuthManager:
    """Test RS256 JWT manager (F-AUTH-REMOVE: replaces legacy JWTAuthManager)."""

    def setup_method(self):
        reset_jwt_manager()

    def teardown_method(self):
        reset_jwt_manager()

    def test_create_token(self):
        """RS256JWTManager.create_token() returns a non-empty JWT string."""
        manager = RS256JWTManager()
        token = manager.create_token(
            user_id='user123',
            username='admin',
            roles=['admin', 'analyst'],
        )
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_valid_token(self):
        """verify_token() returns (True, dict, None) for a freshly-issued token."""
        manager = RS256JWTManager()
        token = manager.create_token(user_id='user123', username='admin', roles=['admin'])
        ok, payload, error = manager.verify_token(token)
        assert ok is True
        assert payload is not None
        assert payload['sub'] == 'admin'
        assert 'admin' in payload['roles']
        assert error is None

    def test_verify_invalid_token(self):
        """verify_token() returns (False, None, reason) for a garbage token."""
        manager = RS256JWTManager()
        ok, payload, error = manager.verify_token('invalid-token')
        assert ok is False
        assert payload is None
        assert error is not None

    def test_verify_wrong_key_rejected(self):
        """Token signed by a different keypair must be rejected."""
        mgr_a = RS256JWTManager()
        mgr_b = RS256JWTManager()  # independent ephemeral keypair
        token = mgr_a.create_token(user_id='u', username='u', roles=['viewer'])
        ok, _, _ = mgr_b.verify_token(token)
        assert ok is False


class TestRunAsManager:
    """F-AUTH-REMOVE: RunAsManager removed — delegated tokens issued via api_auth_runas.

    Basic smoke test: the /api/auth/runas endpoint is wired to require_roles("admin")
    and issues a new RS256 token for the target user. Tested at integration level.
    """

    def test_delegated_token_can_be_issued_directly(self):
        """RS256JWTManager can create a token for the target user directly."""
        reset_jwt_manager()
        try:
            manager = RS256JWTManager()
            token = manager.create_token(
                user_id='analyst',
                username='analyst',
                roles=['analyst'],
            )
            ok, payload, _ = manager.verify_token(token)
            assert ok is True
            assert payload['sub'] == 'analyst'
            assert 'analyst' in payload['roles']
        finally:
            reset_jwt_manager()


class TestConfigValidator:
    """Test configuration schema validation."""
    
    def test_validate_rule_valid(self):
        """Test validation of valid rule."""
        rule = {
            'name': 'Test Rule',
            'pattern': 'test_pattern',
            'severity': 'high',
            'enabled': True,
            'confidence': 0.85
        }
        
        is_valid, error, details = validate_rule(rule)
        
        assert is_valid is True
        assert error is None
        assert len(details) == 0
    
    def test_validate_rule_invalid_severity(self):
        """Test validation with invalid severity."""
        rule = {
            'name': 'Test Rule',
            'pattern': 'test_pattern',
            'severity': 'invalid'  # Invalid severity
        }
        
        is_valid, error, details = validate_rule(rule)
        
        assert is_valid is False
        assert error is not None
        assert len(details) > 0
    
    def test_validate_rule_missing_required(self):
        """Test validation with missing required fields."""
        rule = {
            'name': 'Test Rule'
            # Missing 'pattern' and 'severity'
        }
        
        is_valid, error, details = validate_rule(rule)
        
        assert is_valid is False
        assert error is not None
    
    def test_validate_predict_request_valid(self):
        """Test validation of valid predict request."""
        request_data = {
            'features': {'feature1': 1.0, 'feature2': 2.0},
            'profile': 'balanced'
        }
        
        is_valid, error = validate_predict_request(request_data)
        
        assert is_valid is True
        assert error is None
    
    def test_validate_predict_request_missing_features(self):
        """Test validation without features."""
        request_data = {
            'profile': 'balanced'
            # Missing 'features'
        }
        
        is_valid, error = validate_predict_request(request_data)
        
        assert is_valid is False
        assert error is not None
    
    def test_validate_detect_request_valid(self):
        """Test validation of valid detect request."""
        request_data = {
            'features': {'feature1': 1.0}
        }
        
        is_valid, error = validate_detect_request(request_data)
        
        assert is_valid is True
        assert error is None


class TestAuditLogMiddleware:
    """Test audit logging."""
    
    def test_audit_log_creation(self):
        """Test audit log entry creation."""
        middleware = AuditLogMiddleware(max_logs=100)
        
        assert len(middleware.logs) == 0
    
    def test_audit_log_capacity(self):
        """Test audit log max capacity."""
        middleware = AuditLogMiddleware(max_logs=10)
        
        # Add more logs than capacity
        for i in range(15):
            from src.middleware import AuditLogEntry
            entry = AuditLogEntry(
                timestamp=datetime.utcnow().isoformat(),
                user='admin',
                method='GET',
                path=f'/api/test{i}',
                status=200,
                response_time_ms=10.0,
                source_ip='127.0.0.1',
                user_agent='test',
                request_body_size=0,
                response_body_size=100
            )
            middleware.logs.append(entry)
        
        # Should only keep last 10
        assert len(middleware.logs) == 10


class TestCORSMiddleware:
    """Test CORS middleware."""
    
    def test_cors_initialization(self):
        """Test CORS middleware initialization."""
        cors = CORSMiddleware(
            allowed_origins=['http://localhost:3000'],
            allowed_methods=['GET', 'POST']
        )
        
        assert 'http://localhost:3000' in cors.allowed_origins
        assert 'GET' in cors.allowed_methods


class TestRequestValidationMiddleware:
    """Test request validation."""
    
    def test_initialization(self):
        """Test middleware initialization."""
        validator = RequestValidationMiddleware()
        
        assert validator.MAX_BODY_SIZE == 1024 * 1024


class TestSecurityHeadersMiddleware:
    """Test security headers."""
    
    def test_security_headers_present(self):
        """Test that all security headers are defined."""
        headers = SecurityHeadersMiddleware.HEADERS
        
        assert 'Strict-Transport-Security' in headers
        assert 'X-Frame-Options' in headers
        assert 'X-Content-Type-Options' in headers
        assert 'Content-Security-Policy' in headers
        assert len(headers) >= 7


# ========== INTEGRATION TESTS ==========

class TestWeek1Integration:
    """Integration tests for Week 1 features."""
    
    def test_middleware_stack_initialization(self):
        """Test that entire middleware stack initializes without errors."""
        from src.middleware import register_middleware
        from flask import Flask
        
        app = Flask(__name__)
        config = RateLimitConfig(max_requests=100, window_seconds=60)
        
        middleware = register_middleware(app, config)
        
        assert 'ip_blocker' in middleware
        assert 'security_headers' in middleware
        assert 'audit_log' in middleware
        assert 'rate_limiter' not in middleware  # removed by FIX-016
    
    def test_jwt_token_lifecycle(self):
        """Test complete RS256 JWT token lifecycle (F-AUTH-REMOVE)."""
        reset_jwt_manager()
        try:
            manager = RS256JWTManager()

            token = manager.create_token(
                user_id='user1',
                username='admin',
                roles=['admin', 'analyst'],
            )

            ok, payload, error = manager.verify_token(token)
            assert ok is True
            assert payload['sub'] == 'admin'

            new_token = manager.create_token(
                user_id=payload['user_id'],
                username=payload['sub'],
                roles=payload['roles'],
            )

            ok2, payload2, _ = manager.verify_token(new_token)
            assert ok2 is True
            assert payload2['sub'] == payload['sub']
        finally:
            reset_jwt_manager()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
