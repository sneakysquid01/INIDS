"""
INIDS Security & Observability Middleware Stack

Implements:
- Rate limiting (per API key + per IP)
- IP blocking (after failed auth attempts)
- Security headers (OWASP standards)
- Request audit logging
"""

import time
import logging
import json
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, Set, Tuple, Optional, Callable, Any
from dataclasses import dataclass, asdict
from functools import wraps
from threading import RLock

from flask import request, Response, jsonify

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests: int = 300  # Per window (compatible with rate_limiter.py)
    window_seconds: int = 60
    burst_allowance: int = 10  # Allow burst up to this % over limit temporarily
    # Alias support
    max_requests: int = None
    
    def __post_init__(self):
        if self.max_requests is not None and self.requests == 300:
            object.__setattr__(self, 'requests', self.max_requests)


@dataclass
class AuditLogEntry:
    """Audit log entry for all API requests."""
    timestamp: str
    user: str
    method: str
    path: str
    status: int
    response_time_ms: float
    source_ip: str
    user_agent: str
    request_body_size: int
    response_body_size: int
    error: Optional[str] = None


class RateLimitMiddleware:
    """Rate limiting middleware - per API key and per IP.
    
    Prevents abuse by limiting requests per user and per IP address.
    Uses sliding window algorithm.
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = RLock()
    
    def get_client_key(self) -> str:
        """Get unique key for rate limiting (API key > IP)."""
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return f"api_key:{api_key}"
        return f"ip:{request.remote_addr}"
    
    def is_rate_limited(self) -> Tuple[bool, int]:
        """Check if request should be rate limited.
        
        Returns:
            (is_limited, retry_after_seconds)
        """
        with self.lock:
            key = self.get_client_key()
            now = time.time()
            
            # Remove old requests outside window
            window_start = now - self.config.window_seconds
            while self.requests[key] and self.requests[key][0] < window_start:
                self.requests[key].popleft()
            
            request_count = len(self.requests[key])
            
            # Check if over limit
            if request_count >= self.config.requests:
                # Calculate retry-after
                oldest_request = self.requests[key][0]
                retry_after = int(oldest_request + self.config.window_seconds - now) + 1
                return True, retry_after
            
            # Add current request
            self.requests[key].append(now)
            return False, 0
    
    def before_request(self) -> Optional[Response]:
        """Flask before_request hook."""
        is_limited, retry_after = self.is_rate_limited()
        
        if is_limited:
            logger.warning(f"Rate limit exceeded for {self.get_client_key()}")
            return jsonify({
                'error': 'Rate limit exceeded',
                'retry_after': retry_after
            }), 429, {'Retry-After': str(retry_after)}
        
        return None


class IPBlockingMiddleware:
    """IP blocking middleware - automatic blocking after failed auth."""
    
    def __init__(self, max_failures: int = 5, block_time_seconds: int = 300):
        self.max_failures = max_failures
        self.block_time = block_time_seconds
        self.failures: Dict[str, deque] = defaultdict(deque)
        self.blocked: Dict[str, float] = {}  # {ip: unblock_time}
        self.whitelist: Set[str] = {'127.0.0.1', 'localhost', '::1'}
        self.lock = RLock()
    
    def add_failure(self, ip: str) -> None:
        """Record authentication failure for IP."""
        if ip in self.whitelist:
            return
        
        with self.lock:
            now = time.time()
            # Remove old failures outside window
            window_start = now - self.block_time
            while self.failures[ip] and self.failures[ip][0] < window_start:
                self.failures[ip].popleft()
            
            self.failures[ip].append(now)
            
            # Check if should block
            if len(self.failures[ip]) >= self.max_failures:
                self.blocked[ip] = now + self.block_time
                logger.warning(f"IP blocked after {self.max_failures} failures: {ip}")
    
    def is_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked."""
        if ip in self.whitelist:
            return False
        
        with self.lock:
            if ip not in self.blocked:
                return False
            
            now = time.time()
            if now > self.blocked[ip]:
                del self.blocked[ip]
                return False
            
            return True
    
    def before_request(self) -> Optional[Response]:
        """Flask before_request hook."""
        ip = request.remote_addr
        
        if self.is_blocked(ip):
            logger.warning(f"Blocked request from {ip}")
            return jsonify({'error': 'Access denied - IP blocked'}), 403
        
        return None
    
    def after_request(self, response: Response) -> Response:
        """Flask after_request hook - track failed auth."""
        # Track 401/403 responses as failures
        if response.status_code in [401, 403]:
            self.add_failure(request.remote_addr)
        
        return response


class SecurityHeadersMiddleware:
    """Add OWASP security headers to responses."""
    
    HEADERS = {
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
        'X-Frame-Options': 'DENY',
        'X-Content-Type-Options': 'nosniff',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
    }
    
    def after_request(self, response: Response) -> Response:
        """Flask after_request hook."""
        for header, value in self.HEADERS.items():
            response.headers[header] = value
        return response


class AuditLogMiddleware:
    """Audit logging for all API requests."""
    
    def __init__(self, max_logs: int = 10000):
        self.logs: deque = deque(maxlen=max_logs)
        self.lock = RLock()
    
    def before_request(self) -> None:
        """Store request start time."""
        request.start_time = time.time()
    
    def after_request(self, response: Response) -> Response:
        """Log request after completion."""
        elapsed_ms = (time.time() - request.start_time) * 1000
        
        # Get body sizes
        request_body_size = int(request.environ.get('CONTENT_LENGTH', 0) or 0)

        # Safe handling for streamed/static responses
        response_body_size = 0
        if not getattr(response, "direct_passthrough", False):
            try:
                response_body_size = len(response.get_data())
            except Exception:
                response_body_size = 0
        
        # Extract user from JWT or request
        user = request.headers.get('X-User-ID', 'anonymous')
        
        # Don't log sensitive data (passwords in request body)
        error = None
        if response.status_code >= 400:
            try:
                error = response.get_json().get('error') if response.is_json else None
            except Exception:
                error = None
        
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            user=user,
            method=request.method,
            path=request.path,
            status=response.status_code,
            response_time_ms=elapsed_ms,
            source_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent', 'unknown'),
            request_body_size=request_body_size,
            response_body_size=response_body_size,
            error=error
        )
        
        with self.lock:
            self.logs.append(entry)
        
        # Log to logger as well
        logger.info(f"{entry.method} {entry.path} -> {entry.status} ({elapsed_ms:.1f}ms) from {entry.source_ip}")
        
        return response
    
    def get_logs(self, limit: int = 100) -> list:
        """Get recent audit logs."""
        with self.lock:
            return [asdict(log) for log in list(self.logs)[-limit:]]
    
    def get_user_activity(self, user: str, hours: int = 24) -> list:
        """Get activity for specific user."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        cutoff_iso = cutoff.isoformat()
        
        with self.lock:
            return [
                asdict(log) for log in self.logs
                if log.user == user and log.timestamp > cutoff_iso
            ]

class CORSMiddleware:
    """CORS middleware with security controls."""
    
    def __init__(self, allowed_origins: list = None, allowed_methods: list = None):
        self.allowed_origins = allowed_origins or ['http://localhost:5000', 'http://127.0.0.1:5000']
        self.allowed_methods = allowed_methods or ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    
    def after_request(self, response: Response) -> Response:
        """Add CORS headers."""
        origin = request.headers.get('Origin')
        
        if origin in self.allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = ', '.join(self.allowed_methods)
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-API-Key, X-User-ID'
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Access-Control-Max-Age'] = '3600'
        
        return response


class RequestValidationMiddleware:
    """Validate request payloads."""
    
    MAX_BODY_SIZE = 1024 * 1024  # 1 MB
    
    def before_request(self) -> Optional[Response]:
        """Validate request."""
        # Check content length
        content_length = int(request.environ.get('CONTENT_LENGTH', 0))
        if content_length > self.MAX_BODY_SIZE:
            return jsonify({
                'error': 'Payload too large',
                'max_size': self.MAX_BODY_SIZE
            }), 413
        
        # Validate JSON if present
        if request.is_json:
            try:
                request.get_json()
            except Exception as e:
                return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400
        
        return None


class ContentSecurityMiddleware:
    """Ensure responses are properly formatted."""
    
    def after_request(self, response: Response) -> Response:
        """Ensure proper content types."""
        if response.is_json:
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
        
        return response


def register_middleware(app, rate_limit_config: RateLimitConfig = None):
    """Register all middleware with Flask app.
    
    Args:
        app: Flask application
        rate_limit_config: Rate limit configuration
    """
    # Initialize middleware instances
    rate_limiter = RateLimitMiddleware(rate_limit_config or RateLimitConfig())
    ip_blocker = IPBlockingMiddleware()
    security_headers = SecurityHeadersMiddleware()
    audit_log = AuditLogMiddleware()
    cors = CORSMiddleware()
    request_validator = RequestValidationMiddleware()
    content_security = ContentSecurityMiddleware()
    
    # Register before_request hooks (order matters - security first)
    @app.before_request
    def before_request():
        # Check IP block first
        result = ip_blocker.before_request()
        if result:
            return result
        
        # Rate limiting
        result = rate_limiter.before_request()
        if result:
            return result
        
        # Request validation
        result = request_validator.before_request()
        if result:
            return result
        
        # Audit logging
        audit_log.before_request()
    
    # Register after_request hooks
    @app.after_request
    def after_request(response):
        response = ip_blocker.after_request(response)
        response = security_headers.after_request(response)
        response = cors.after_request(response)
        response = content_security.after_request(response)
        response = audit_log.after_request(response)
        return response
    
    # Store middleware references on app for API access
    app.rate_limiter = rate_limiter
    app.ip_blocker = ip_blocker
    app.audit_log = audit_log
    
    logger.info("Security middleware stack registered")
    
    return {
        'rate_limiter': rate_limiter,
        'ip_blocker': ip_blocker,
        'security_headers': security_headers,
        'audit_log': audit_log,
        'cors': cors,
        'request_validator': request_validator,
    }
