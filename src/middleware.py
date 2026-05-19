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
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from typing import Dict, Set, Tuple, Optional, Callable, Any
from dataclasses import dataclass, asdict
from functools import wraps
from threading import RLock

from flask import g, request, Response, jsonify

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
        'Content-Security-Policy': "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net https://cdn.socket.io; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net; connect-src 'self' ws://127.0.0.1:5000 ws://localhost:5000 https://fonts.googleapis.com https://fonts.gstatic.com https://cdn.jsdelivr.net https://cdn.socket.io; img-src 'self' data:",
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
        
        # Extract user from verified auth context (g.auth set by require_roles).
        # Never read from client-controlled X-User-ID header (FIX-003).
        _auth_ctx = getattr(g, "auth", None)
        user = getattr(_auth_ctx, "username", None) or "anonymous"
        
        # Don't log sensitive data (passwords in request body)
        error = None
        if response.status_code >= 400:
            try:
                error = response.get_json().get('error') if response.is_json else None
            except Exception:
                error = None
        
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
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
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff.isoformat()
        
        with self.lock:
            return [
                asdict(log) for log in self.logs
                if log.user == user and log.timestamp > cutoff_iso
            ]

class CORSMiddleware:
    """CORS middleware with security controls."""

    _DEFAULT_ORIGINS = ['http://localhost:5000', 'http://127.0.0.1:5000']
    _DEFAULT_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    _DEFAULT_HEADERS = ['Authorization', 'Content-Type', 'X-API-Key', 'X-Correlation-ID']

    def __init__(
        self,
        allowed_origins: list = None,
        allowed_methods: list = None,
        allowed_headers: list = None,
        allow_credentials: bool = True,
    ):
        self.allowed_origins = allowed_origins or self._DEFAULT_ORIGINS
        self.allowed_methods = allowed_methods or self._DEFAULT_METHODS
        self.allowed_headers = allowed_headers or self._DEFAULT_HEADERS
        self.allow_credentials = allow_credentials

    def handle_preflight(self) -> Optional[Response]:
        """Return a CORS preflight response for OPTIONS requests from allowed origins."""
        if request.method != 'OPTIONS':
            return None
        origin = request.headers.get('Origin')
        if origin not in self.allowed_origins:
            return None
        resp = Response(status=200)
        resp.headers['Access-Control-Allow-Origin'] = origin
        resp.headers['Access-Control-Allow-Methods'] = ', '.join(self.allowed_methods)
        resp.headers['Access-Control-Allow-Headers'] = ', '.join(self.allowed_headers)
        if self.allow_credentials:
            resp.headers['Access-Control-Allow-Credentials'] = 'true'
        resp.headers['Access-Control-Max-Age'] = '3600'
        return resp

    def add_headers(self, response: Response) -> Response:
        """Add CORS headers to actual (non-preflight) responses."""
        origin = request.headers.get('Origin')
        if origin in self.allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = ', '.join(self.allowed_methods)
            response.headers['Access-Control-Allow-Headers'] = ', '.join(self.allowed_headers)
            if self.allow_credentials:
                response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Access-Control-Max-Age'] = '3600'
        return response

    def after_request(self, response: Response) -> Response:
        """Alias for add_headers — backward compatibility."""
        return self.add_headers(response)


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


def register_middleware(
    app,
    rate_limit_config: RateLimitConfig = None,
    cors_origins: list = None,
):
    """Register all middleware with Flask app.

    Args:
        app: Flask application
        rate_limit_config: Rate limit configuration
        cors_origins: Allowed CORS origins (reads INIDS_CORS_ORIGINS env var as fallback)
    """
    import os as _os
    if cors_origins is None:
        raw = _os.getenv("INIDS_CORS_ORIGINS", "").strip()
        cors_origins = [o.strip() for o in raw.split(",") if o.strip()] if raw else CORSMiddleware._DEFAULT_ORIGINS

    ip_blocker = IPBlockingMiddleware()
    security_headers = SecurityHeadersMiddleware()
    audit_log = AuditLogMiddleware()
    cors = CORSMiddleware(
        allowed_origins=cors_origins,
        allowed_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        allowed_headers=['Authorization', 'Content-Type', 'X-API-Key', 'X-Correlation-ID'],
        allow_credentials=True,
    )
    request_validator = RequestValidationMiddleware()
    content_security = ContentSecurityMiddleware()

    @app.before_request
    def before_request():
        # CORS preflight first — must bypass auth/rate-limit
        result = cors.handle_preflight()
        if result:
            return result

        result = ip_blocker.before_request()
        if result:
            return result

        # C-05 (Step 19): RateLimitMiddleware removed from chain.
        # Rate limiting is now handled by UnifiedRateLimiter in before_request
        # (Tier 1 global IP) and require_roles() (Tier 2 per-user).

        result = request_validator.before_request()
        if result:
            return result

        audit_log.before_request()

    @app.after_request
    def after_request(response):
        response = ip_blocker.after_request(response)
        response = security_headers.after_request(response)
        response = cors.add_headers(response)
        response = content_security.after_request(response)
        response = audit_log.after_request(response)
        return response

    app.ip_blocker = ip_blocker
    app.audit_log = audit_log

    logger.info("Security middleware stack registered")

    return {
        'ip_blocker': ip_blocker,
        'security_headers': security_headers,
        'audit_log': audit_log,
        'cors': cors,
        'request_validator': request_validator,
    }
