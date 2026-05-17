"""Request correlation utilities for distributed tracing.

Provides request correlation IDs to track requests across the system,
enabling better debugging and performance monitoring of distributed flows.
"""

import logging
import uuid
from contextvars import ContextVar
from typing import Optional
from flask import request, g
from functools import wraps

logger = logging.getLogger(__name__)

# Context variable to store correlation ID
_correlation_id_context: ContextVar[str] = ContextVar('correlation_id', default='')

# HTTP header for correlation ID
CORRELATION_ID_HEADER = 'X-Correlation-ID'
REQUEST_ID_HEADER = 'X-Request-ID'

_MAX_CORRELATION_ID_LEN = 64
# Characters that allow log injection — reject the entire header value if present
_LOG_INJECTION_CHARS = frozenset({'\n', '\r', '\x00'})


def generate_correlation_id() -> str:
    """Generate a new correlation ID using UUID v4."""
    return f"req_{uuid.uuid4().hex[:16]}"


def sanitize_correlation_id(value: str | None) -> str:
    """Sanitize an incoming X-Correlation-ID header value.

    Returns a server-generated ID when the value is absent or contains
    log-injection characters (\\n, \\r, null byte). Non-printable characters
    are stripped; the result is truncated to 64 characters.
    """
    if not value:
        return generate_correlation_id()

    if any(c in value for c in _LOG_INJECTION_CHARS):
        logger.warning("correlation_id_rejected reason=log_injection_chars value_len=%d", len(value))
        return generate_correlation_id()

    sanitized = "".join(c for c in value if c.isprintable())
    sanitized = sanitized[:_MAX_CORRELATION_ID_LEN]

    if not sanitized:
        return generate_correlation_id()

    return sanitized


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context.

    Args:
        correlation_id: The correlation ID to set
    """
    _correlation_id_context.set(correlation_id)
    try:
        g.correlation_id = correlation_id
    except RuntimeError:
        pass  # outside Flask application context — ContextVar is the authority


def get_correlation_id() -> str:
    """Get the current correlation ID.
    
    Returns:
        The current correlation ID, or generates a new one if not set
    """
    # Try context var first
    correlation_id = _correlation_id_context.get()
    if correlation_id:
        return correlation_id
    
    # Try Flask g object
    if hasattr(g, 'correlation_id'):
        return g.correlation_id
    
    # Try request headers — sanitize before trusting
    if request:
        raw = request.headers.get(CORRELATION_ID_HEADER) or request.headers.get(REQUEST_ID_HEADER)
        if raw:
            correlation_id = sanitize_correlation_id(raw)
            set_correlation_id(correlation_id)
            return correlation_id
    
    # Generate new one
    correlation_id = generate_correlation_id()
    set_correlation_id(correlation_id)
    return correlation_id


def attach_correlation_id_to_logs(
    logger_instance: logging.Logger,
    correlation_id: Optional[str] = None
) -> None:
    """Attach correlation ID to all log records from a logger.
    
    Args:
        logger_instance: Logger instance to modify
        correlation_id: Correlation ID to attach (uses current if None)
    """
    if correlation_id is None:
        correlation_id = get_correlation_id()
    
    # Create a filter that adds correlation ID
    class CorrelationFilter(logging.Filter):
        def filter(self, record):
            record.correlation_id = correlation_id
            return True
    
    # Add the filter to the logger
    if not any(isinstance(f, CorrelationFilter) for f in logger_instance.filters):
        logger_instance.addFilter(CorrelationFilter())


def correlation_id_middleware(app) -> None:
    """Register Flask middleware to handle correlation IDs on all requests.
    
    Args:
        app: Flask application instance
    """
    @app.before_request
    def before_request():
        """Process incoming request and set up correlation ID."""
        raw = request.headers.get(CORRELATION_ID_HEADER) or request.headers.get(REQUEST_ID_HEADER)
        correlation_id = sanitize_correlation_id(raw)

        set_correlation_id(correlation_id)
        g.correlation_id = correlation_id
        g.request_start_time = __import__('time').time()
    
    @app.after_request
    def after_request(response):
        """Attach correlation ID to response headers."""
        correlation_id = get_correlation_id()
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        
        # Add request duration if available
        if hasattr(g, 'request_start_time'):
            import time
            duration_ms = (time.time() - g.request_start_time) * 1000
            response.headers['X-Request-Duration-MS'] = f"{duration_ms:.2f}"
        
        return response


def with_correlation_id(func):
    """Decorator to ensure correlation ID is set for async functions or methods.
    
    Usage:
        @with_correlation_id
        def my_function():
            correlation_id = get_correlation_id()
            # Do work...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Ensure correlation ID is set
        if not _correlation_id_context.get():
            set_correlation_id(generate_correlation_id())
        
        return func(*args, **kwargs)
    
    return wrapper


def create_correlation_logger(name: str) -> logging.Logger:
    """Create a logger that automatically includes correlation IDs.
    
    Args:
        name: Logger name
        
    Returns:
        A logger instance configured with correlation ID support
    """
    logger_instance = logging.getLogger(name)
    
    # Add formatter that includes correlation ID
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(correlation_id)s] - %(levelname)s - %(message)s'
    )
    
    # Ensure formatter is applied to handlers
    for handler in logger_instance.handlers:
        handler.setFormatter(formatter)
    
    # Add correlation filter
    attach_correlation_id_to_logs(logger_instance)
    
    return logger_instance


class CorrelationContextManager:
    """Context manager for setting correlation ID scope.
    
    Usage:
        with CorrelationContextManager('my-correlation-id'):
            # Code here has the specified correlation ID
            pass
    """
    
    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or generate_correlation_id()
        self.previous_id = None
    
    def __enter__(self):
        self.previous_id = _correlation_id_context.get()
        set_correlation_id(self.correlation_id)
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_id:
            set_correlation_id(self.previous_id)
        else:
            _correlation_id_context.set('')
        return False
