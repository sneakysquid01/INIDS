"""CSRF (Cross-Site Request Forgery) protection utilities.

Provides Flask middleware and decorators for CSRF token generation, validation,
and form protection.
"""

import logging
import secrets
import hashlib
from typing import Optional, Tuple
from flask import request, g, session, render_template_string
from functools import wraps

logger = logging.getLogger(__name__)

# Session key for CSRF token
CSRF_TOKEN_SESSION_KEY = '_csrf_token'
CSRF_TOKEN_HEADER = 'X-CSRF-Token'
CSRF_TOKEN_FORM_FIELD = 'csrf_token'


def generate_csrf_token() -> str:
    """Generate a new CSRF token using secure random bytes.
    
    Returns:
        A secure CSRF token string
    """
    return secrets.token_urlsafe(32)


def get_csrf_token() -> str:
    """Get or create CSRF token for current session.
    
    Returns:
        The CSRF token for the current session
    """
    if CSRF_TOKEN_SESSION_KEY not in session:
        session[CSRF_TOKEN_SESSION_KEY] = generate_csrf_token()
    
    return session[CSRF_TOKEN_SESSION_KEY]


def validate_csrf_token(token: str) -> bool:
    """Validate a CSRF token against the session token.
    
    Args:
        token: Token to validate
        
    Returns:
        True if token is valid, False otherwise
    """
    if not token:
        logger.warning("CSRF token validation: token is empty")
        return False
    
    session_token = session.get(CSRF_TOKEN_SESSION_KEY)
    if not session_token:
        logger.warning("CSRF token validation: no session token")
        return False
    
    # Use timing-safe comparison to prevent timing attacks
    try:
        return secrets.compare_digest(token, session_token)
    except (TypeError, ValueError) as e:
        logger.warning("CSRF token validation error: %s", e)
        return False


def require_csrf_token(func):
    """Decorator to require valid CSRF token for POST/PUT/DELETE requests.
    
    Usage:
        @app.route('/api/action', methods=['POST'])
        @require_csrf_token
        def handle_action():
            # This will validate CSRF token automatically
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Only validate on state-changing methods
        if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
            return func(*args, **kwargs)
        
        # Get CSRF token from request
        token = None
        
        # Try header first (for AJAX requests)
        token = request.headers.get(CSRF_TOKEN_HEADER)
        
        # Try form data if not in header
        if not token and request.form:
            token = request.form.get(CSRF_TOKEN_FORM_FIELD)
        
        # Try JSON body if not in form
        if not token and request.is_json:
            try:
                data = request.get_json(silent=True) or {}
                token = data.get(CSRF_TOKEN_FORM_FIELD)
            except (ValueError, TypeError):
                pass
        
        # Validate token
        if not token or not validate_csrf_token(token):
            logger.warning(
                "CSRF validation failed for %s %s from %s",
                request.method,
                request.path,
                request.remote_addr
            )
            return {
                "error": "CSRF token invalid or missing",
                "code": "csrf_validation_failed"
            }, 403
        
        return func(*args, **kwargs)
    
    return wrapper


def csrf_protect_middleware(app):
    """Register Flask middleware to automatically handle CSRF tokens.

    Stateless JWT-authenticated ``/api/*`` paths are exempt — CSRF is not
    exploitable when the browser cannot inject a JWT from another origin.
    """
    @app.before_request
    def before_request():
        """Ensure CSRF token is available for HTML session; skip for API paths."""
        if request.path.startswith('/api/'):
            return None
        # Generate CSRF token if not present
        get_csrf_token()
    
    @app.context_processor
    def inject_csrf_token():
        """Inject CSRF token into template context."""
        return {'csrf_token': get_csrf_token()}


def render_form_with_csrf(html_template: str) -> str:
    """Render template with CSRF token injected.
    
    Args:
        html_template: HTML template string containing {{ csrf_token() }}
        
    Returns:
        Rendered HTML with CSRF token
    """
    def csrf_token():
        return get_csrf_token()
    
    return render_template_string(html_template, csrf_token=csrf_token)


def create_csrf_token_field(form_name: str = CSRF_TOKEN_FORM_FIELD) -> str:
    """Create HTML hidden input field with CSRF token.
    
    Args:
        form_name: Name attribute for the hidden input
        
    Returns:
        HTML hidden input with CSRF token
    """
    token = get_csrf_token()
    return (
        f'<input type="hidden" name="{form_name}" '
        f'id="{form_name}" value="{token}" />'
    )


def create_csrf_header_value() -> str:
    """Get CSRF token value for use in X-CSRF-Token header.
    
    Returns:
        CSRF token for HTTP header
    """
    return get_csrf_token()


class CSRFProtectedJsonResponse:
    """Helper for returning JSON responses with CSRF token.
    
    Usage:
        from flask import jsonify
        response_data = CSRFProtectedJsonResponse()
        response_data.set_data({'message': 'Success'})
        response_data.add_csrf_token()
        return jsonify(response_data.get_data())
    """
    
    def __init__(self):
        self.data = {}
    
    def set_data(self, data: dict) -> 'CSRFProtectedJsonResponse':
        """Set response data.
        
        Args:
            data: Dictionary of response data
            
        Returns:
            Self for method chaining
        """
        self.data = data
        return self
    
    def add_csrf_token(self) -> 'CSRFProtectedJsonResponse':
        """Add CSRF token to response.
        
        Returns:
            Self for method chaining
        """
        self.data['_csrf_token'] = get_csrf_token()
        return self
    
    def add_error(self, error: str, code: Optional[str] = None) -> 'CSRFProtectedJsonResponse':
        """Add error message to response.
        
        Args:
            error: Error message
            code: Optional error code
            
        Returns:
            Self for method chaining
        """
        self.data['error'] = error
        if code:
            self.data['error_code'] = code
        self.data['_csrf_token'] = get_csrf_token()
        return self
    
    def get_data(self) -> dict:
        """Get response data.
        
        Returns:
            Dictionary of response data
        """
        return self.data


def exempt_from_csrf(*func_or_view):
    """Decorator to exempt a view from CSRF protection.
    
    Usage:
        @app.route('/webhook', methods=['POST'])
        @exempt_from_csrf
        def handle_webhook():
            pass
    """
    def decorator(func):
        func.csrf_exempt = True
        return func
    
    if len(func_or_view) == 1 and callable(func_or_view[0]):
        return decorator(func_or_view[0])
    
    return decorator
