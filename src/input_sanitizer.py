"""Input validation and sanitization utilities.

Provides comprehensive functions for sanitizing and validating user inputs
to prevent injection attacks, malformed data, and resource exhaustion.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote, unquote

logger = logging.getLogger(__name__)

# Validation patterns
ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
IPV4_PATTERN = re.compile(
    r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
    r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
)
IPV6_PATTERN = re.compile(
    r'^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|'
    r'([0-9a-fA-F]{1,4}:){1,7}:|'
    r'([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|'
    r'([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|'
    r'([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|'
    r'([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|'
    r'([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|'
    r'[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|'
    r':((:[0-9a-fA-F]{1,4}){1,7}|:)|'
    r'fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|'
    r'::(ffff(:0{1,4}){0,1}:){0,1}'
    r'((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3}'
    r'(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|'
    r'([0-9a-fA-F]{1,4}:){1,4}:'
    r'((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3}'
    r'(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$'
)
PORT_PATTERN = re.compile(r'^[0-9]{1,5}$')
SEVERITY_PATTERN = re.compile(r'^(low|medium|high|critical)$', re.IGNORECASE)

# SQL injection patterns (simple detection)
SQL_KEYWORDS = {'select', 'insert', 'update', 'delete', 'drop', 'union', 'exec', 'script'}
SQL_INJECTION_PATTERN = re.compile(
    r"('|\"|-{2}|;|\/\*|\*\/|xp_|sp_|exec|execute)",
    re.IGNORECASE
)

# XSS patterns (simple detection)
XSS_PATTERN = re.compile(
    r'(<script|<iframe|<object|<embed|<img|javascript:|onerror=|onclick=|onload=)',
    re.IGNORECASE
)


class SanitizationError(ValueError):
    """Raised when input sanitization fails."""
    pass


def sanitize_string(
    value: Any,
    *,
    max_length: int = 1000,
    allow_special_chars: bool = False,
    lowercase: bool = False,
    allow_spaces: bool = False,
) -> str:
    """Sanitize and validate a string input.
    
    Args:
        value: Input value to sanitize
        max_length: Maximum allowed string length
        allow_special_chars: Allow special characters
        lowercase: Convert to lowercase
        allow_spaces: Allow whitespace characters
        
    Returns:
        Sanitized string
        
    Raises:
        SanitizationError: If input fails validation
    """
    if not isinstance(value, str):
        try:
            value = str(value)
        except (TypeError, ValueError) as e:
            raise SanitizationError(f"Cannot convert to string: {e}")
    
    # Check XSS patterns
    if XSS_PATTERN.search(value):
        raise SanitizationError("Potential XSS attack detected")
    
    # Check SQL injection patterns
    if not allow_special_chars and SQL_INJECTION_PATTERN.search(value):
        raise SanitizationError("Potential SQL injection detected")
    
    # Length check
    if len(value) > max_length:
        raise SanitizationError(f"String exceeds maximum length of {max_length}")
    
    # Strip leading/trailing whitespace
    value = value.strip()
    
    # Normalize whitespace if not allowing spaces
    if not allow_spaces:
        value = value.replace('\n', '').replace('\r', '').replace('\t', '')
    
    # Lowercase if requested
    if lowercase:
        value = value.lower()
    
    # If special chars not allowed, check character set
    if not allow_special_chars:
        if not ALPHANUMERIC_PATTERN.match(value) and value:
            raise SanitizationError("String contains invalid characters")
    
    return value


def sanitize_id(value: Any, *, max_length: int = 100) -> str:
    """Sanitize an ID field (alert_id, engine_id, etc.)
    
    Args:
        value: ID value to sanitize
        max_length: Maximum ID length
        
    Returns:
        Sanitized ID
        
    Raises:
        SanitizationError: If ID fails validation
    """
    try:
        sanitized = sanitize_string(
            value,
            max_length=max_length,
            allow_special_chars=False,
            allow_spaces=False,
        )
    except SanitizationError as e:
        raise SanitizationError(f"Invalid ID format: {e}")
    
    if not sanitized:
        raise SanitizationError("ID cannot be empty")
    
    return sanitized


def sanitize_ip_address(value: Any) -> str:
    """Validate and sanitize an IP address (IPv4 or IPv6).
    
    Args:
        value: IP address to validate
        
    Returns:
        Sanitized IP address
        
    Raises:
        SanitizationError: If IP is invalid
    """
    if not isinstance(value, str):
        try:
            value = str(value)
        except (TypeError, ValueError) as e:
            raise SanitizationError(f"Cannot convert to string: {e}")
    
    value = value.strip()
    
    if not value or len(value) > 45:  # Max IPv6 length
        raise SanitizationError("Invalid IP address length")
    
    if IPV4_PATTERN.match(value) or IPV6_PATTERN.match(value):
        return value
    
    raise SanitizationError("Invalid IP address format")


def sanitize_port(value: Any) -> int:
    """Validate and convert a port number.
    
    Args:
        value: Port number
        
    Returns:
        Valid port number (1-65535)
        
    Raises:
        SanitizationError: If port is invalid
    """
    try:
        if isinstance(value, str):
            value = value.strip()
            if not PORT_PATTERN.match(value):
                raise ValueError("Invalid port format")
        port_num = int(value)
        if not (1 <= port_num <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return port_num
    except (ValueError, TypeError) as e:
        raise SanitizationError(f"Invalid port: {e}")


def sanitize_severity(value: Any) -> str:
    """Validate and sanitize a severity level.
    
    Args:
        value: Severity level string
        
    Returns:
        Normalized severity (lowercase)
        
    Raises:
        SanitizationError: If severity is invalid
    """
    try:
        sanitized = sanitize_string(value, max_length=20, allow_special_chars=False)
    except SanitizationError as e:
        raise SanitizationError(f"Invalid severity format: {e}")
    
    if not SEVERITY_PATTERN.match(sanitized):
        raise SanitizationError("Severity must be one of: low, medium, high, critical")
    
    return sanitized.lower()


def sanitize_url_path(value: Any, *, max_length: int = 2000) -> str:
    """Sanitize a URL path component.
    
    Args:
        value: URL path to sanitize
        max_length: Maximum path length
        
    Returns:
        Sanitized URL path
        
    Raises:
        SanitizationError: If path is invalid
    """
    if not isinstance(value, str):
        try:
            value = str(value)
        except (TypeError, ValueError) as e:
            raise SanitizationError(f"Cannot convert to string: {e}")
    
    value = value.strip()
    
    if len(value) > max_length:
        raise SanitizationError(f"Path exceeds maximum length of {max_length}")
    
    # Prevent directory traversal
    if '..' in value or value.startswith('/'):
        raise SanitizationError("Invalid path format (directory traversal attempt)")
    
    # URL decode to check for encoded traversal
    try:
        decoded = unquote(value)
        if '..' in decoded:
            raise SanitizationError("Invalid path format (encoded directory traversal)")
    except (ValueError, TypeError):
        pass
    
    return value


def sanitize_json_object(data: Dict[str, Any], *, max_size: int = 1000000) -> Dict[str, Any]:
    """Validate JSON object size and structure.
    
    Args:
        data: Dictionary to validate
        max_size: Maximum size in bytes (approximate)
        
    Returns:
        Validated dictionary
        
    Raises:
        SanitizationError: If object is invalid
    """
    if not isinstance(data, dict):
        raise SanitizationError("Expected dictionary")
    
    import json
    try:
        json_str = json.dumps(data)
        if len(json_str) > max_size:
            raise SanitizationError(f"JSON exceeds maximum size of {max_size} bytes")
    except (TypeError, ValueError) as e:
        raise SanitizationError(f"Invalid JSON structure: {e}")
    
    return data


def sanitize_integer(
    value: Any,
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """Sanitize and validate an integer input.
    
    Args:
        value: Integer value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Sanitized integer
        
    Raises:
        SanitizationError: If value is invalid
    """
    try:
        if isinstance(value, str):
            value = value.strip()
        int_val = int(value)
    except (ValueError, TypeError) as e:
        raise SanitizationError(f"Invalid integer: {e}")
    
    if min_value is not None and int_val < min_value:
        raise SanitizationError(f"Value must be >= {min_value}")
    
    if max_value is not None and int_val > max_value:
        raise SanitizationError(f"Value must be <= {max_value}")
    
    return int_val


def sanitize_float(
    value: Any,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """Sanitize and validate a float input.
    
    Args:
        value: Float value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Sanitized float
        
    Raises:
        SanitizationError: If value is invalid
    """
    try:
        if isinstance(value, str):
            value = value.strip()
        float_val = float(value)
    except (ValueError, TypeError) as e:
        raise SanitizationError(f"Invalid float: {e}")
    
    if min_value is not None and float_val < min_value:
        raise SanitizationError(f"Value must be >= {min_value}")
    
    if max_value is not None and float_val > max_value:
        raise SanitizationError(f"Value must be <= {max_value}")
    
    return float_val
