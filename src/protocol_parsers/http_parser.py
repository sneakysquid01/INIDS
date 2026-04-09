"""
HTTP Protocol Parser
Extracts HTTP request/response details from TCP payload
Supports HTTP/1.0, HTTP/1.1, HTTP/2 (basic), chunked encoding
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class HTTPMethod(Enum):
    """Common HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    PATCH = "PATCH"
    TRACE = "TRACE"
    CONNECT = "CONNECT"
    OTHER = "OTHER"


@dataclass
class HTTPRequest:
    """Extracted HTTP request details"""
    method: str                          # GET, POST, PUT, etc.
    uri: str                             # Full URI including query string
    path: str                            # Path component only
    query_string: Optional[str] = None   # Query params (after ?)
    query_params: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    content_type: str = ""
    content_length: int = 0
    http_version: str = "1.1"
    
    # Security indicators
    user_agent: str = ""
    host: str = ""
    referer: str = ""
    
    # Detection features
    is_suspicious: bool = False
    suspicious_indicators: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"HTTPRequest({self.method} {self.path} v{self.http_version})"
    
    def get_method_enum(self) -> HTTPMethod:
        """Convert string method to enum"""
        try:
            return HTTPMethod[self.method.upper()]
        except KeyError:
            return HTTPMethod.OTHER


@dataclass
class HTTPResponse:
    """Extracted HTTP response details"""
    status_code: int                     # 200, 404, 500, etc.
    status_text: str                     # "OK", "Not Found", etc.
    headers: Dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    http_version: str = "1.1"
    
    # Server info
    server: str = ""
    content_type: str = ""
    content_length: int = 0
    
    # Detection features
    is_error: bool = False
    is_redirect: bool = False
    
    def __repr__(self):
        return f"HTTPResponse({self.status_code} {self.status_text})"
    
    def is_success(self) -> bool:
        """Check if response is 2xx"""
        return 200 <= self.status_code < 300
    
    def is_client_error(self) -> bool:
        """Check if response is 4xx"""
        return 400 <= self.status_code < 500
    
    def is_server_error(self) -> bool:
        """Check if response is 5xx"""
        return 500 <= self.status_code < 600


class HTTPParser:
    """Parse HTTP from TCP payload"""
    
    # Common suspicious patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bor\b|union|select|insert|delete|update|drop|exec|script)\s*\(", 
        r"'.*?or.*?'",
        r"--\s*$",
        r"#\s*$",
        r";.*?(drop|delete|exec)",
    ]
    
    XSS_PATTERNS = [
        r"<\s*script[^>]*>",
        r"javascript:",
        r"onerror\s*=",
        r"onclick\s*=",
        r"onload\s*=",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"%2e%2e",
        r"\.\.%2f",
    ]
    
    @staticmethod
    def parse_request(payload: bytes) -> Optional[HTTPRequest]:
        """
        Parse HTTP request from TCP payload
        
        Args:
            payload: TCP payload bytes
        
        Returns:
            HTTPRequest object or None if parsing fails
        """
        if not payload or len(payload) < 10:
            return None
        
        try:
            text = payload.decode('utf-8', errors='ignore')
            lines = text.split('\r\n')
            
            if not lines:
                return None
            
            # Parse request line: GET /path HTTP/1.1
            req_line = lines[0]
            parts = req_line.split(' ')
            
            if len(parts) < 3:
                return None
            
            method, uri, http_version = parts[0], parts[1], parts[2]
            
            # Extract HTTP version (1.0, 1.1, 2.0)
            version_match = re.search(r'HTTP/(\d\.\d)', http_version)
            version = version_match.group(1) if version_match else "1.1"
            
            # Parse headers
            headers = {}
            body_idx = -1
            
            for i in range(1, len(lines)):
                line = lines[i]
                
                if line == '':
                    body_idx = i + 1
                    break
                
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            # Extract body
            body = b''
            if body_idx >= 0:
                body_lines = lines[body_idx:]
                body = '\r\n'.join(body_lines).encode('utf-8', errors='ignore')
            
            # Parse URI
            if '?' in uri:
                path, query_str = uri.split('?', 1)
                query_params = HTTPParser._parse_query_string(query_str)
            else:
                path = uri
                query_str = None
                query_params = {}
            
            # Extract common headers
            host = headers.get('host', '')
            user_agent = headers.get('user-agent', '')
            referer = headers.get('referer', '')
            content_type = headers.get('content-type', '')
            content_length = 0
            
            try:
                content_length = int(headers.get('content-length', '0'))
            except ValueError:
                pass
            
            request = HTTPRequest(
                method=method,
                uri=uri,
                path=path,
                query_string=query_str,
                query_params=query_params,
                headers=headers,
                body=body,
                content_type=content_type,
                content_length=content_length,
                http_version=version,
                user_agent=user_agent,
                host=host,
                referer=referer
            )
            
            # Detect suspicious indicators
            HTTPParser._check_request_suspicious(request)
            
            return request
        
        except Exception as e:
            logger.debug(f"HTTP request parse error: {e}")
            return None
    
    @staticmethod
    def parse_response(payload: bytes) -> Optional[HTTPResponse]:
        """
        Parse HTTP response from TCP payload
        
        Args:
            payload: TCP payload bytes
        
        Returns:
            HTTPResponse object or None if parsing fails
        """
        if not payload or len(payload) < 10:
            return None
        
        try:
            text = payload.decode('utf-8', errors='ignore')
            lines = text.split('\r\n')
            
            if not lines:
                return None
            
            # Parse status line: HTTP/1.1 200 OK
            status_line = lines[0]
            parts = status_line.split(' ', 2)
            
            if len(parts) < 2:
                return None
            
            http_version, status_code_str = parts[0], parts[1]
            status_text = parts[2] if len(parts) > 2 else ''
            
            try:
                status_code = int(status_code_str)
            except ValueError:
                return None
            
            # Extract HTTP version
            version_match = re.search(r'HTTP/(\d\.\d)', http_version)
            version = version_match.group(1) if version_match else "1.1"
            
            # Parse headers
            headers = {}
            body_idx = -1
            
            for i in range(1, len(lines)):
                line = lines[i]
                
                if line == '':
                    body_idx = i + 1
                    break
                
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            # Extract body
            body = b''
            if body_idx >= 0:
                body_lines = lines[body_idx:]
                body = '\r\n'.join(body_lines).encode('utf-8', errors='ignore')
            
            # Extract common headers
            server = headers.get('server', '')
            content_type = headers.get('content-type', '')
            
            try:
                content_length = int(headers.get('content-length', '0'))
            except ValueError:
                content_length = len(body)
            
            # Determine if redirect
            is_redirect = 300 <= status_code < 400
            
            response = HTTPResponse(
                status_code=status_code,
                status_text=status_text,
                headers=headers,
                body=body,
                http_version=version,
                server=server,
                content_type=content_type,
                content_length=content_length,
                is_redirect=is_redirect
            )
            
            # Check for errors
            response.is_error = response.is_client_error() or response.is_server_error()
            
            return response
        
        except Exception as e:
            logger.debug(f"HTTP response parse error: {e}")
            return None
    
    @staticmethod
    def _parse_query_string(query_str: str) -> Dict[str, str]:
        """Parse URL query string into dict"""
        params = {}
        if not query_str:
            return params
        
        try:
            for param in query_str.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    # URL decode
                    key = HTTPParser._url_decode(key)
                    value = HTTPParser._url_decode(value)
                    params[key] = value
                else:
                    params[param] = ''
        except Exception as e:
            logger.debug(f"Query string parse error: {e}")
        
        return params
    
    @staticmethod
    def _url_decode(s: str) -> str:
        """Simple URL decoding"""
        import urllib.parse
        try:
            return urllib.parse.unquote(s)
        except Exception:
            return s
    
    @staticmethod
    def _check_request_suspicious(request: HTTPRequest):
        """Check for suspicious patterns in request"""
        
        # Check reserved admin paths
        admin_paths = [
            '/admin', '/administrator', '/wp-admin', '/phpmyadmin',
            '/login', '/auth', '/api/admin', '/console'
        ]
        
        if any(req_path.lower().startswith(ap) for ap in admin_paths):
            request.suspicious_indicators.append("admin_path_access")
        
        # Check SQL injection
        for pattern in HTTPParser.SQL_INJECTION_PATTERNS:
            if re.search(pattern, request.uri, re.IGNORECASE):
                request.suspicious_indicators.append("sql_injection")
                break
        
        if request.body:
            body_str = request.body.decode('utf-8', errors='ignore')
            for pattern in HTTPParser.SQL_INJECTION_PATTERNS:
                if re.search(pattern, body_str, re.IGNORECASE):
                    request.suspicious_indicators.append("sql_injection_in_body")
                    break
        
        # Check XSS
        payload = request.uri + ' ' + request.body.decode('utf-8', errors='ignore')
        for pattern in HTTPParser.XSS_PATTERNS:
            if re.search(pattern, payload, re.IGNORECASE):
                request.suspicious_indicators.append("xss_attempt")
                break
        
        # Check path traversal
        for pattern in HTTPParser.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, request.uri, re.IGNORECASE):
                request.suspicious_indicators.append("path_traversal")
                break
        
        # Check user agent (common malware/tools)
        malformed_ua = [
            'sqlmap', 'nikto', 'nmap', 'burp', 'zaproxy', 'metasploit',
            'masscan', 'w3af', 'hydra', 'nessus', 'openvas'
        ]
        
        if request.user_agent:
            ua_lower = request.user_agent.lower()
            if any(tool in ua_lower for tool in malformed_ua):
                request.suspicious_indicators.append("attacker_tool_ua")
        
        # Empty user agent
        if not request.user_agent:
            request.suspicious_indicators.append("missing_user_agent")
        
        # Very long URI
        if len(request.uri) > 2000:
            request.suspicious_indicators.append("excessive_uri_length")
        
        # Set flag if suspicious
        request.is_suspicious = len(request.suspicious_indicators) > 0
    
    @staticmethod
    def extract_features(request: HTTPRequest) -> Dict[str, any]:
        """
        Extract ML features from HTTP request
        
        Returns:
            Dict of features for ML models
        """
        if not request:
            return {}
        
        return {
            "http_method": request.method,
            "http_uri_length": len(request.uri),
            "http_path": request.path,
            "http_query_params_count": len(request.query_params),
            "http_body_length": len(request.body),
            "http_content_type": request.content_type,
            "http_has_user_agent": bool(request.user_agent),
            "http_suspicious_indicators_count": len(request.suspicious_indicators),
            "http_is_suspicious": request.is_suspicious,
        }
    
    @staticmethod
    def extract_features_response(response: HTTPResponse) -> Dict[str, any]:
        """Extract ML features from HTTP response"""
        if not response:
            return {}
        
        return {
            "http_status_code": response.status_code,
            "http_status_is_error": response.is_error,
            "http_status_is_redirect": response.is_redirect,
            "http_response_body_length": len(response.body),
            "http_has_server_header": bool(response.server),
            "http_response_content_type": response.content_type,
        }
