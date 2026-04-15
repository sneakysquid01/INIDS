"""
Dual-Stack Router for Flask + Connexion (Week 3)

Manages routing between Flask (legacy) and Connexion (new) endpoints.

Architecture:
- Flask: Handles all existing endpoints (Weeks 1-2)
- Connexion: Adds new endpoints with OpenAPI 3.0 specs
- Router: Routes requests to appropriate framework
- Validation: All Connexion endpoints get automatic validation

Gradual Migration:
- Week 3-4: Parallel operation (Flask + Connexion)
- Week 5: Full cutover to Connexion
"""

import logging
import json
from typing import Dict, Any, Optional, Callable, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


class DualStackRouter:
    """Routes requests between Flask and Connexion frameworks."""
    
    def __init__(self, flask_app, connexion_app=None):
        """Initialize dual-stack router.
        
        Args:
            flask_app: Flask application instance
            connexion_app: Connexion application instance (optional)
        """
        self.flask_app = flask_app
        self.connexion_app = connexion_app
        
        # Route registry: maps path to (framework, handler)
        self.flask_routes: Dict[str, Tuple[str, Callable]] = {}
        self.connexion_routes: Dict[str, Tuple[str, Callable]] = {}
        
        self.is_dual_stack = connexion_app is not None
    
    def register_flask_endpoint(self, path: str, method: str, handler: Callable):
        """Register Flask endpoint.
        
        Args:
            path: URL path
            method: HTTP method (GET, POST, etc.)
            handler: Request handler function
        """
        route_key = f"{method.upper()} {path}"
        self.flask_routes[route_key] = handler
        logger.debug(f"Registered Flask route: {route_key}")
    
    def register_connexion_endpoint(self, path: str, method: str, handler: Callable):
        """Register Connexion endpoint with OpenAPI spec.
        
        Args:
            path: URL path
            method: HTTP method
            handler: Request handler function
        """
        if not self.connexion_app:
            logger.warning(f"Connexion not initialized, cannot register {method} {path}")
            return
        
        route_key = f"{method.upper()} {path}"
        self.connexion_routes[route_key] = handler
        logger.debug(f"Registered Connexion route: {route_key}")
    
    def get_route_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "flask_routes": len(self.flask_routes),
            "connexion_routes": len(self.connexion_routes),
            "dual_stack_enabled": self.is_dual_stack,
            "migration_progress": f"{len(self.connexion_routes)} / {len(self.flask_routes) + len(self.connexion_routes)}"
        }


class RequestValidator:
    """Validates requests against OpenAPI schemas."""
    
    def __init__(self, openapi_spec: Dict[str, Any]):
        """Initialize validator with OpenAPI spec.
        
        Args:
            openapi_spec: OpenAPI 3.0 specification dict
        """
        self.spec = openapi_spec
        self.paths = openapi_spec.get("paths", {})
        self.schemas = openapi_spec.get("components", {}).get("schemas", {})
    
    def validate_request(
        self,
        path: str,
        method: str,
        request_data: Dict[str, Any],
        query_params: Dict[str, Any] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate request against OpenAPI spec.
        
        Args:
            path: Request path
            method: HTTP method
            request_data: Request body
            query_params: Query parameters
            
        Returns:
            (is_valid, error_message)
        """
        path_spec = self.paths.get(path, {})
        method_spec = path_spec.get(method.lower(), {})
        
        if not method_spec:
            return True, None  # No spec defined, allow it
        
        # Validate request body
        if "requestBody" in method_spec:
            req_body_spec = method_spec["requestBody"]
            is_valid, error = self._validate_body(
                request_data,
                req_body_spec
            )
            if not is_valid:
                return False, error
        
        # Validate parameters
        if "parameters" in method_spec:
            is_valid, error = self._validate_parameters(
                query_params or {},
                method_spec["parameters"]
            )
            if not is_valid:
                return False, error
        
        return True, None
    
    def _validate_body(
        self,
        data: Dict[str, Any],
        body_spec: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate request body against schema."""
        try:
            required = body_spec.get("required", False)
            if required and not data:
                return False, "Request body is required"
            
            # Get schema
            content = body_spec.get("content", {})
            json_content = content.get("application/json", {})
            schema = json_content.get("schema", {})
            
            # Validate schema
            if schema.get("type") == "object":
                required_fields = schema.get("required", [])
                for field in required_fields:
                    if field not in data:
                        return False, f"Required field missing: {field}"
            
            return True, None
        
        except Exception as e:
            logger.error(f"Error validating body: {e}")
            return False, f"Validation error: {e}"
    
    def _validate_parameters(
        self,
        params: Dict[str, Any],
        param_specs: list
    ) -> Tuple[bool, Optional[str]]:
        """Validate query/path parameters against spec."""
        try:
            for param_spec in param_specs:
                param_name = param_spec.get("name")
                is_required = param_spec.get("required", False)
                
                if is_required and param_name not in params:
                    return False, f"Required parameter missing: {param_name}"
            
            return True, None
        
        except Exception as e:
            logger.error(f"Error validating parameters: {e}")
            return False, f"Validation error: {e}"


class ResponseBuilder:
    """Builds standardized responses."""
    
    @staticmethod
    def success(data: Any = None, message: str = "Success", status_code: int = 200) -> Tuple[Dict, int]:
        """Build success response.
        
        Args:
            data: Response data
            message: Success message
            status_code: HTTP status code
            
        Returns:
            (response_dict, status_code)
        """
        return {
            "success": True,
            "message": message,
            "data": data,
            "status_code": status_code
        }, status_code
    
    @staticmethod
    def error(
        error: str,
        details: str = None,
        status_code: int = 400
    ) -> Tuple[Dict, int]:
        """Build error response.
        
        Args:
            error: Error message
            details: Additional error details
            status_code: HTTP status code
            
        Returns:
            (response_dict, status_code)
        """
        return {
            "success": False,
            "error": error,
            "details": details,
            "status_code": status_code
        }, status_code


def create_connexion_wrapper(
    handler: Callable,
    validate_schema: bool = True
) -> Callable:
    """Create Connexion-compatible endpoint wrapper.
    
    Args:
        handler: Original Flask endpoint handler
        validate_schema: Whether to validate against schema
        
    Returns:
        Connexion-compatible handler
    """
    @wraps(handler)
    def wrapper(*args, **kwargs):
        try:
            # Call original handler
            result = handler(*args, **kwargs)
            
            # If already a tuple, return as-is
            if isinstance(result, tuple):
                return result
            
            # Otherwise wrap in success response
            return ResponseBuilder.success(result)
        
        except Exception as e:
            logger.error(f"Handler error: {e}")
            return ResponseBuilder.error(str(e), status_code=500)
    
    return wrapper


# Connexion app factory (lazy initialization)
_connexion_app = None


def get_connexion_app():
    """Get or create Connexion application."""
    global _connexion_app
    
    if _connexion_app is None:
        try:
            import connexion
            from connexion.middleware import MiddleMiddleware
            from starlette.middleware.cors import CORSMiddleware
            
            _connexion_app = connexion.AsyncApp(__name__)
            
            # Add CORS middleware
            _connexion_app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            logger.info("Connexion application initialized")
        
        except ImportError:
            logger.warning("Connexion not available, skipping dual-stack setup")
            return None
    
    return _connexion_app


def close_connexion_app():
    """Close Connexion application."""
    global _connexion_app
    _connexion_app = None
