"""
Connexion/OpenAPI Framework Integration (Week 3)

Enables dual-stack (Flask + Connexion) architecture for:
- OpenAPI 3.0 specification-driven API
- Automatic request/response validation
- Interactive API documentation (Swagger UI)
- Gradual migration from Flask to Connexion

Strategy:
1. Flask handles legacy endpoints (Weeks 1-2)
2. Connexion adds new/migrated endpoints with OpenAPI specs
3. Both coexist during transition (Weeks 3-4)
4. Full cutover to Connexion in Week 5
"""

import yaml
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ParameterIn(str, Enum):
    """OpenAPI parameter location."""
    QUERY = "query"
    HEADER = "header"
    PATH = "path"
    COOKIE = "cookie"


class ParameterStyle(str, Enum):
    """OpenAPI parameter style."""
    FORM = "form"
    SIMPLE = "simple"
    MATRIX = "matrix"
    LABEL = "label"
    PIPE_DELIMITED = "pipeDelimited"
    SPACE_DELIMITED = "spaceDelimited"


@dataclass
class Schema:
    """OpenAPI schema definition."""
    type: str  # object, string, number, integer, boolean, array
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    description: str = ""
    example: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAPI schema dict."""
        result = {
            "type": self.type,
        }
        if self.properties:
            result["properties"] = self.properties
        if self.required:
            result["required"] = self.required
        if self.description:
            result["description"] = self.description
        if self.example is not None:
            result["example"] = self.example
        return result


@dataclass
class Parameter:
    """OpenAPI parameter definition."""
    name: str
    param_in: ParameterIn
    description: str = ""
    required: bool = False
    schema: Optional[Schema] = None
    style: ParameterStyle = ParameterStyle.SIMPLE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAPI parameter dict."""
        return {
            "name": self.name,
            "in": self.param_in.value,
            "description": self.description,
            "required": self.required,
            "schema": self.schema.to_dict() if self.schema else {"type": "string"},
            "style": self.style.value,
        }


@dataclass
class Response:
    """OpenAPI response definition."""
    description: str
    content_type: str = "application/json"
    schema: Optional[Schema] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAPI response dict."""
        result = {"description": self.description}
        if self.schema:
            result["content"] = {
                self.content_type: {
                    "schema": self.schema.to_dict()
                }
            }
        return result


@dataclass
class Operation:
    """OpenAPI operation (method) definition."""
    operation_id: str
    summary: str
    description: str = ""
    parameters: List[Parameter] = field(default_factory=list)
    request_body: Optional[Schema] = None
    responses: Dict[int, Response] = field(default_factory=lambda: {
        200: Response("Successful response"),
        400: Response("Bad request"),
        401: Response("Unauthorized"),
        404: Response("Not found"),
        500: Response("Internal server error"),
    })
    tags: List[str] = field(default_factory=list)
    requires_auth: bool = False
    requires_roles: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAPI operation dict."""
        result = {
            "operationId": self.operation_id,
            "summary": self.summary,
            "tags": self.tags or ["default"],
        }
        
        if self.description:
            result["description"] = self.description
        
        if self.parameters:
            result["parameters"] = [p.to_dict() for p in self.parameters]
        
        if self.request_body:
            result["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": self.request_body.to_dict()
                    }
                }
            }
        
        result["responses"] = {
            str(code): resp.to_dict()
            for code, resp in self.responses.items()
        }
        
        # Add security requirement if auth is needed
        if self.requires_auth:
            result["security"] = [{"BearerAuth": self.requires_roles or []}]
        
        return result


@dataclass
class Path:
    """OpenAPI path definition."""
    path: str
    get: Optional[Operation] = None
    post: Optional[Operation] = None
    put: Optional[Operation] = None
    delete: Optional[Operation] = None
    patch: Optional[Operation] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAPI path dict."""
        result = {}
        for method_name in ["get", "post", "put", "delete", "patch"]:
            method_op = getattr(self, method_name)
            if method_op:
                result[method_name] = method_op.to_dict()
        return result


class OpenAPISpec:
    """OpenAPI 3.0 specification builder."""
    
    def __init__(
        self,
        title: str = "INIDS API",
        version: str = "1.0.0",
        description: str = "",
        base_path: str = "/api"
    ):
        """Initialize OpenAPI spec.
        
        Args:
            title: API title
            version: API version
            description: API description
            base_path: Base path for all endpoints
        """
        self.title = title
        self.version = version
        self.description = description
        self.base_path = base_path
        self.paths: Dict[str, Path] = {}
        self.schemas: Dict[str, Schema] = {}
        self.tags: Dict[str, str] = {}
    
    def add_path(self, path: Path):
        """Add path to spec."""
        self.paths[f"{self.base_path}{path.path}"] = path
    
    def add_schema(self, name: str, schema: Schema):
        """Add reusable schema definition."""
        self.schemas[name] = schema
    
    def add_tag(self, name: str, description: str):
        """Add tag definition."""
        self.tags[name] = description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAPI dict."""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": self.title,
                "version": self.version,
            },
            "servers": [
                {"url": self.base_path, "description": "API Base"}
            ],
            "paths": {
                path_key: path.to_dict()
                for path_key, path in self.paths.items()
            },
        }
        
        if self.description:
            spec["info"]["description"] = self.description
        
        if self.schemas:
            spec["components"] = {
                "schemas": {
                    name: schema.to_dict()
                    for name, schema in self.schemas.items()
                },
                "securitySchemes": {
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT",
                        "description": "JWT token authentication"
                    }
                }
            }
        
        if self.tags:
            spec["tags"] = [
                {"name": name, "description": desc}
                for name, desc in self.tags.items()
            ]
        
        return spec
    
    def to_yaml(self) -> str:
        """Export to YAML format."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def to_json(self) -> str:
        """Export to JSON format."""
        import json
        return json.dumps(self.to_dict(), indent=2)


# Pre-built OpenAPI specs for INIDS endpoints
def create_auth_api_spec() -> OpenAPISpec:
    """Create OpenAPI spec for authentication endpoints."""
    spec = OpenAPISpec(
        title="INIDS Authentication API",
        version="1.0.0",
        description="JWT token generation and validation",
        base_path="/api/auth"
    )
    
    spec.add_tag("auth", "Authentication endpoints")
    
    # POST /login
    login_op = Operation(
        operation_id="login",
        summary="Login and get JWT token",
        description="Generate JWT token for authenticated user",
        tags=["auth"],
        request_body=Schema(
            type="object",
            properties={
                "username": {"type": "string", "description": "Username"},
                "password": {"type": "string", "description": "Password"},
                "roles": {"type": "array", "items": {"type": "string"}, "description": "Assigned roles"}
            },
            required=["username"],
            example={"username": "admin", "password": "secret", "roles": ["admin"]}
        ),
        responses={
            200: Response(
                "Token generated successfully",
                schema=Schema(
                    type="object",
                    properties={
                        "token": {"type": "string"},
                        "expires_in": {"type": "integer"},
                        "user": {"type": "string"},
                        "roles": {"type": "array", "items": {"type": "string"}}
                    }
                )
            ),
            401: Response("Authentication failed"),
        }
    )
    
    spec.add_path(Path(
        path="/login",
        post=login_op
    ))
    
    # GET /validate
    validate_op = Operation(
        operation_id="validate",
        summary="Validate JWT token",
        description="Check if current JWT token is valid",
        tags=["auth"],
        requires_auth=True,
        responses={
            200: Response(
                "Token is valid",
                schema=Schema(
                    type="object",
                    properties={
                        "valid": {"type": "boolean"},
                        "user": {"type": "string"},
                        "roles": {"type": "array", "items": {"type": "string"}},
                        "expires_at": {"type": "string"}
                    }
                )
            ),
            401: Response("Invalid or expired token"),
        }
    )
    
    spec.add_path(Path(
        path="/validate",
        get=validate_op
    ))
    
    return spec


def create_detection_api_spec() -> OpenAPISpec:
    """Create OpenAPI spec for detection endpoints."""
    spec = OpenAPISpec(
        title="INIDS Detection API",
        version="1.0.0",
        description="Network intrusion detection and prediction",
        base_path="/api"
    )
    
    spec.add_tag("detection", "Detection and prediction endpoints")
    spec.add_tag("health", "Health check endpoints")
    
    # POST /predict
    predict_op = Operation(
        operation_id="predict",
        summary="Predict threat level",
        description="Analyze network flow and predict threat level",
        tags=["detection"],
        requires_auth=True,
        requires_roles=["analyst", "admin"],
        request_body=Schema(
            type="object",
            properties={
                "features": {"type": "object", "description": "Network flow features"},
                "profile": {"type": "string", "description": "Detection profile"},
                "source_ip": {"type": "string", "description": "Source IP address"},
                "attack_type": {"type": "string", "description": "Expected attack type"}
            },
            required=["features"],
            example={
                "features": {"duration": 10.5, "src_bytes": 1024},
                "profile": "balanced",
                "source_ip": "192.168.1.100"
            }
        ),
        responses={
            200: Response(
                "Prediction result",
                schema=Schema(
                    type="object",
                    properties={
                        "risk_score": {"type": "number"},
                        "threat_type": {"type": "string"},
                        "confidence": {"type": "number"},
                        "engines": {"type": "object"}
                    }
                )
            ),
            400: Response("Invalid request"),
            503: Response("No model available"),
        }
    )
    
    spec.add_path(Path(
        path="/predict",
        post=predict_op
    ))
    
    # GET /health
    health_op = Operation(
        operation_id="health_check",
        summary="System health check",
        description="Get current system health status",
        tags=["health"],
        responses={
            200: Response(
                "System is healthy",
                schema=Schema(
                    type="object",
                    properties={
                        "status": {"type": "string"},
                        "timestamp": {"type": "string"},
                        "model_status": {"type": "string"}
                    }
                )
            ),
            503: Response("System is unhealthy"),
        }
    )
    
    spec.add_path(Path(
        path="/health",
        get=health_op
    ))
    
    return spec


def create_audit_api_spec() -> OpenAPISpec:
    """Create OpenAPI spec for audit endpoints."""
    spec = OpenAPISpec(
        title="INIDS Audit API",
        version="1.0.0",
        description="Audit logs and activity tracking",
        base_path="/api/audit"
    )
    
    spec.add_tag("audit", "Audit and logging endpoints")
    
    # GET /logs
    logs_op = Operation(
        operation_id="get_audit_logs",
        summary="Get audit logs",
        description="Retrieve system audit logs with optional filtering",
        tags=["audit"],
        requires_auth=True,
        parameters=[
            Parameter(
                name="limit",
                param_in=ParameterIn.QUERY,
                description="Maximum number of logs to return",
                schema=Schema(type="integer", example=100)
            ),
            Parameter(
                name="user",
                param_in=ParameterIn.QUERY,
                description="Filter by user",
                schema=Schema(type="string")
            ),
        ],
        responses={
            200: Response(
                "List of audit logs",
                schema=Schema(
                    type="array",
                    properties={},
                    example=[{
                        "timestamp": "2026-04-15T10:35:42Z",
                        "user": "admin",
                        "method": "POST",
                        "path": "/api/predict",
                        "status": 200
                    }]
                )
            ),
        }
    )
    
    spec.add_path(Path(
        path="/logs",
        get=logs_op
    ))
    
    return spec


# Registry of all API specs
OPENAPI_SPECS = {
    "auth": create_auth_api_spec(),
    "detection": create_detection_api_spec(),
    "audit": create_audit_api_spec(),
}


def get_combined_openapi_spec() -> OpenAPISpec:
    """Get combined OpenAPI spec from all modules."""
    combined = OpenAPISpec(
        title="INIDS Platform API",
        version="2.0.0",
        description="Enterprise AI-Powered Network Intrusion Detection & Prevention System",
        base_path="/api"
    )
    
    # Merge all spec paths and schemas
    for spec_name, spec in OPENAPI_SPECS.items():
        for path_key, path in spec.paths.items():
            # Remove base path from key to avoid duplication
            clean_key = path_key.replace(spec.base_path, "/api")
            combined.paths[clean_key] = path
        
        for schema_name, schema in spec.schemas.items():
            combined.add_schema(f"{spec_name}_{schema_name}", schema)
        
        for tag_name, tag_desc in spec.tags.items():
            combined.add_tag(tag_name, tag_desc)
    
    return combined
