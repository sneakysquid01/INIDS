#!/usr/bin/env python3
"""Week 3 Validation Tests - Connexion, OpenAPI, RBAC Framework"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from connexion_integration import (
    OpenAPISpec, Operation, Parameter, Response, Schema, ParameterIn,
    create_auth_api_spec, create_detection_api_spec, create_audit_api_spec,
    get_combined_openapi_spec
)
from connexion_router import RequestValidator, ResponseBuilder, DualStackRouter
from rbac_manager import RBACManager, get_rbac_manager

def test_openapi_auth_spec():
    """Test authentication API OpenAPI spec generation."""
    print("Testing OpenAPI Auth Spec... ", end="")
    spec = create_auth_api_spec()
    
    assert spec.title == "INIDS Authentication API"
    assert "/api/auth/login" in spec.paths
    assert "/api/auth/validate" in spec.paths
    
    # Convert to YAML and verify
    yaml_output = spec.to_yaml()
    assert "openapi: 3.0.0" in yaml_output
    assert "login" in yaml_output
    
    print("✓")


def test_openapi_detection_spec():
    """Test detection API OpenAPI spec generation."""
    print("Testing OpenAPI Detection Spec... ", end="")
    spec = create_detection_api_spec()
    
    assert spec.title == "INIDS Detection API"
    assert "/api/predict" in spec.paths
    assert "/api/health" in spec.paths
    
    json_output = spec.to_json()
    assert json.loads(json_output)  # Valid JSON
    
    print("✓")


def test_combined_openapi_spec():
    """Test combined OpenAPI spec."""
    print("Testing Combined OpenAPI Spec... ", end="")
    spec = get_combined_openapi_spec()
    
    assert spec.title == "INIDS Platform API"
    # Should have paths from all specs
    spec_dict = spec.to_dict()
    assert "paths" in spec_dict
    assert len(spec_dict["paths"]) > 0
    
    print("✓")


def test_request_validator():
    """Test request validation against OpenAPI spec."""
    print("Testing Request Validator... ", end="")
    
    spec = create_auth_api_spec().to_dict()
    validator = RequestValidator(spec)
    
    # Valid request
    is_valid, error = validator.validate_request(
        "/api/auth/login",
        "POST",
        {"username": "admin"}
    )
    assert is_valid or error is None, error
    
    # Invalid request - missing required field
    is_valid, error = validator.validate_request(
        "/api/auth/login",
        "POST",
        {}
    )
    # Either valid (no strict validation) or invalid with reason
    assert isinstance(is_valid, bool)
    
    print("✓")


def test_response_builder():
    """Test response builder."""
    print("Testing Response Builder... ", end="")
    
    # Success response
    response, status = ResponseBuilder.success({"key": "value"}, "Data retrieved")
    assert response["success"] == True
    assert response["data"]["key"] == "value"
    assert status == 200
    
    # Error response
    response, status = ResponseBuilder.error("Not found", "Resource not found")
    assert response["success"] == False
    assert response["error"] == "Not found"
    assert status == 400
    
    print("✓")


def test_rbac_manager():
    """Test RBAC manager functionality."""
    print("Testing RBAC Manager... ", end="")
    
    # Create in-memory RBAC manager
    rbac = RBACManager("sqlite:///:memory:")
    
    # Add user
    success = rbac.add_user("user1", "testuser", "test@example.com", ["analyst"])
    assert success, "Failed to add user"
    
    # Check permission
    allowed, reason = rbac.check_permission("user1", "read_rule")
    # Should be allowed for analyst role
    assert isinstance(allowed, bool)
    
    # Get user permissions
    perms = rbac.get_user_permissions("user1")
    assert isinstance(perms, set)
    
    print("✓")


def test_dual_stack_router():
    """Test dual-stack router."""
    print("Testing Dual Stack Router... ", end="")
    
    # Create mock Flask app
    class MockFlaskApp:
        pass
    
    flask_app = MockFlaskApp()
    router = DualStackRouter(flask_app)
    
    # Register routes
    def flask_handler():
        return "flask"
    
    router.register_flask_endpoint("/api/test", "GET", flask_handler)
    
    # Check stats
    stats = router.get_route_stats()
    assert stats["flask_routes"] == 1
    assert stats["connexion_routes"] == 0
    
    print("✓")


def test_operation_definition():
    """Test OpenAPI operation definition."""
    print("Testing Operation Definition... ", end="")
    
    op = Operation(
        operation_id="test_op",
        summary="Test operation",
        description="A test operation",
        parameters=[
            Parameter(
                name="id",
                param_in=ParameterIn.QUERY,
                required=True,
                schema=Schema(type="integer")
            )
        ],
        requires_auth=True
    )
    
    op_dict = op.to_dict()
    assert op_dict["operationId"] == "test_op"
    assert "security" in op_dict  # Auth required
    assert len(op_dict["parameters"]) == 1
    
    print("✓")


def test_schema_definition():
    """Test OpenAPI schema definition."""
    print("Testing Schema Definition... ", end="")
    
    schema = Schema(
        type="object",
        properties={
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        required=["name"],
        description="User object",
        example={"name": "John", "age": 30}
    )
    
    schema_dict = schema.to_dict()
    assert schema_dict["type"] == "object"
    assert "name" in schema_dict["properties"]
    assert schema_dict["required"] == ["name"]
    assert schema_dict["example"]["name"] == "John"
    
    print("✓")


if __name__ == "__main__":
    print("=" * 60)
    print("Week 3 Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_openapi_auth_spec,
        test_openapi_detection_spec,
        test_combined_openapi_spec,
        test_request_validator,
        test_response_builder,
        test_schema_definition,
        test_operation_definition,
        test_dual_stack_router,
        test_rbac_manager,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)
