#!/usr/bin/env python3
"""Week 7 Validation Tests - Production Hardening & Reliability"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from production_hardening import (
    SecurityLevel, SecurityAuditLog, PerformanceMetric,
    SecurityHardeningManager, PerformanceOptimizer, ReliabilityManager,
    CircuitBreaker
)


def test_security_audit_log():
    """Test SecurityAuditLog dataclass."""
    print("Testing SecurityAuditLog... ", end="")
    
    log = SecurityAuditLog(
        timestamp=datetime.now(timezone.utc),
        event_type="login",
        user_id="user-123",
        resource="admin_panel",
        action="access",
        result="success",
        details={"ip": "192.168.1.1"}
    )
    
    assert log.event_type == "login"
    assert log.user_id == "user-123"
    assert log.result == "success"
    
    print("✓")


def test_performance_metric():
    """Test PerformanceMetric dataclass."""
    print("Testing PerformanceMetric... ", end="")
    
    metric = PerformanceMetric(
        timestamp=datetime.now(timezone.utc),
        operation="detect_anomaly",
        duration_ms=45.2,
        memory_usage_mb=128.5,
        cpu_percent=25.3,
        success=True,
        details={"model": "v1.0"}
    )
    
    assert metric.operation == "detect_anomaly"
    assert metric.duration_ms == 45.2
    assert metric.success == True
    
    print("✓")


def test_security_level_enum():
    """Test SecurityLevel enumeration."""
    print("Testing SecurityLevel Enum... ", end="")
    
    assert SecurityLevel.LOW.value == "low"
    assert SecurityLevel.MEDIUM.value == "medium"
    assert SecurityLevel.HIGH.value == "high"
    assert SecurityLevel.CRITICAL.value == "critical"
    
    print("✓")


def test_security_input_validation():
    """Test input validation."""
    print("Testing Input Validation... ", end="")
    
    manager = SecurityHardeningManager()
    
    # Valid inputs
    assert manager.validate_input("hello", "string", 1000) == True
    assert manager.validate_input("user@example.com", "email", 1000) == True
    assert manager.validate_input("192.168.1.1", "ip", 1000) == True
    assert manager.validate_input("https://example.com", "url", 1000) == True
    
    # Invalid inputs
    assert manager.validate_input("a" * 2000, "string", 1000) == False
    assert manager.validate_input("invalid", "email", 1000) == False
    assert manager.validate_input("256.0.0.1", "ip", 1000) == False
    
    print("✓")


def test_security_secrets_management():
    """Test secrets management."""
    print("Testing Secrets Management... ", end="")
    
    manager = SecurityHardeningManager()
    
    # Store secret
    assert manager.store_secret("db_password", "super_secret_123") == True
    
    # Retrieve secret
    secret = manager.retrieve_secret("db_password")
    assert secret is not None
    
    # Non-existent secret
    assert manager.retrieve_secret("non_existent") is None
    
    print("✓")


def test_security_audit_logging():
    """Test audit event logging."""
    print("Testing Audit Logging... ", end="")
    
    manager = SecurityHardeningManager()
    
    # Log event
    assert manager.log_audit_event(
        event_type="modification",
        user_id="admin-1",
        resource="config",
        action="update",
        result="success",
        details={"field": "threshold"}
    ) == True
    
    assert len(manager.audit_logs) == 1
    assert manager.audit_logs[0].event_type == "modification"
    
    print("✓")


def test_security_ip_blocking():
    """Test IP blocking."""
    print("Testing IP Blocking... ", end="")
    
    manager = SecurityHardeningManager()
    
    # Block IP
    assert manager.block_ip("192.168.1.100", "suspicious activity") == True
    assert manager.is_ip_blocked("192.168.1.100") == True
    assert manager.is_ip_blocked("192.168.1.101") == False
    
    print("✓")


def test_security_rate_limiting():
    """Test rate limiting."""
    print("Testing Rate Limiting... ", end="")
    
    manager = SecurityHardeningManager()
    
    # Within limit
    for _ in range(5):
        assert manager.enforce_rate_limit("client-1", 10) == True
    
    # Exceed limit
    for _ in range(5):
        manager.enforce_rate_limit("client-1", 10)
    
    assert manager.enforce_rate_limit("client-1", 10) == False
    
    print("✓")


def test_audit_log_retrieval():
    """Test audit log retrieval and filtering."""
    print("Testing Audit Log Retrieval... ", end="")
    
    manager = SecurityHardeningManager()
    
    # Log multiple events
    for i in range(5):
        manager.log_audit_event(
            event_type="login" if i % 2 == 0 else "access",
            user_id=f"user-{i}",
            resource="resource",
            action="action",
            result="success"
        )
    
    # Get all logs
    logs = manager.get_audit_logs()
    assert len(logs) == 5
    
    # Filter by event type
    login_logs = manager.get_audit_logs(event_type="login")
    assert len(login_logs) >= 2
    
    # Filter by user
    user_logs = manager.get_audit_logs(user_id="user-0")
    assert len(user_logs) >= 1
    
    print("✓")


def test_performance_recording():
    """Test performance metric recording."""
    print("Testing Performance Recording... ", end="")
    
    optimizer = PerformanceOptimizer()
    
    # Record metrics
    optimizer.record_metric("detect", 50.5, 128.0, 25.0, True)
    optimizer.record_metric("train", 150.2, 256.0, 75.0, True)
    
    assert len(optimizer.metrics) == 2
    assert optimizer.metrics[0].operation == "detect"
    
    print("✓")


def test_caching():
    """Test caching functionality."""
    print("Testing Caching... ", end="")
    
    optimizer = PerformanceOptimizer()
    
    # Set cache
    assert optimizer.cache_set("key1", "value1", ttl_minutes=1) == True
    
    # Get cache
    assert optimizer.cache_get("key1") == "value1"
    
    # Non-existent key
    assert optimizer.cache_get("key2") is None
    
    print("✓")


def test_performance_summary():
    """Test performance summary generation."""
    print("Testing Performance Summary... ", end="")
    
    optimizer = PerformanceOptimizer()
    
    # Record metrics
    optimizer.record_metric("op1", 50.0, 100.0, 25.0, True)
    optimizer.record_metric("op2", 60.0, 110.0, 30.0, True)
    optimizer.record_metric("op3", 40.0, 90.0, 20.0, False)
    
    summary = optimizer.get_performance_summary()
    
    assert summary["total_operations"] == 3
    assert summary["success_rate_percent"] == 66.67
    
    print("✓")


def test_circuit_breaker_closed():
    """Test circuit breaker in closed state."""
    print("Testing Circuit Breaker Closed... ", end="")
    
    breaker = CircuitBreaker("service-1", failure_threshold=3)
    
    # Circuit should be available
    assert breaker.is_available() == True
    assert breaker.state == "closed"
    
    # Record success
    breaker.record_success()
    assert breaker.is_available() == True
    
    print("✓")


def test_circuit_breaker_open():
    """Test circuit breaker opening."""
    print("Testing Circuit Breaker Open... ", end="")
    
    breaker = CircuitBreaker("service-2", failure_threshold=3)
    
    # Record failures
    for _ in range(3):
        breaker.record_failure()
    
    # Circuit should be open
    assert breaker.state == "open"
    assert breaker.is_available() == False
    
    print("✓")


def test_circuit_breaker_recovery():
    """Test circuit breaker recovery."""
    print("Testing Circuit Breaker Recovery... ", end="")
    
    breaker = CircuitBreaker("service-3", failure_threshold=2, recovery_timeout_sec=0)
    
    # Open circuit
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "open"
    
    # After recovery timeout, should transition to half-open
    import time
    time.sleep(0.1)
    assert breaker.is_available() == True
    assert breaker.state == "half-open"
    
    print("✓")


def test_reliability_manager_circuit_breaker():
    """Test reliability manager circuit breaker registration."""
    print("Testing Reliability Manager Circuit Breaker... ", end="")
    
    manager = ReliabilityManager()
    
    # Register circuit breaker
    breaker = manager.register_circuit_breaker(
        "db_service",
        failure_threshold=5,
        recovery_timeout_sec=60
    )
    
    assert "db_service" in manager.circuit_breakers
    assert breaker.name == "db_service"
    
    print("✓")


if __name__ == "__main__":
    print("=" * 60)
    print("Week 7 Validation Tests - Production Hardening")
    print("=" * 60)
    print()
    
    tests = [
        test_security_audit_log,
        test_performance_metric,
        test_security_level_enum,
        test_security_input_validation,
        test_security_secrets_management,
        test_security_audit_logging,
        test_security_ip_blocking,
        test_security_rate_limiting,
        test_audit_log_retrieval,
        test_performance_recording,
        test_caching,
        test_performance_summary,
        test_circuit_breaker_closed,
        test_circuit_breaker_open,
        test_circuit_breaker_recovery,
        test_reliability_manager_circuit_breaker,
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
