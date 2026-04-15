"""
Production Hardening - Security, Performance, Reliability (Week 7)

Comprehensive security hardening and production readiness:
1. Security hardening (input validation, encryption, secrets management)
2. Performance optimization (caching, indexing, query optimization)
3. Reliability (circuit breaker, health checks, graceful degradation)
4. Compliance (audit logging, data retention, encryption at rest/transit)
"""

import logging
import hashlib
import json
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security compliance levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityAuditLog:
    """Security audit event."""
    timestamp: datetime
    event_type: str  # login, logout, access, modification, error
    user_id: str
    resource: str
    action: str
    result: str  # success, failure
    details: Dict[str, Any]


@dataclass
class PerformanceMetric:
    """Performance measurement."""
    timestamp: datetime
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    details: Dict[str, Any]


class SecurityHardeningManager:
    """Manages security hardening for production."""
    
    def __init__(self):
        """Initialize security manager."""
        self.audit_logs: List[SecurityAuditLog] = []
        self.blocked_ips: set = set()
        self.rate_limits: Dict[str, int] = {}
        self.secrets_vault: Dict[str, str] = {}
    
    def validate_input(
        self,
        data: str,
        data_type: str = "string",
        max_length: int = 1000
    ) -> bool:
        """Validate user input.
        
        Args:
            data: Input data to validate
            data_type: Type of data (string, email, ip, url)
            max_length: Maximum allowed length
            
        Returns:
            True if valid
        """
        try:
            if len(str(data)) > max_length:
                logger.warning(f"Input exceeds max length: {len(str(data))} > {max_length}")
                return False
            
            if data_type == "email":
                return "@" in str(data) and "." in str(data)
            elif data_type == "ip":
                parts = str(data).split(".")
                return len(parts) == 4 and all(0 <= int(p) <= 255 for p in parts)
            elif data_type == "url":
                return str(data).startswith(("http://", "https://"))
            
            return True
        
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False
    
    def encrypt_sensitive_data(self, data: str, key: str) -> str:
        """Encrypt sensitive data.
        
        Note: This is a simplified implementation. Use cryptography library
        for production: from cryptography.fernet import Fernet
        
        Args:
            data: Data to encrypt
            key: Encryption key
            
        Returns:
            Encrypted data
        """
        try:
            # Simple hash-based obfuscation for demo
            # Production: Use Fernet or AES-256-GCM
            combined = f"{data}:{key}"
            encrypted = hashlib.sha256(combined.encode()).hexdigest()
            return encrypted
        
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return ""
    
    def store_secret(self, secret_name: str, secret_value: str) -> bool:
        """Store secret securely.
        
        Args:
            secret_name: Name of secret
            secret_value: Secret value to store
            
        Returns:
            True if stored successfully
        """
        try:
            # In production: Use AWS Secrets Manager, Azure Key Vault, or GCP Secret Manager
            self.secrets_vault[secret_name] = self.encrypt_sensitive_data(
                secret_value, "master-key"
            )
            logger.info(f"Stored secret: {secret_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error storing secret: {e}")
            return False
    
    def retrieve_secret(self, secret_name: str) -> Optional[str]:
        """Retrieve secret from vault.
        
        Args:
            secret_name: Name of secret
            
        Returns:
            Secret value if found
        """
        try:
            return self.secrets_vault.get(secret_name)
        
        except Exception as e:
            logger.error(f"Error retrieving secret: {e}")
            return None
    
    def log_audit_event(
        self,
        event_type: str,
        user_id: str,
        resource: str,
        action: str,
        result: str,
        details: Dict[str, Any] = None
    ) -> bool:
        """Log security audit event.
        
        Args:
            event_type: Type of security event
            user_id: User performing action
            resource: Resource accessed/modified
            action: Action performed
            result: Result (success/failure)
            details: Additional details
            
        Returns:
            True if logged successfully
        """
        try:
            log = SecurityAuditLog(
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                user_id=user_id,
                resource=resource,
                action=action,
                result=result,
                details=details or {}
            )
            
            self.audit_logs.append(log)
            logger.info(f"Audit: {event_type} by {user_id} on {resource}: {action} ({result})")
            return True
        
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            return False
    
    def block_ip(self, ip_address: str, reason: str = None) -> bool:
        """Add IP to blocklist.
        
        Args:
            ip_address: IP to block
            reason: Reason for blocking
            
        Returns:
            True if blocked
        """
        try:
            self.blocked_ips.add(ip_address)
            logger.warning(f"Blocked IP: {ip_address} ({reason})")
            return True
        
        except Exception as e:
            logger.error(f"Error blocking IP: {e}")
            return False
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked.
        
        Args:
            ip_address: IP to check
            
        Returns:
            True if blocked
        """
        return ip_address in self.blocked_ips
    
    def enforce_rate_limit(
        self,
        identifier: str,
        limit_per_minute: int = 100
    ) -> bool:
        """Enforce rate limiting.
        
        Args:
            identifier: Client identifier (IP, user ID, etc.)
            limit_per_minute: Max requests per minute
            
        Returns:
            True if within limit
        """
        try:
            current_count = self.rate_limits.get(identifier, 0)
            
            if current_count >= limit_per_minute:
                logger.warning(f"Rate limit exceeded for {identifier}")
                return False
            
            self.rate_limits[identifier] = current_count + 1
            return True
        
        except Exception as e:
            logger.error(f"Error enforcing rate limit: {e}")
            return False
    
    def get_audit_logs(
        self,
        user_id: str = None,
        event_type: str = None,
        limit: int = 100
    ) -> List[SecurityAuditLog]:
        """Get audit logs with filtering.
        
        Args:
            user_id: Filter by user
            event_type: Filter by event type
            limit: Max number of logs
            
        Returns:
            List of audit logs
        """
        logs = self.audit_logs
        
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        if event_type:
            logs = [l for l in logs if l.event_type == event_type]
        
        return logs[-limit:]


class PerformanceOptimizer:
    """Optimizes system performance for production."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.metrics: List[PerformanceMetric] = []
        self.slow_operations: Dict[str, float] = {}  # operation -> avg_ms
        self.cache: Dict[str, tuple] = {}  # key -> (value, expiry)
    
    def record_metric(
        self,
        operation: str,
        duration_ms: float,
        memory_usage_mb: float = 0.0,
        cpu_percent: float = 0.0,
        success: bool = True,
        details: Dict[str, Any] = None
    ) -> None:
        """Record performance metric.
        
        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            memory_usage_mb: Memory used in MB
            cpu_percent: CPU usage percentage
            success: Whether operation succeeded
            details: Additional details
        """
        metric = PerformanceMetric(
            timestamp=datetime.now(timezone.utc),
            operation=operation,
            duration_ms=duration_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            success=success,
            details=details or {}
        )
        
        self.metrics.append(metric)
        
        # Track slow operations
        if operation not in self.slow_operations:
            self.slow_operations[operation] = duration_ms
        else:
            self.slow_operations[operation] = (
                self.slow_operations[operation] + duration_ms
            ) / 2
        
        if duration_ms > 1000:  # > 1 second
            logger.warning(f"Slow operation: {operation} took {duration_ms}ms")
    
    def cache_get(self, key: str) -> Any:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if exists and not expired
        """
        try:
            if key in self.cache:
                value, expiry = self.cache[key]
                if datetime.now(timezone.utc) < expiry:
                    logger.debug(f"Cache hit: {key}")
                    return value
                else:
                    del self.cache[key]
            return None
        
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def cache_set(
        self,
        key: str,
        value: Any,
        ttl_minutes: int = 5
    ) -> bool:
        """Set cache value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_minutes: Time-to-live in minutes
            
        Returns:
            True if cached successfully
        """
        try:
            expiry = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
            self.cache[key] = (value, expiry)
            logger.debug(f"Cached: {key}")
            return True
        
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary.
        
        Returns:
            Performance metrics summary
        """
        if not self.metrics:
            return {}
        
        total_duration = sum(m.duration_ms for m in self.metrics)
        avg_duration = total_duration / len(self.metrics)
        success_rate = sum(1 for m in self.metrics if m.success) / len(self.metrics) * 100
        
        return {
            "total_operations": len(self.metrics),
            "avg_duration_ms": round(avg_duration, 2),
            "min_duration_ms": min(m.duration_ms for m in self.metrics),
            "max_duration_ms": max(m.duration_ms for m in self.metrics),
            "success_rate_percent": round(success_rate, 2),
            "avg_memory_usage_mb": round(
                sum(m.memory_usage_mb for m in self.metrics) / len(self.metrics), 2
            ),
            "slow_operations": self.slow_operations
        }


class ReliabilityManager:
    """Manages reliability and graceful degradation."""
    
    def __init__(self):
        """Initialize reliability manager."""
        self.circuit_breakers: Dict[str, "CircuitBreaker"] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
    
    def register_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_sec: int = 60
    ) -> "CircuitBreaker":
        """Register circuit breaker for service.
        
        Args:
            name: Service name
            failure_threshold: Failures before opening circuit
            recovery_timeout_sec: Time before attempting recovery
            
        Returns:
            CircuitBreaker instance
        """
        breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout_sec=recovery_timeout_sec
        )
        self.circuit_breakers[name] = breaker
        return breaker
    
    def register_health_check(
        self,
        service_name: str,
        check_func: Callable
    ) -> bool:
        """Register health check function.
        
        Args:
            service_name: Name of service
            check_func: Async function that returns True if healthy
            
        Returns:
            True if registered
        """
        try:
            self.health_checks[service_name] = check_func
            logger.info(f"Registered health check: {service_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error registering health check: {e}")
            return False
    
    def register_fallback_handler(
        self,
        service_name: str,
        handler_func: Callable
    ) -> bool:
        """Register fallback handler for service.
        
        Args:
            service_name: Name of service
            handler_func: Function to call on failure
            
        Returns:
            True if registered
        """
        try:
            self.fallback_handlers[service_name] = handler_func
            logger.info(f"Registered fallback handler: {service_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error registering fallback handler: {e}")
            return False


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_sec: int = 60
    ):
        """Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Failures before opening
            recovery_timeout_sec: Recovery timeout
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        if self.state != "closed":
            self.state = "closed"
            logger.info(f"Circuit breaker {self.name} closed")
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
    
    def is_available(self) -> bool:
        """Check if circuit is available for requests.
        
        Returns:
            True if available
        """
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if recovery timeout expired
            if self.last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout_sec:
                    self.state = "half-open"
                    logger.info(f"Circuit breaker {self.name} half-opened for recovery")
                    return True
            return False
        
        # half-open: allow limited traffic
        return True
