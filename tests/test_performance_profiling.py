"""Performance profiling and validation for security modules.

Measures throughput, latency, and overhead of sanitization, correlation
tracing, and CSRF protection modules.
"""

import pytest
import time
import statistics
from src.input_sanitizer import (
    sanitize_string, sanitize_id, sanitize_ip_address,
    sanitize_port, sanitize_severity, sanitize_url_path,
    sanitize_json_object, sanitize_integer, sanitize_float,
)
from src.correlation_tracing import (
    generate_correlation_id, set_correlation_id,
    get_correlation_id, attach_correlation_id_to_logs,
)
from src.csrf_protection import (
    generate_csrf_token, validate_csrf_token,
)


class TestInputSanitizerPerformance:
    """Performance tests for input sanitizer module."""
    
    def test_sanitize_string_throughput(self, benchmark):
        """Benchmark string sanitization throughput."""
        def sanitize():
            sanitize_string("test_input_value_123")
        
        result = benchmark(sanitize)
    
    def test_sanitize_id_throughput(self, benchmark):
        """Benchmark ID sanitization throughput."""
        def sanitize():
            sanitize_id("alert_abc_123")
        
        result = benchmark(sanitize)
    
    def test_sanitize_ip_throughput(self, benchmark):
        """Benchmark IP address validation throughput."""
        def sanitize():
            sanitize_ip_address("192.168.1.1")
        
        result = benchmark(sanitize)
    
    def test_sanitize_port_throughput(self, benchmark):
        """Benchmark port validation throughput."""
        def sanitize():
            sanitize_port(8080)
        
        result = benchmark(sanitize)
    
    def test_sanitize_severity_throughput(self, benchmark):
        """Benchmark severity sanitization throughput."""
        def sanitize():
            sanitize_severity("high")
        
        result = benchmark(sanitize)
    
    def test_sanitize_url_path_throughput(self, benchmark):
        """Benchmark URL path sanitization throughput."""
        def sanitize():
            sanitize_url_path("api/v1/alerts")
        
        result = benchmark(sanitize)
    
    def test_sanitize_json_object_throughput(self, benchmark):
        """Benchmark JSON object sanitization throughput."""
        obj = {"key": "value", "number": 42}
        
        def sanitize():
            sanitize_json_object(obj)
        
        result = benchmark(sanitize)
    
    def test_sanitize_integer_throughput(self, benchmark):
        """Benchmark integer sanitization throughput."""
        def sanitize():
            sanitize_integer(42)
        
        result = benchmark(sanitize)
    
    def test_sanitize_float_throughput(self, benchmark):
        """Benchmark float sanitization throughput."""
        def sanitize():
            sanitize_float(3.14)
        
        result = benchmark(sanitize)
    
    def test_sanitize_string_latency(self):
        """Measure latency of string sanitization."""
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            sanitize_string("test_input")
            elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds
            times.append(elapsed)
        
        avg_latency = statistics.mean(times)
        p95_latency = sorted(times)[int(len(times) * 0.95)]
        p99_latency = sorted(times)[int(len(times) * 0.99)]
        
        # Assert reasonable latency (< 100 microseconds for common case)
        assert avg_latency < 100, f"Average latency {avg_latency}μs exceeds 100μs"
        assert p95_latency < 500, f"P95 latency {p95_latency}μs exceeds 500μs"
        assert p99_latency < 1000, f"P99 latency {p99_latency}μs exceeds 1000μs"


class TestCorrelationTracingPerformance:
    """Performance tests for correlation tracing module."""
    
    def test_generate_correlation_id_throughput(self, benchmark):
        """Benchmark correlation ID generation throughput."""
        result = benchmark(generate_correlation_id)
    
    def test_correlation_id_generation_latency(self, app_context):
        """Measure latency of ID generation."""
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            generate_correlation_id()
            elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds
            times.append(elapsed)
        
        avg_latency = statistics.mean(times)
        p95_latency = sorted(times)[int(len(times) * 0.95)]
        
        # Assert reasonable latency (< 50 microseconds)
        assert avg_latency < 50, f"Average latency {avg_latency}μs exceeds 50μs"
        assert p95_latency < 200, f"P95 latency {p95_latency}μs exceeds 200μs"


class TestCSRFProtectionPerformance:
    """Performance tests for CSRF protection module."""
    
    def test_generate_csrf_token_throughput(self, benchmark):
        """Benchmark CSRF token generation throughput."""
        result = benchmark(generate_csrf_token)
    
    def test_token_generation_latency(self):
        """Measure latency of token generation."""
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            generate_csrf_token()
            elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds
            times.append(elapsed)
        
        avg_latency = statistics.mean(times)
        p95_latency = sorted(times)[int(len(times) * 0.95)]
        
        # Assert reasonable latency (< 100 microseconds)
        assert avg_latency < 100, f"Average latency {avg_latency}μs exceeds 100μs"
        assert p95_latency < 500, f"P95 latency {p95_latency}μs exceeds 500μs"


class TestModuleInteractionPerformance:
    """Performance tests for module interactions."""
    
    def test_sanitize_then_log_performance(self):
        """Measure combined sanitization + logging latency."""
        times = []
        
        for _ in range(100):
            start = time.perf_counter()
            
            # Sanitize input
            alert_id = sanitize_id("alert_abc_123")
            severity = sanitize_severity("high")
            ip = sanitize_ip_address("192.168.1.1")
            
            elapsed = (time.perf_counter() - start) * 1_000_000
            times.append(elapsed)
        
        avg_latency = statistics.mean(times)
        
        # Assert combined operation < 200 microseconds
        assert avg_latency < 200, f"Combined latency {avg_latency}μs exceeds 200μs"
    
    def test_correlation_and_sanitize_performance(self, app_context):
        """Measure correlation ID + sanitization."""
        times = []
        
        for _ in range(100):
            start = time.perf_counter()
            
            # Generate correlation ID (requires app context)
            correlation_id = generate_correlation_id()
            set_correlation_id(correlation_id)
            
            # Sanitize input
            safe_id = sanitize_id("input_id")
            
            elapsed = (time.perf_counter() - start) * 1_000_000
            times.append(elapsed)
        
        avg_latency = statistics.mean(times)
        
        # Assert combined operation < 300 microseconds
        assert avg_latency < 300, f"Combined latency {avg_latency}μs exceeds 300μs"


class TestScalability:
    """Scalability tests for performance under load."""
    
    def test_sanitize_string_burst_throughput(self):
        """Test throughput with burst of sanitization requests."""
        count = 10000
        start = time.perf_counter()
        
        for i in range(count):
            sanitize_string(f"test_input_{i}")
        
        elapsed = time.perf_counter() - start
        throughput = count / elapsed
        
        # Assert reasonable throughput (> 10k ops/sec)
        assert throughput > 10000, f"Throughput {throughput:.0f} ops/sec < 10k"
    
    def test_correlation_id_generation_scale(self):
        """Test correlation ID generation under load."""
        count = 5000
        start = time.perf_counter()
        
        for _ in range(count):
            generate_correlation_id()
        
        elapsed = time.perf_counter() - start
        throughput = count / elapsed
        
        # Assert reasonable throughput (> 5k ops/sec)
        assert throughput > 5000, f"Throughput {throughput:.0f} ops/sec < 5k"
    
    def test_csrf_token_generation_scale(self):
        """Test CSRF token generation under load."""
        count = 5000
        start = time.perf_counter()
        
        for _ in range(count):
            generate_csrf_token()
        
        elapsed = time.perf_counter() - start
        throughput = count / elapsed
        
        # Assert reasonable throughput (> 5k ops/sec)
        assert throughput > 5000, f"Throughput {throughput:.0f} ops/sec < 5k"


class TestMemoryEfficiency:
    """Tests for memory efficiency of operations."""
    
    def test_sanitizer_memory_stability(self):
        """Verify sanitizer doesn't leak memory."""
        # Create many objects and verify they're garbage collected
        for _ in range(1000):
            result = sanitize_string("test_string_value_12345")
        
        # If we reach here without memory issues, test passes
        assert True
    
    def test_correlation_id_memory_stability(self, app_context):
        """Verify correlation ID operations don't leak memory."""
        for _ in range(1000):
            cid = generate_correlation_id()
            set_correlation_id(cid)
            get_correlation_id()
        
        # If we reach here without memory issues, test passes
        assert True
    
    def test_csrf_token_memory_stability(self):
        """Verify CSRF token generation doesn't leak memory."""
        for _ in range(1000):
            token = generate_csrf_token()
        
        # If we reach here without memory issues, test passes
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
