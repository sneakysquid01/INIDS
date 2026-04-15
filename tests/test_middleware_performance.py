"""Middleware performance tests for Flask integration.

Measures the overhead of correlation ID and CSRF protection middleware
when integrated with Flask applications.
"""

import pytest
import time
import statistics
from flask import Flask, jsonify
from src.correlation_tracing import correlation_id_middleware, get_correlation_id
from src.csrf_protection import csrf_protect_middleware


@pytest.fixture
def app_with_middleware():
    """Create Flask app with both middleware registered."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-secret-key'
    app.config['TESTING'] = True
    
    # Register middleware
    correlation_id_middleware(app)
    csrf_protect_middleware(app)
    
    @app.route('/health', methods=['GET'])
    def health():
        correlation_id = get_correlation_id()
        return jsonify({
            'status': 'ok',
            'correlation_id': correlation_id
        })
    
    @app.route('/api/data', methods=['GET'])
    def get_data():
        return jsonify({'data': 'test', 'count': 42})
    
    return app


@pytest.fixture
def app_without_middleware():
    """Create Flask app without middleware for comparison."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok'})
    
    @app.route('/api/data', methods=['GET'])
    def get_data():
        return jsonify({'data': 'test', 'count': 42})
    
    return app


class TestMiddlewareOverhead:
    """Tests measuring middleware performance overhead."""
    
    def test_correlation_middleware_overhead(self, app_with_middleware, app_without_middleware):
        """Measure correlation middleware overhead."""
        client_with = app_with_middleware.test_client()
        client_without = app_without_middleware.test_client()
        
        # Warm up
        client_with.get('/health')
        client_without.get('/health')
        
        # Measure with middleware
        times_with = []
        for _ in range(100):
            start = time.perf_counter()
            client_with.get('/health')
            elapsed = (time.perf_counter() - start) * 1_000_000
            times_with.append(elapsed)
        
        # Measure without middleware
        times_without = []
        for _ in range(100):
            start = time.perf_counter()
            client_without.get('/health')
            elapsed = (time.perf_counter() - start) * 1_000_000
            times_without.append(elapsed)
        
        avg_with = statistics.mean(times_with)
        avg_without = statistics.mean(times_without)
        overhead_us = avg_with - avg_without
        overhead_pct = (overhead_us / avg_without) * 100 if avg_without > 0 else 0
        
        # Middleware overhead should be minimal (< 10ms per request)
        assert overhead_us < 10000, f"Overhead {overhead_us}μs exceeds 10ms"
    
    def test_middleware_request_latency(self, app_with_middleware):
        """Measure request latency with middleware."""
        client = app_with_middleware.test_client()
        
        # Warm up
        client.get('/health')
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            response = client.get('/health')
            elapsed = (time.perf_counter() - start) * 1_000_000
            times.append(elapsed)
            assert response.status_code == 200
        
        avg_latency = statistics.mean(times)
        p95_latency = sorted(times)[int(len(times) * 0.95)]
        
        # Assert reasonable latency
        assert avg_latency < 50000, f"Average latency {avg_latency}μs exceeds 50ms"
        assert p95_latency < 100000, f"P95 latency {p95_latency}μs exceeds 100ms"
    
    def test_correlation_id_header_processing(self, app_with_middleware):
        """Measure performance with correlation ID in request headers."""
        client = app_with_middleware.test_client()
        
        times = []
        for i in range(100):
            start = time.perf_counter()
            response = client.get(
                '/health',
                headers={'X-Correlation-ID': f'test-id-{i}'}
            )
            elapsed = (time.perf_counter() - start) * 1_000_000
            times.append(elapsed)
            assert response.status_code == 200
        
        avg_latency = statistics.mean(times)
        
        # Should handle headers efficiently
        assert avg_latency < 50000, f"Average latency {avg_latency}μs exceeds 50ms"


class TestConcurrentRequests:
    """Tests for performance under concurrent-like scenarios."""
    
    def test_sequential_requests_throughput(self, app_with_middleware):
        """Measure throughput for sequential requests."""
        client = app_with_middleware.test_client()
        
        request_count = 1000
        start = time.perf_counter()
        
        for i in range(request_count):
            response = client.get('/api/data')
            assert response.status_code == 200
        
        elapsed = time.perf_counter() - start
        throughput = request_count / elapsed
        
        # Assert reasonable throughput (> 100 req/sec)
        assert throughput > 100, f"Throughput {throughput:.0f} req/sec < 100"
    
    def test_request_latency_distribution(self, app_with_middleware):
        """Measure request latency distribution."""
        client = app_with_middleware.test_client()
        
        times = []
        for _ in range(500):
            start = time.perf_counter()
            response = client.get('/api/data')
            elapsed = (time.perf_counter() - start) * 1_000_000
            times.append(elapsed)
            assert response.status_code == 200
        
        sorted_times = sorted(times)
        avg = statistics.mean(times)
        median = statistics.median(times)
        p50 = sorted_times[int(len(times) * 0.50)]
        p90 = sorted_times[int(len(times) * 0.90)]
        p95 = sorted_times[int(len(times) * 0.95)]
        p99 = sorted_times[int(len(times) * 0.99)]
        
        # Verify latency percentiles are reasonable
        assert p50 < 20000, f"P50 latency {p50}μs exceeds 20ms"
        assert p90 < 50000, f"P90 latency {p90}μs exceeds 50ms"
        assert p95 < 100000, f"P95 latency {p95}μs exceeds 100ms"


class TestResponseIntegrity:
    """Tests verifying middleware doesn't corrupt responses."""
    
    def test_response_data_integrity(self, app_with_middleware):
        """Verify response data is not modified by middleware."""
        client = app_with_middleware.test_client()
        
        response = client.get('/api/data')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['data'] == 'test'
        assert data['count'] == 42
    
    def test_correlation_id_in_response(self, app_with_middleware):
        """Verify correlation ID appears in response."""
        client = app_with_middleware.test_client()
        
        response = client.get('/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'correlation_id' in data
        # Correlation ID should be set (non-None)
        # Note: In test context, it might be auto-generated


class TestEdgeCases:
    """Tests for middleware edge cases."""
    
    def test_rapid_sequential_requests(self, app_with_middleware):
        """Test middleware with rapid sequential requests."""
        client = app_with_middleware.test_client()
        
        for i in range(100):
            response = client.get('/api/data')
            assert response.status_code == 200
    
    def test_large_request_headers(self, app_with_middleware):
        """Test middleware with large request headers."""
        client = app_with_middleware.test_client()
        
        # Large but valid header value
        large_value = 'x' * 1000
        response = client.get(
            '/api/data',
            headers={'X-Custom-Header': large_value}
        )
        
        # Should still work
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
