"""
Phase E Validation Script

Validates performance optimization module completeness and functionality.
No pytest dependency - runs standalone.
"""

import sys
import time


def validate_imports():
    """Validate all imports work"""
    print("[VALIDATION] Checking imports...")
    
    try:
        from src.performance import (
            ObjectPool,
            PoolStats,
            PoolManager,
            CPUAffinityManager,
            ConnectionPool,
            ConnectionPoolStats,
            PerformanceMonitor,
            ContextTimer,
            BenchmarkRunner,
            BottleneckDetector,
            get_pool_manager,
            get_affinity_manager,
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def validate_memory_pooling():
    """Validate memory pooling functionality"""
    print("[VALIDATION] Checking memory pooling...")
    
    try:
        from src.performance import ObjectPool
        
        # Create pool
        pool = ObjectPool(
            object_factory=lambda: {"data": []},
            initial_size=10,
            max_size=50,
        )
        
        # Test acquire/release
        obj = pool.acquire()
        assert obj is not None
        
        pool.release(obj)
        
        # Verify stats
        stats = pool.get_stats()
        assert stats.total_allocated > 0
        
        print("  ✓ Memory pooling works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Memory pooling validation failed: {e}")
        return False


def validate_pool_manager():
    """Validate pool manager"""
    print("[VALIDATION] Checking pool manager...")
    
    try:
        from src.performance import get_pool_manager
        
        manager = get_pool_manager()
        
        # Test all pools
        batch = manager.acquire_batch()
        manager.release_batch(batch)
        
        ctx = manager.acquire_flow_context()
        manager.release_flow_context(ctx)
        
        result = manager.acquire_detection_result()
        manager.release_detection_result(result)
        
        # Get stats
        stats = manager.get_all_stats()
        assert len(stats) == 3
        
        print("  ✓ Pool manager works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Pool manager validation failed: {e}")
        return False


def validate_cpu_affinity():
    """Validate CPU affinity"""
    print("[VALIDATION] Checking CPU affinity...")
    
    try:
        from src.performance import get_affinity_manager, get_cpu_info
        
        # Get CPU info
        cpu_info = get_cpu_info()
        assert cpu_info.cpu_count > 0
        
        # Get manager
        manager = get_affinity_manager()
        
        # Get distribution
        distribution = manager.get_optimal_worker_distribution(num_workers=4)
        assert len(distribution) == 4
        
        print("  ✓ CPU affinity works correctly")
        return True
    except Exception as e:
        print(f"  ✗ CPU affinity validation failed: {e}")
        return False


def validate_connection_pooling():
    """Validate connection pooling"""
    print("[VALIDATION] Checking connection pooling...")
    
    try:
        from src.performance import ConnectionPool
        
        # Create pool
        pool = ConnectionPool(
            connection_factory=lambda: {"id": 1},
            initial_size=5,
            max_size=10,
        )
        
        # Test acquire/release
        conn = pool.acquire()
        assert conn is not None
        
        pool.release(conn)
        
        # Get stats
        stats = pool.get_stats()
        assert stats.total_created > 0
        
        pool.close_all()
        
        print("  ✓ Connection pooling works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Connection pooling validation failed: {e}")
        return False


def validate_performance_monitoring():
    """Validate performance monitoring"""
    print("[VALIDATION] Checking performance monitoring...")
    
    try:
        from src.performance import PerformanceMonitor, ContextTimer
        
        monitor = PerformanceMonitor()
        
        # Record operations
        monitor.record_operation("test", 0.001)
        monitor.record_operation("test", 0.002)
        
        # Get metrics
        metrics = monitor.get_metrics("test")
        assert "test" in metrics
        assert metrics["test"].count == 2
        
        # Test context timer
        with ContextTimer(monitor, "timer_test"):
            time.sleep(0.001)
        
        metrics = monitor.get_metrics("timer_test")
        assert "timer_test" in metrics
        
        print("  ✓ Performance monitoring works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Performance monitoring validation failed: {e}")
        return False


def validate_benchmarking():
    """Validate benchmark runner"""
    print("[VALIDATION] Checking benchmark runner...")
    
    try:
        from src.performance import BenchmarkRunner
        
        def test_func():
            return sum(range(100))
        
        runner = BenchmarkRunner(iterations=100, warmup=10)
        metrics = runner.run(test_func)
        
        assert metrics.count == 100
        assert metrics.avg_time > 0
        assert metrics.throughput > 0
        
        print("  ✓ Benchmark runner works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Benchmark runner validation failed: {e}")
        return False


def validate_bottleneck_detection():
    """Validate bottleneck detection"""
    print("[VALIDATION] Checking bottleneck detection...")
    
    try:
        from src.performance import PerformanceMonitor, BottleneckDetector
        
        monitor = PerformanceMonitor()
        
        # Record slow operation
        monitor.record_operation("slow", 0.050)
        
        # Detect bottlenecks
        detector = BottleneckDetector(monitor, threshold_ms=10.0)
        bottlenecks = detector.detect()
        
        # Should find at least one
        assert len(bottlenecks) > 0
        
        print("  ✓ Bottleneck detection works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Bottleneck detection validation failed: {e}")
        return False


def validate_multi_pool_stats():
    """Validate pool statistics calculation"""
    print("[VALIDATION] Checking pool statistics...")
    
    try:
        from src.performance import ObjectPool, PoolStats
        
        pool = ObjectPool(
            object_factory=lambda: {"data": []},
            initial_size=10,
            max_size=50,
        )
        
        # Get stats
        stats = pool.get_stats()
        
        # Verify stats properties
        assert stats.total_allocated >= 10
        assert stats.reuse_ratio() >= 0.0
        assert stats.reuse_ratio() <= 1.0
        
        print("  ✓ Pool statistics work correctly")
        return True
    except Exception as e:
        print(f"  ✗ Pool statistics validation failed: {e}")
        return False


def validate_operation_metrics():
    """Validate operation metrics"""
    print("[VALIDATION] Checking operation metrics...")
    
    try:
        from src.performance import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Record multiple measurements
        for _ in range(10):
            monitor.record_operation("op", 0.001)
        
        # Get metrics
        metrics = monitor.get_metrics("op")["op"]
        
        # Verify calculations
        assert metrics.count == 10
        assert metrics.avg_time > 0
        assert metrics.throughput > 0
        assert metrics.min_time > 0
        assert metrics.max_time > 0
        assert metrics.min_time <= metrics.avg_time <= metrics.max_time
        
        print("  ✓ Operation metrics work correctly")
        return True
    except Exception as e:
        print(f"  ✗ Operation metrics validation failed: {e}")
        return False


def validate_thread_safety():
    """Validate thread safety of pools"""
    print("[VALIDATION] Checking thread safety...")
    
    try:
        import threading
        from src.performance import ObjectPool
        
        pool = ObjectPool(
            object_factory=lambda: {"id": 1},
            initial_size=5,
            max_size=50,
        )
        
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    obj = pool.acquire()
                    if obj is None:
                        errors.append("Failed to acquire")
                    pool.release(obj)
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        print("  ✓ Thread safety validated")
        return True
    except Exception as e:
        print(f"  ✗ Thread safety validation failed: {e}")
        return False


def main():
    """Run all validations"""
    print("\n" + "="*60)
    print("PHASE E: PERFORMANCE OPTIMIZATION - VALIDATION SCRIPT")
    print("="*60 + "\n")
    
    validations = [
        validate_imports,
        validate_memory_pooling,
        validate_pool_manager,
        validate_cpu_affinity,
        validate_connection_pooling,
        validate_performance_monitoring,
        validate_benchmarking,
        validate_bottleneck_detection,
        validate_multi_pool_stats,
        validate_operation_metrics,
        validate_thread_safety,
    ]
    
    passed = 0
    failed = 0
    
    for validation_func in validations:
        try:
            if validation_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Validation error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(validations)} validations passed")
    if failed == 0:
        print("STATUS: ✓ ALL VALIDATIONS PASSED")
    else:
        print(f"STATUS: ✗ {failed} VALIDATIONS FAILED")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
