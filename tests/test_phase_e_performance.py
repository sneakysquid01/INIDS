"""
Phase E Tests: Performance Optimization

Tests for:
- Memory pooling and object reuse
- CPU affinity and thread binding
- Connection pooling and management
- Performance monitoring and profiling
"""

import threading
import time

from src.performance import (
    ObjectPool,
    PoolManager,
    CPUAffinityManager,
    ConnectionPool,
    PerformanceMonitor,
    ContextTimer,
    BenchmarkRunner,
    BottleneckDetector,
    get_pool_manager,
    get_affinity_manager,
)


def test_object_pool():
    """Test object pool creation and reuse"""
    print("[TEST] Object Pool Creation and Reuse")
    
    # Create pool
    def create_dict():
        return {"key": "value"}
    
    def reset_dict(d):
        d.clear()
    
    pool = ObjectPool(
        object_factory=create_dict,
        initial_size=10,
        max_size=50,
        reset_func=reset_dict,
    )
    
    # Acquire object
    obj1 = pool.acquire()
    assert obj1 is not None
    
    # Release and reacquire
    pool.release(obj1)
    obj2 = pool.acquire()
    assert obj2 is obj1  # Same object reused
    
    # Get stats
    stats = pool.get_stats()
    assert stats.reuse_count > 0
    
    print("✓ Object pool reuse works correctly")
    return True


def test_pool_manager():
    """Test global pool manager"""
    print("[TEST] Pool Manager")
    
    manager = get_pool_manager()
    
    # Test batch pool
    batch1 = manager.acquire_batch()
    assert batch1 is not None
    manager.release_batch(batch1)
    
    # Test flow context pool
    ctx1 = manager.acquire_flow_context()
    assert ctx1 is not None
    manager.release_flow_context(ctx1)
    
    # Test detection result pool
    result1 = manager.acquire_detection_result()
    assert result1 is not None
    manager.release_detection_result(result1)
    
    # Get stats
    stats = manager.get_all_stats()
    assert "batch_pool" in stats
    assert "flow_context_pool" in stats
    assert "detection_result_pool" in stats
    
    print("✓ Pool manager works correctly")
    return True


def test_cpu_affinity_manager():
    """Test CPU affinity manager"""
    print("[TEST] CPU Affinity Manager")
    
    manager = get_affinity_manager()
    
    # Get CPU info
    cpu_info = manager.get_cpu_info()
    assert cpu_info.cpu_count > 0
    assert len(cpu_info.available_cpus) > 0
    
    # Get optimal distribution
    distribution = manager.get_optimal_worker_distribution(num_workers=4)
    assert len(distribution) == 4
    assert all(cpu < cpu_info.cpu_count for cpu in distribution)
    
    # Try to bind current thread (may not work on all platforms)
    try:
        success = manager.bind_thread(cpu_core=0)
        # Success variable may vary by platform
    except:
        pass
    
    print("✓ CPU affinity manager works correctly")
    return True


def test_connection_pool():
    """Test generic connection pool"""
    print("[TEST] Connection Pool")
    
    # Create pool with mock connections
    connection_count = 0
    
    def create_mock_connection():
        nonlocal connection_count
        connection_count += 1
        return {"id": connection_count, "valid": True}
    
    pool = ConnectionPool(
        connection_factory=create_mock_connection,
        initial_size=5,
        max_size=10,
    )
    
    # Acquire connections
    conn1 = pool.acquire()
    assert conn1 is not None
    assert conn1.connection["valid"]
    
    conn2 = pool.acquire()
    assert conn2 is not None
    assert conn1.connection_id != conn2.connection_id
    
    # Release and verify reuse
    pool.release(conn1)
    conn3 = pool.acquire()
    assert conn3.connection_id == conn1.connection_id  # Reused
    
    # Get stats
    stats = pool.get_stats()
    assert stats.total_created >= 2
    assert stats.reused_count > 0
    
    print("✓ Connection pool works correctly")
    return True


def test_performance_monitor():
    """Test performance monitoring"""
    print("[TEST] Performance Monitoring")
    
    monitor = PerformanceMonitor()
    
    # Record operations
    for i in range(10):
        elapsed = 0.001 + i * 0.0001
        monitor.record_operation("test_op", elapsed)
    
    # Get metrics
    metrics = monitor.get_metrics("test_op")
    assert "test_op" in metrics
    op_metrics = metrics["test_op"]
    assert op_metrics.count == 10
    assert op_metrics.avg_time > 0
    
    print("✓ Performance monitoring works correctly")
    return True


def test_context_timer():
    """Test context timer"""
    print("[TEST] Context Timer")
    
    monitor = PerformanceMonitor()
    
    # Use context timer
    with ContextTimer(monitor, "test_operation"):
        time.sleep(0.01)  # Sleep 10ms
    
    # Verify recorded
    metrics = monitor.get_metrics("test_operation")
    assert "test_operation" in metrics
    assert metrics["test_operation"].count == 1
    assert metrics["test_operation"].avg_time >= 0.009  # At least 9ms
    
    print("✓ Context timer works correctly")
    return True


def test_benchmark_runner():
    """Test benchmark runner"""
    print("[TEST] Benchmark Runner")
    
    def expensive_operation():
        # Simulate work
        total = 0
        for i in range(1000):
            total += i
        return total
    
    runner = BenchmarkRunner(iterations=100, warmup=10)
    metrics = runner.run(expensive_operation)
    
    assert metrics.count == 100
    assert metrics.avg_time > 0
    assert metrics.throughput > 0
    
    print("✓ Benchmark runner works correctly")
    return True


def test_bottleneck_detector():
    """Test bottleneck detection"""
    print("[TEST] Bottleneck Detection")
    
    monitor = PerformanceMonitor()
    
    # Record some operations
    monitor.record_operation("fast_op", 0.001)  # 1ms
    monitor.record_operation("slow_op", 0.050)  # 50ms (above threshold)
    
    # Detect bottlenecks
    detector = BottleneckDetector(monitor, threshold_ms=10.0)
    bottlenecks = detector.detect()
    
    # Should find slow_op as bottleneck
    found_slow_op = any("slow_op" in key for key in bottlenecks.keys())
    assert found_slow_op
    
    print("✓ Bottleneck detector works correctly")
    return True


def test_multi_threaded_pool():
    """Test object pool under concurrent access"""
    print("[TEST] Multi-Threaded Pool")
    
    pool = ObjectPool(
        object_factory=lambda: {"data": []},
        initial_size=5,
        max_size=50,
    )
    
    results = []
    errors = []
    
    def worker():
        try:
            for _ in range(10):
                obj = pool.acquire()
                assert obj is not None
                time.sleep(0.001)
                pool.release(obj)
            results.append("success")
        except Exception as e:
            errors.append(str(e))
    
    # Create threads
    threads = [threading.Thread(target=worker) for _ in range(5)]
    
    # Start all threads
    for t in threads:
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    assert len(errors) == 0
    assert len(results) == 5
    
    stats = pool.get_stats()
    assert stats.reuse_count > 0
    
    print("✓ Multi-threaded pool works correctly")
    return True


def run_all_tests():
    """Run all Phase E tests"""
    print("\n" + "="*60)
    print("PHASE E: PERFORMANCE OPTIMIZATION - TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_object_pool,
        test_pool_manager,
        test_cpu_affinity_manager,
        test_connection_pool,
        test_performance_monitor,
        test_context_timer,
        test_benchmark_runner,
        test_bottleneck_detector,
        test_multi_threaded_pool,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
