"""
INIDS Performance Optimization Module

Comprehensive performance enhancement tools:
- Memory pooling (object reuse)
- CPU affinity (thread binding)
- Connection pooling (persistent connections)
- Performance monitoring & profiling
- Bottleneck detection
"""

from .memory_pooling import (
    ObjectPool,
    PoolStats,
    PoolManager,
    WorkerPacketBatchPool,
    FlowContextPool,
    DetectionResultPool,
    get_pool_manager,
    init_pools,
)

from .cpu_affinity import (
    CPUAffinityManager,
    CPUAffinityWrapper,
    WorkerThreadWithAffinity,
    get_cpu_info,
    get_affinity_manager,
    init_affinity,
)

from .connection_pooling import (
    Connection,
    ConnectionPool,
    ConnectionPoolStats,
    RedisConnectionPool,
    HTTPConnectionPool,
    get_redis_pool,
    get_http_pool,
)

from .profiling import (
    PerformanceMonitor,
    PerformanceMetric,
    OperationMetrics,
    ContextTimer,
    BenchmarkRunner,
    BottleneckDetector,
    get_monitor,
    get_bottleneck_detector,
    init_performance_monitoring,
)

__all__ = [
    # Memory pooling
    "ObjectPool",
    "PoolStats",
    "PoolManager",
    "WorkerPacketBatchPool",
    "FlowContextPool",
    "DetectionResultPool",
    "get_pool_manager",
    "init_pools",
    # CPU affinity
    "CPUAffinityManager",
    "CPUAffinityWrapper",
    "WorkerThreadWithAffinity",
    "get_cpu_info",
    "get_affinity_manager",
    "init_affinity",
    # Connection pooling
    "Connection",
    "ConnectionPool",
    "ConnectionPoolStats",
    "RedisConnectionPool",
    "HTTPConnectionPool",
    "get_redis_pool",
    "get_http_pool",
    # Profiling
    "PerformanceMonitor",
    "PerformanceMetric",
    "OperationMetrics",
    "ContextTimer",
    "BenchmarkRunner",
    "BottleneckDetector",
    "get_monitor",
    "get_bottleneck_detector",
    "init_performance_monitoring",
]
