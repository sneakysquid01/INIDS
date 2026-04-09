"""
INIDS Performance Optimization Module: Memory Pooling

Object pooling to reduce garbage collection overhead and allocation costs.
Reuses WorkerPacketBatch, FlowContext, and detection results.
"""

import threading
from collections import deque
from typing import Optional, List, Callable
from dataclasses import dataclass, field
import time


@dataclass
class PoolStats:
    """Object pool statistics"""
    total_allocated: int = 0      # Total objects ever created
    current_pooled: int = 0       # Objects currently in pool
    current_active: int = 0       # Objects currently in use
    reuse_count: int = 0          # Number of reuses
    allocation_count: int = 0     # Number of allocations
    gc_saved: int = 0             # GC objects saved
    
    def reuse_ratio(self) -> float:
        """Ratio of reused objects vs allocated"""
        total = self.reuse_count + self.allocation_count
        return self.reuse_count / total if total > 0 else 0.0
    
    def gc_savings_mb(self, bytes_per_object: int = 2048) -> float:
        """Estimated GC savings in MB"""
        return (self.gc_saved * bytes_per_object) / (1024 * 1024)


class ObjectPool:
    """
    Thread-safe object pool for reusing allocated objects.
    
    Reduces memory allocation and garbage collection overhead
    by reusing frequently allocated objects.
    """
    
    def __init__(
        self,
        object_factory: Callable,
        initial_size: int = 100,
        max_size: int = 1000,
        reset_func: Optional[Callable] = None,
    ):
        """
        Initialize object pool.
        
        Args:
            object_factory: Callable that creates a new object
            initial_size: Initial pool size
            max_size: Maximum pool size
            reset_func: Optional function to reset object state
        """
        self.object_factory = object_factory
        self.max_size = max_size
        self.reset_func = reset_func
        self.pool: deque = deque()
        self.lock = threading.Lock()
        self.stats = PoolStats()
        
        # Pre-allocate initial objects
        for _ in range(initial_size):
            self.pool.append(object_factory())
            self.stats.total_allocated += 1
    
    def acquire(self) -> object:
        """
        Acquire object from pool or allocate new one.
        
        Returns:
            Object from pool or newly allocated
        """
        with self.lock:
            if self.pool:
                obj = self.pool.popleft()
                self.stats.reuse_count += 1
                self.stats.current_active += 1
                self.stats.current_pooled = len(self.pool)
                return obj
            else:
                if self.stats.total_allocated < self.max_size:
                    obj = self.object_factory()
                    self.stats.total_allocated += 1
                    self.stats.allocation_count += 1
                    self.stats.current_active += 1
                    return obj
                else:
                    # Pool exhausted, allocate anyway (will cause GC)
                    obj = self.object_factory()
                    self.stats.allocation_count += 1
                    self.stats.current_active += 1
                    return obj
    
    def release(self, obj: object) -> None:
        """
        Release object back to pool.
        
        Args:
            obj: Object to release
        """
        with self.lock:
            self.stats.gc_saved += 1
            self.stats.current_active = max(0, self.stats.current_active - 1)
            
            if len(self.pool) < self.max_size:
                # Reset object state if reset function provided
                if self.reset_func:
                    self.reset_func(obj)
                
                self.pool.append(obj)
                self.stats.current_pooled = len(self.pool)
    
    def get_stats(self) -> PoolStats:
        """Get pool statistics"""
        with self.lock:
            return PoolStats(
                total_allocated=self.stats.total_allocated,
                current_pooled=self.stats.current_pooled,
                current_active=self.stats.current_active,
                reuse_count=self.stats.reuse_count,
                allocation_count=self.stats.allocation_count,
                gc_saved=self.stats.gc_saved,
            )
    
    def clear(self) -> None:
        """Clear pool"""
        with self.lock:
            self.pool.clear()
            self.stats.current_pooled = 0


class WorkerPacketBatchPool:
    """
    Specialized pool for WorkerPacketBatch objects.
    
    Reduces allocation overhead for high-throughput packet processing.
    """
    
    def __init__(self, initial_size: int = 100, max_size: int = 1000):
        """
        Initialize batch pool.
        
        Args:
            initial_size: Initial number of batch objects
            max_size: Maximum batch objects to pool
        """
        def create_batch():
            # Import here to avoid circular dependency
            from src.distributed_detection.worker_pool import WorkerPacketBatch
            return WorkerPacketBatch(
                batch_id=0,
                packets=[],
                flow_contexts={},
                timestamps_received=0.0,
            )
        
        def reset_batch(batch):
            batch.batch_id = 0
            batch.packets.clear()
            batch.flow_contexts.clear()
            batch.timestamps_received = 0.0
        
        self.pool = ObjectPool(
            object_factory=create_batch,
            initial_size=initial_size,
            max_size=max_size,
            reset_func=reset_batch,
        )
    
    def acquire(self):
        """Acquire batch from pool"""
        return self.pool.acquire()
    
    def release(self, batch):
        """Release batch back to pool"""
        self.pool.release(batch)
    
    def get_stats(self) -> PoolStats:
        """Get pool statistics"""
        return self.pool.get_stats()


class FlowContextPool:
    """
    Specialized pool for FlowContext objects.
    
    Reduces allocation overhead for flow tracking.
    """
    
    def __init__(self, initial_size: int = 1000, max_size: int = 10000):
        """
        Initialize flow context pool.
        
        Args:
            initial_size: Initial number of flow contexts
            max_size: Maximum flow contexts to pool
        """
        def create_flow_context():
            from src.packet_capture import FlowContext, TCPState
            return FlowContext(
                src_ip="",
                dst_ip="",
                src_port=0,
                dst_port=0,
                proto="",
                state=TCPState.NEW,
            )
        
        def reset_flow_context(ctx):
            ctx.src_ip = ""
            ctx.dst_ip = ""
            ctx.src_port = 0
            ctx.dst_port = 0
            ctx.proto = ""
            ctx.packets_toserver = 0
            ctx.packets_toclient = 0
            ctx.bytes_toserver = 0
            ctx.bytes_toclient = 0
            ctx.detection_action = "ALLOW"
            ctx.features_cache = {}
        
        self.pool = ObjectPool(
            object_factory=create_flow_context,
            initial_size=initial_size,
            max_size=max_size,
            reset_func=reset_flow_context,
        )
    
    def acquire(self):
        """Acquire flow context from pool"""
        return self.pool.acquire()
    
    def release(self, ctx):
        """Release flow context back to pool"""
        self.pool.release(ctx)
    
    def get_stats(self) -> PoolStats:
        """Get pool statistics"""
        return self.pool.get_stats()


class DetectionResultPool:
    """
    Specialized pool for detection result dictionaries.
    
    Reduces allocation overhead for detection result aggregation.
    """
    
    def __init__(self, initial_size: int = 500, max_size: int = 5000):
        """
        Initialize detection result pool.
        
        Args:
            initial_size: Initial number of result dicts
            max_size: Maximum result dicts to pool
        """
        def create_result():
            return {
                "flow_id": 0,
                "score": 0.0,
                "reason": "",
                "timestamp": 0.0,
                "threats": [],
                "metadata": {},
            }
        
        def reset_result(result):
            result["flow_id"] = 0
            result["score"] = 0.0
            result["reason"] = ""
            result["timestamp"] = 0.0
            result["threats"].clear()
            result["metadata"].clear()
        
        self.pool = ObjectPool(
            object_factory=create_result,
            initial_size=initial_size,
            max_size=max_size,
            reset_func=reset_result,
        )
    
    def acquire(self):
        """Acquire result dict from pool"""
        return self.pool.acquire()
    
    def release(self, result):
        """Release result dict back to pool"""
        self.pool.release(result)
    
    def get_stats(self) -> PoolStats:
        """Get pool statistics"""
        return self.pool.get_stats()


class PoolManager:
    """
    Centralized management of all object pools.
    
    Coordinates pool lifecycle and statistics across
    WorkerPacketBatch, FlowContext, and DetectionResult pools.
    """
    
    def __init__(self):
        """Initialize pool manager"""
        self.batch_pool = WorkerPacketBatchPool(initial_size=100, max_size=1000)
        self.flow_context_pool = FlowContextPool(initial_size=1000, max_size=10000)
        self.detection_result_pool = DetectionResultPool(initial_size=500, max_size=5000)
    
    def acquire_batch(self):
        """Acquire packet batch"""
        return self.batch_pool.acquire()
    
    def release_batch(self, batch):
        """Release packet batch"""
        self.batch_pool.release(batch)
    
    def acquire_flow_context(self):
        """Acquire flow context"""
        return self.flow_context_pool.acquire()
    
    def release_flow_context(self, ctx):
        """Release flow context"""
        self.flow_context_pool.release(ctx)
    
    def acquire_detection_result(self):
        """Acquire detection result"""
        return self.detection_result_pool.acquire()
    
    def release_detection_result(self, result):
        """Release detection result"""
        self.detection_result_pool.release(result)
    
    def get_all_stats(self) -> dict:
        """
        Get statistics from all pools.
        
        Returns:
            Dict with stats for each pool type
        """
        return {
            "batch_pool": self.batch_pool.get_stats(),
            "flow_context_pool": self.flow_context_pool.get_stats(),
            "detection_result_pool": self.detection_result_pool.get_stats(),
        }
    
    def print_stats(self) -> None:
        """Print pool statistics"""
        stats = self.get_all_stats()
        
        print("\n" + "="*60)
        print("MEMORY POOL STATISTICS")
        print("="*60)
        
        for pool_name, pool_stats in stats.items():
            print(f"\n{pool_name}:")
            print(f"  Total allocated: {pool_stats.total_allocated}")
            print(f"  Current pooled: {pool_stats.current_pooled}")
            print(f"  Current active: {pool_stats.current_active}")
            print(f"  Reuse count: {pool_stats.reuse_count}")
            print(f"  Allocation count: {pool_stats.allocation_count}")
            print(f"  Reuse ratio: {pool_stats.reuse_ratio():.1%}")
            print(f"  GC objects saved: {pool_stats.gc_saved}")
            print(f"  GC savings: {pool_stats.gc_savings_mb():.2f} MB")
        
        print("\n" + "="*60 + "\n")


# Global pool manager instance
_pool_manager = None


def get_pool_manager() -> PoolManager:
    """Get global pool manager instance"""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = PoolManager()
    return _pool_manager


def init_pools(batch_pool_size: int = 100, flow_pool_size: int = 1000) -> PoolManager:
    """
    Initialize global pool manager.
    
    Args:
        batch_pool_size: Size of batch pool
        flow_pool_size: Size of flow context pool
    
    Returns:
        Initialized PoolManager
    """
    global _pool_manager
    _pool_manager = PoolManager()
    return _pool_manager
