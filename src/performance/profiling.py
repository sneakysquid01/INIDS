"""
INIDS Performance Optimization Module: Profiling & Monitoring

Real-time performance monitoring, profiling, and bottleneck detection.
"""

import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import logging


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    name: str
    value: float              # Measurement value
    unit: str                 # Unit (ms, µs, pps, etc)
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.name}: {self.value:.2f} {self.unit}"


@dataclass
class OperationMetrics:
    """Metrics for a specific operation"""
    name: str
    count: int = 0             # Number of times executed
    total_time: float = 0.0    # Total time spent (seconds)
    min_time: float = float('inf')
    max_time: float = 0.0
    last_time: float = 0.0
    
    @property
    def avg_time(self) -> float:
        """Average time per operation"""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def throughput(self) -> float:
        """Operations per second"""
        return self.count / self.total_time if self.total_time > 0 else 0.0
    
    def update(self, elapsed_time: float) -> None:
        """Update with new measurement"""
        self.count += 1
        self.total_time += elapsed_time
        self.last_time = elapsed_time
        self.min_time = min(self.min_time, elapsed_time)
        self.max_time = max(self.max_time, elapsed_time)


class PerformanceMonitor:
    """
    Real-time performance monitoring and metrics collection.
    
    Tracks operation latencies, throughput, and resource usage.
    """
    
    def __init__(self, max_metrics: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_metrics: Maximum metrics to keep in history
        """
        self.max_metrics = max_metrics
        self.operations: Dict[str, OperationMetrics] = {}
        self.metrics_history: deque = deque(maxlen=max_metrics)
        self.lock = threading.Lock()
        self.logger = logging.getLogger("INIDS.Performance.Monitor")
    
    def record_operation(
        self,
        operation_name: str,
        elapsed_time: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record operation timing.
        
        Args:
            operation_name: Name of operation
            elapsed_time: Time taken (seconds)
            tags: Optional metadata tags
        """
        with self.lock:
            # Update operation metrics
            if operation_name not in self.operations:
                self.operations[operation_name] = OperationMetrics(name=operation_name)
            
            self.operations[operation_name].update(elapsed_time)
            
            # Add to history
            metric = PerformanceMetric(
                name=operation_name,
                value=elapsed_time * 1000,  # Convert to ms
                unit="ms",
                tags=tags or {},
            )
            self.metrics_history.append(metric)
    
    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, OperationMetrics]:
        """
        Get operation metrics.
        
        Args:
            operation_name: Specific operation or all if None
        
        Returns:
            Dict of operation metrics
        """
        with self.lock:
            if operation_name:
                return {operation_name: self.operations.get(operation_name)}
            else:
                return self.operations.copy()
    
    def get_history(self) -> List[PerformanceMetric]:
        """Get metrics history"""
        with self.lock:
            return list(self.metrics_history)
    
    def print_report(self, top_n: int = 10) -> None:
        """
        Print performance report.
        
        Args:
            top_n: Top N slowest operations to show
        """
        with self.lock:
            if not self.operations:
                print("No metrics collected yet")
                return
            
            print("\n" + "="*60)
            print("PERFORMANCE METRICS REPORT")
            print("="*60)
            
            # Sort by average time (slowest first)
            sorted_ops = sorted(
                self.operations.values(),
                key=lambda x: x.avg_time,
                reverse=True,
            )
            
            print(f"\nTop {top_n} Slowest Operations:\n")
            print(f"{'Operation':<30} {'Count':>8} {'Avg (ms)':>12} {'Max (ms)':>12} {'Throughput':>12}")
            print("-" * 75)
            
            for op in sorted_ops[:top_n]:
                throughput = f"{op.throughput:.0f} ops/s" if op.throughput > 0 else "N/A"
                print(f"{op.name:<30} {op.count:>8} {op.avg_time*1000:>12.2f} {op.max_time*1000:>12.2f} {throughput:>12}")
            
            print("\n" + "="*60 + "\n")


class ContextTimer:
    """
    Context manager for timing operations.
    
    Usage:
        monitor = PerformanceMonitor()
        with ContextTimer(monitor, "packet_decode"):
            # Packet decoding code
            pass
    """
    
    def __init__(
        self,
        monitor: PerformanceMonitor,
        operation_name: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize context timer.
        
        Args:
            monitor: PerformanceMonitor instance
            operation_name: Operation name
            tags: Optional metadata
        """
        self.monitor = monitor
        self.operation_name = operation_name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        """Enter context"""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        elapsed = time.perf_counter() - self.start_time
        self.monitor.record_operation(self.operation_name, elapsed, self.tags)
        return False


class BenchmarkRunner:
    """
    Benchmark runner for performance testing.
    
    Runs function multiple times and collects statistics.
    """
    
    def __init__(self, iterations: int = 1000, warmup: int = 100):
        """
        Initialize benchmark runner.
        
        Args:
            iterations: Number of iterations
            warmup: Warmup runs before measurement
        """
        self.iterations = iterations
        self.warmup = warmup
        self.logger = logging.getLogger("INIDS.Performance.Benchmark")
    
    def run(
        self,
        function: Callable,
        *args,
        **kwargs,
    ) -> OperationMetrics:
        """
        Run benchmark on function.
        
        Args:
            function: Function to benchmark
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            OperationMetrics with results
        """
        metrics = OperationMetrics(name=function.__name__)
        
        # Warmup
        for _ in range(self.warmup):
            try:
                function(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Warmup failed: {e}")
                return metrics
        
        # Measurement
        for _ in range(self.iterations):
            start = time.perf_counter()
            try:
                function(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Benchmark iteration failed: {e}")
            elapsed = time.perf_counter() - start
            metrics.update(elapsed)
        
        return metrics
    
    def compare_functions(
        self,
        functions: Dict[str, Callable],
        *args,
        **kwargs,
    ) -> Dict[str, OperationMetrics]:
        """
        Compare performance of multiple functions.
        
        Args:
            functions: Dict of name -> function
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Dict of results for each function
        """
        results = {}
        
        for name, func in functions.items():
            self.logger.info(f"Benchmarking {name}...")
            results[name] = self.run(func, *args, **kwargs)
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[str, OperationMetrics]) -> None:
        """Print benchmark comparison"""
        print("\n" + "="*60)
        print("BENCHMARK COMPARISON")
        print("="*60 + "\n")
        
        print(f"{'Function':<30} {'Avg (µs)':>12} {'Min (µs)':>12} {'Max (µs)':>12} {'Ops/sec':>12}")
        print("-" * 80)
        
        for name, metrics in results.items():
            avg_us = metrics.avg_time * 1_000_000
            min_us = metrics.min_time * 1_000_000
            max_us = metrics.max_time * 1_000_000
            ops_sec = metrics.throughput
            
            print(f"{name:<30} {avg_us:>12.2f} {min_us:>12.2f} {max_us:>12.2f} {ops_sec:>12.0f}")
        
        print("\n" + "="*60 + "\n")


class BottleneckDetector:
    """
    Detects performance bottlenecks in the system.
    
    Analyzes metrics to identify slow operations and resource contention.
    """
    
    def __init__(self, monitor: PerformanceMonitor, threshold_ms: float = 10.0):
        """
        Initialize bottleneck detector.
        
        Args:
            monitor: PerformanceMonitor instance
            threshold_ms: Latency threshold in milliseconds
        """
        self.monitor = monitor
        self.threshold_ms = threshold_ms
        self.logger = logging.getLogger("INIDS.Performance.BottleneckDetector")
    
    def detect(self) -> Dict[str, str]:
        """
        Detect performance bottlenecks.
        
        Returns:
            Dict of detected bottlenecks
        """
        bottlenecks = {}
        
        metrics = self.monitor.get_metrics()
        
        for op_name, op_metrics in metrics.items():
            avg_ms = op_metrics.avg_time * 1000
            
            # Check average latency
            if avg_ms > self.threshold_ms:
                bottlenecks[op_name] = f"High latency: {avg_ms:.2f}ms (threshold: {self.threshold_ms}ms)"
            
            # Check variance
            if op_metrics.max_time > 0 and op_metrics.min_time > 0:
                variance = (op_metrics.max_time - op_metrics.min_time) / op_metrics.min_time
                if variance > 10:  # 10x difference
                    bottlenecks[f"{op_name}_variance"] = f"High variance: max {op_metrics.max_time*1000:.2f}ms, min {op_metrics.min_time*1000:.2f}ms"
        
        return bottlenecks
    
    def print_report(self) -> None:
        """Print bottleneck report"""
        bottlenecks = self.detect()
        
        if not bottlenecks:
            print("✓ No bottlenecks detected")
            return
        
        print("\n" + "="*60)
        print("BOTTLENECK DETECTION REPORT")
        print("="*60 + "\n")
        
        for issue, description in bottlenecks.items():
            print(f"⚠ {issue}")
            print(f"  {description}\n")
        
        print("="*60 + "\n")


# Global instances
_monitor: Optional[PerformanceMonitor] = None
_detector: Optional[BottleneckDetector] = None


def get_monitor() -> PerformanceMonitor:
    """Get global performance monitor"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


def get_bottleneck_detector() -> BottleneckDetector:
    """Get global bottleneck detector"""
    global _detector
    if _detector is None:
        _detector = BottleneckDetector(get_monitor())
    return _detector


def init_performance_monitoring() -> Dict:
    """Initialize performance monitoring"""
    global _monitor, _detector
    _monitor = PerformanceMonitor()
    _detector = BottleneckDetector(_monitor)
    
    return {
        "monitor": _monitor,
        "detector": _detector,
    }
