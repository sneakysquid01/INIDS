"""Performance analysis and reporting utilities.

Generates comprehensive performance reports with metrics and recommendations.
"""

import json
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any


class PerformanceMetrics:
    """Collects and analyzes performance metrics."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.measurements: Dict[str, List[float]] = {}
        self.start_time = datetime.now()
    
    def record_measurement(self, operation: str, duration_us: float):
        """Record a performance measurement."""
        if operation not in self.measurements:
            self.measurements[operation] = []
        self.measurements[operation].append(duration_us)
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.measurements:
            return {}
        
        times = self.measurements[operation]
        sorted_times = sorted(times)
        
        return {
            'count': len(times),
            'mean_us': statistics.mean(times),
            'median_us': statistics.median(times),
            'stdev_us': statistics.stdev(times) if len(times) > 1 else 0,
            'min_us': min(times),
            'max_us': max(times),
            'p50_us': sorted_times[int(len(times) * 0.50)],
            'p90_us': sorted_times[int(len(times) * 0.90)],
            'p95_us': sorted_times[int(len(times) * 0.95)],
            'p99_us': sorted_times[int(len(times) * 0.99)],
            'throughput_ops_sec': (len(times) / (sum(times) / 1_000_000)) if sum(times) > 0 else 0,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        stats = {}
        for operation in self.measurements.keys():
            stats[operation] = self.get_stats(operation)
        
        return {
            'module': self.module_name,
            'timestamp': self.start_time.isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'operations': stats,
        }


class PerformanceReport:
    """Generates performance reports."""
    
    def __init__(self, title: str = "Performance Analysis Report"):
        self.title = title
        self.sections: List[Tuple[str, str]] = []
    
    def add_section(self, heading: str, content: str):
        """Add a section to the report."""
        self.sections.append((heading, content))
    
    def generate_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# {self.title}",
            f"**Generated**: {datetime.now().isoformat()}",
            "",
        ]
        
        for heading, content in self.sections:
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(content)
            lines.append("")
        
        return "\n".join(lines)
    
    def save(self, filepath: Path):
        """Save report to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.generate_markdown())


class PerformanceAnalyzer:
    """Analyzes and interprets performance data."""
    
    # Target performance thresholds
    THRESHOLDS = {
        'sanitizer_operation_us': 100,
        'correlation_id_operation_us': 50,
        'csrf_token_operation_us': 100,
        'middleware_overhead_us': 10000,  # 10ms
        'min_throughput_ops_sec': 5000,
    }
    
    @staticmethod
    def analyze_latency(times_us: List[float], operation: str) -> Dict[str, Any]:
        """Analyze latency metrics."""
        if not times_us:
            return {'error': 'No measurements'}
        
        sorted_times = sorted(times_us)
        mean_us = statistics.mean(times_us)
        p99_us = sorted_times[int(len(times_us) * 0.99)]
        
        return {
            'operation': operation,
            'mean_us': mean_us,
            'p99_us': p99_us,
            'measurements': len(times_us),
            'acceptable': p99_us < PerformanceAnalyzer.THRESHOLDS.get(f'{operation}_us', 1000),
        }
    
    @staticmethod
    def analyze_throughput(operations: int, duration_seconds: float) -> Dict[str, Any]:
        """Analyze throughput."""
        throughput = operations / duration_seconds if duration_seconds > 0 else 0
        min_throughput = PerformanceAnalyzer.THRESHOLDS['min_throughput_ops_sec']
        
        return {
            'operations': operations,
            'duration_seconds': duration_seconds,
            'throughput_ops_sec': throughput,
            'acceptable': throughput >= min_throughput,
            'target_throughput_ops_sec': min_throughput,
        }
    
    @staticmethod
    def generate_performance_summary(metrics: Dict[str, Any]) -> str:
        """Generate performance summary."""
        lines = []
        
        for module, stats in metrics.items():
            lines.append(f"\n### {module}")
            
            if 'error' in stats:
                lines.append(f"⚠️ Error: {stats['error']}")
                continue
            
            for operation, op_stats in stats.get('operations', {}).items():
                if not op_stats:
                    continue
                
                mean = op_stats.get('mean_us', 0)
                p95 = op_stats.get('p95_us', 0)
                throughput = op_stats.get('throughput_ops_sec', 0)
                
                status = "✅" if p95 < 500 else "⚠️"
                
                lines.append(f"\n**{operation}**:")
                lines.append(f"- Mean: {mean:.1f}μs")
                lines.append(f"- P95: {p95:.1f}μs {status}")
                lines.append(f"- Throughput: {throughput:.0f} ops/sec")
        
        return "\n".join(lines)


def create_performance_benchmark_report() -> str:
    """Create a comprehensive performance benchmark report."""
    report = PerformanceReport("Security Modules Performance Analysis")
    
    # Executive Summary
    report.add_section(
        "Executive Summary",
        """
This report presents comprehensive performance analysis of three security modules:
1. **Input Sanitization** - Validates and sanitizes user inputs
2. **Correlation Tracing** - Tracks requests across service boundaries
3. **CSRF Protection** - Prevents cross-site request forgery attacks

All modules are designed for high-performance, low-latency operations suitable
for integration into high-throughput detection pipelines.
"""
    )
    
    # Performance Targets
    report.add_section(
        "Performance Targets",
        """
| Component | Target | Threshold |
|-----------|--------|-----------|
| Input Sanitization Operation | < 100μs avg | < 500μs p95 |
| Correlation ID Generation | < 50μs avg | < 200μs p95 |
| CSRF Token Generation | < 100μs avg | < 500μs p95 |
| Middleware Overhead | < 10ms per request | - |
| Minimum Throughput | > 5,000 ops/sec | - |
| Request Latency | < 50ms avg | < 100ms p95 |

**Key Goals**:
- Sub-millisecond operation latency
- Minimal Flask middleware overhead
- High throughput for bulk operations
- Linear scalability under load
"""
    )
    
    # Benchmark Methodology
    report.add_section(
        "Methodology",
        """
### Test Environment
- **Python Version**: 3.14.0
- **Framework**: pytest with pytest-benchmark
- **Test Count**: 1,000-5,000 iterations per operation
- **Hardware**: Development machine

### Test Categories
1. **Throughput Tests**: Measure operations per second
2. **Latency Tests**: Measure operation duration (microseconds)
3. **Scalability Tests**: Measure performance under load
4. **Memory Tests**: Verify no memory leaks
5. **Middleware Tests**: Measure Flask integration overhead

### Measurements
- **Mean**: Average operation duration
- **Median**: 50th percentile
- **P95/P99**: Percentile latencies
- **Throughput**: Operations per second
"""
    )
    
    # Expected Results Summary
    report.add_section(
        "Expected Performance Results",
        """
### Input Sanitization
- String sanitization: ~20-50μs per operation
- IP validation: ~30-80μs per operation
- JSON validation: ~50-150μs per operation
- Bulk throughput: > 10,000 ops/sec

### Correlation Tracing
- ID generation: ~20-40μs per operation
- Context access: ~5-10μs per operation
- Bulk throughput: > 20,000 ops/sec

### CSRF Protection
- Token generation: ~50-100μs per operation
- Token validation: ~30-80μs per operation
- Bulk throughput: > 5,000 ops/sec

### Flask Middleware
- Correlation middleware: < 5ms overhead
- CSRF middleware: < 5ms overhead
- Combined overhead: < 10ms per request
- Request throughput: > 100 req/sec
"""
    )
    
    # How to Run Tests
    report.add_section(
        "Running Performance Tests",
        """
### Basic Performance Tests
```bash
pytest tests/test_performance_profiling.py -v
pytest tests/test_middleware_performance.py -v
```

### Benchmark Suite (requires pytest-benchmark)
```bash
pytest tests/test_performance_profiling.py --benchmark-only
```

### Generate Detailed Report
```bash
pytest tests/test_performance_profiling.py --benchmark-json=bench.json
```

### Profile Memory Usage
```bash
pytest tests/test_performance_profiling.py --profile
```
"""
    )
    
    # Pass Criteria
    report.add_section(
        "Pass/Fail Criteria",
        """
### Must Pass
- ✅ All latency assertions (< specified microsecond thresholds)
- ✅ All throughput assertions (> specified operations/second)
- ✅ No memory leaks (memory stability tests)
- ✅ Response integrity (middleware preserves data)

### Should Pass
- ✅ P95 latencies within acceptable ranges
- ✅ Throughput > 50% above minimum targets
- ✅ Consistent latency distribution

### Optional Optimizations
- ⚠️ P99 latencies < 2ms
- ⚠️ Throughput > 100% above minimum targets
- ⚠️ Middleware overhead < 5ms
"""
    )
    
    # Performance Recommendations
    report.add_section(
        "Performance Recommendations",
        """
### Optimization Opportunities
1. **Caching**: Consider caching validation regex patterns
2. **Pooling**: Implement object pooling for token generation
3. **Batching**: Batch multiple sanitizations when possible
4. **Async**: Consider async middleware for blocking operations

### Monitoring
1. Add performance metrics collection to production
2. Set up alerts for P95/P99 latency increases
3. Regular performance regression testing
4. Monitor middleware overhead under load

### Scaling
1. Horizontal scaling: Middleware is stateless
2. Load balancing: Use standard HTTP load balancers
3. Caching: Cache sanitization results when safe
4. Async processing: Consider async/await patterns
"""
    )
    
    # Next Steps
    report.add_section(
        "Next Steps",
        """
### Phase 9 Completion
1. Run all performance tests
2. Collect benchmark results
3. Compare with targets
4. Document any deviations
5. Create optimization plan if needed

### Phase 10 Preparation
1. Security audit of modules
2. External penetration testing
3. Deployment runbook creation
4. Monitoring and alerting setup
5. Production readiness checklist
"""
    )
    
    return report.generate_markdown()


if __name__ == "__main__":
    # Generate and display report
    report = create_performance_benchmark_report()
    print(report)
    
    # Save report
    report_path = Path("PHASE_9_PERFORMANCE_BENCHMARK_PLAN.md")
    report_path.write_text(report)
    print(f"\n✅ Report saved to {report_path}")
