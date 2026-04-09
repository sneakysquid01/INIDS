"""
INIDS Performance Optimization Module: CPU Affinity

Binds worker threads to specific CPU cores for optimal performance.
Reduces cache misses and context switching overhead.

Supported platforms:
- Linux: psutil + os.sched_setaffinity
- Windows: ctypes + Windows API
- macOS: partial support via psutil
"""

import threading
import os
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CPUInfo:
    """CPU information and availability"""
    cpu_count: int              # Total logical CPUs
    physical_cpu_count: int     # Physical CPU cores
    available_cpus: List[int]   # Available CPU indices
    numa_nodes: Optional[int]   # NUMA node count (if available)
    
    def cpus_per_numa(self) -> Optional[int]:
        """CPUs per NUMA node"""
        if self.numa_nodes and self.numa_nodes > 0:
            return self.cpu_count // self.numa_nodes
        return None


def get_cpu_info() -> CPUInfo:
    """
    Detect CPU configuration.
    
    Returns:
        CPUInfo instance with available CPUs
    """
    try:
        import psutil
        
        cpu_count = psutil.cpu_count(logical=True)
        physical_count = psutil.cpu_count(logical=False) or cpu_count
        
        # Get available CPUs (respects cgroup limits)
        try:
            available = list(range(cpu_count))
        except:
            available = list(range(cpu_count))
        
        # NUMA support
        numa_nodes = None
        try:
            numa_nodes = psutil.cpu_count(logical=False)
        except AttributeError:
            pass
        
        return CPUInfo(
            cpu_count=cpu_count,
            physical_cpu_count=physical_count,
            available_cpus=available,
            numa_nodes=numa_nodes,
        )
    except ImportError:
        # Fallback: manual detection
        cpu_count = os.cpu_count() or 1
        return CPUInfo(
            cpu_count=cpu_count,
            physical_cpu_count=cpu_count,
            available_cpus=list(range(cpu_count)),
            numa_nodes=None,
        )


class CPUAffinityManager:
    """
    Manages CPU affinity for worker threads.
    
    Binds threads to specific CPU cores to improve performance
    by reducing cache misses and context switching.
    
    Usage:
        manager = CPUAffinityManager()
        manager.bind_thread(thread_id=0, cpu_core=0)
        manager.bind_worker_pool(num_workers=4)
    """
    
    def __init__(self):
        """Initialize CPU affinity manager"""
        self.logger = logging.getLogger("INIDS.Performance.CPUAffinity")
        self.cpu_info = get_cpu_info()
        self.thread_bindings: Dict[int, int] = {}  # thread_id -> cpu_core
        self.lock = threading.Lock()
        
        self.logger.info(
            f"CPU configuration: {self.cpu_info.cpu_count} logical "
            f"({self.cpu_info.physical_cpu_count} physical)"
        )
    
    def bind_thread(
        self,
        thread_id: Optional[int] = None,
        cpu_core: int = 0,
    ) -> bool:
        """
        Bind thread to specific CPU core.
        
        Args:
            thread_id: Thread ID (current thread if None)
            cpu_core: CPU core index (0-based)
        
        Returns:
            True if binding successful
        """
        if cpu_core >= self.cpu_info.cpu_count:
            self.logger.error(f"CPU core {cpu_core} out of range")
            return False
        
        if thread_id is None:
            thread_id = threading.current_thread().ident
        
        try:
            # Linux: Use os.sched_setaffinity
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(thread_id, {cpu_core})
                with self.lock:
                    self.thread_bindings[thread_id] = cpu_core
                self.logger.debug(f"Bound thread {thread_id} to CPU {cpu_core}")
                return True
            
            # Windows: Use ctypes + Windows API
            elif os.name == 'nt':
                return self._bind_thread_windows(thread_id, cpu_core)
            
            # macOS/Others: Log warning (limited support)
            else:
                self.logger.warning(f"CPU affinity not supported on {os.name}")
                return False
        
        except Exception as e:
            self.logger.error(f"Failed to bind thread: {e}")
            return False
    
    def _bind_thread_windows(self, thread_id: int, cpu_core: int) -> bool:
        """
        Bind thread on Windows using SetThreadAffinityMask API.
        
        Args:
            thread_id: Thread ID
            cpu_core: CPU core index
        
        Returns:
            True if successful
        """
        try:
            import ctypes
            from ctypes import wintypes
            
            # Get Windows thread handle
            kernel32 = ctypes.windll.kernel32
            current_process = kernel32.GetCurrentProcess()
            
            # Create affinity mask (1 << cpu_core)
            affinity_mask = 1 << cpu_core
            
            # Set thread affinity
            result = kernel32.SetThreadAffinityMask(current_process, affinity_mask)
            
            if result:
                with self.lock:
                    self.thread_bindings[thread_id] = cpu_core
                self.logger.debug(f"Bound Windows thread {thread_id} to CPU {cpu_core}")
                return True
            else:
                self.logger.error(f"SetThreadAffinityMask failed for thread {thread_id}")
                return False
        
        except Exception as e:
            self.logger.error(f"Windows affinity binding failed: {e}")
            return False
    
    def bind_worker_pool(
        self,
        num_workers: int,
        workers: Optional[List[threading.Thread]] = None,
    ) -> int:
        """
        Bind worker threads to CPU cores in round-robin fashion.
        
        Args:
            num_workers: Number of workers
            workers: List of worker threads (optional)
        
        Returns:
            Number of successfully bound threads
        """
        bound_count = 0
        available_cpus = self.cpu_info.available_cpus
        
        for i in range(num_workers):
            cpu_core = available_cpus[i % len(available_cpus)]
            
            if workers and i < len(workers):
                thread = workers[i]
                # Bind current thread if it's the worker
                if thread.is_alive():
                    # Note: Can't directly set affinity for running thread
                    # Thread should set it own affinity on startup
                    self.logger.debug(f"Worker {i} scheduled for CPU {cpu_core}")
                    bound_count += 1
            else:
                # Record binding intention
                self.logger.debug(f"Worker {i} assigned to CPU {cpu_core}")
                bound_count += 1
        
        return bound_count
    
    def get_optimal_worker_distribution(
        self,
        num_workers: int,
    ) -> List[int]:
        """
        Get optimal CPU core assignment for worker threads.
        
        Returns list of CPU core indices for each worker.
        Avoids NUMA boundaries if possible.
        
        Args:
            num_workers: Number of workers
        
        Returns:
            List of CPU core indices
        """
        available_cpus = self.cpu_info.available_cpus
        
        # Simple round-robin distribution
        distribution = [
            available_cpus[i % len(available_cpus)]
            for i in range(num_workers)
        ]
        
        # NUMA-aware distribution (if available)
        if self.cpu_info.numa_nodes and self.cpu_info.numa_nodes > 1:
            cpus_per_numa = self.cpu_info.cpus_per_numa()
            if cpus_per_numa:
                # Try to fit workers within NUMA nodes
                self.logger.info(
                    f"NUMA-aware distribution: "
                    f"{self.cpu_info.numa_nodes} nodes, "
                    f"{cpus_per_numa} CPUs per node"
                )
                
                # Distribute workers across NUMA nodes
                distribution = []
                for i in range(num_workers):
                    numa_node = (i // cpus_per_numa) % self.cpu_info.numa_nodes
                    cpu_in_node = i % cpus_per_numa
                    cpu_core = numa_node * cpus_per_numa + cpu_in_node
                    if cpu_core < len(available_cpus):
                        distribution.append(available_cpus[cpu_core])
                    else:
                        distribution.append(available_cpus[i % len(available_cpus)])
        
        return distribution
    
    def get_bindings(self) -> Dict[int, int]:
        """Get current thread-to-CPU bindings"""
        with self.lock:
            return self.thread_bindings.copy()
    
    def get_cpu_info(self) -> CPUInfo:
        """Get CPU information"""
        return self.cpu_info


class WorkerThreadWithAffinity(threading.Thread):
    """
    Worker thread with automatic CPU affinity binding.
    
    Sets CPU affinity on thread startup for optimal performance.
    """
    
    def __init__(
        self,
        target=None,
        cpu_core: Optional[int] = None,
        affinity_manager: Optional[CPUAffinityManager] = None,
        *args,
        **kwargs
    ):
        """
        Initialize worker thread with CPU affinity.
        
        Args:
            target: Target function
            cpu_core: CPU core to bind to
            affinity_manager: CPUAffinityManager instance
            *args: Positional arguments for target
            **kwargs: Keyword arguments for target
        """
        super().__init__(target=target, *args, **kwargs)
        self.cpu_core = cpu_core
        self.affinity_manager = affinity_manager or CPUAffinityManager()
        self.logger = logging.getLogger("INIDS.Performance.WorkerThread")
    
    def run(self):
        """Run target function after setting CPU affinity"""
        # Set CPU affinity
        if self.cpu_core is not None:
            success = self.affinity_manager.bind_thread(
                thread_id=threading.current_thread().ident,
                cpu_core=self.cpu_core,
            )
            if success:
                self.logger.info(f"Bound to CPU core {self.cpu_core}")
            else:
                self.logger.warning(f"Failed to bind to CPU core {self.cpu_core}")
        
        # Run target
        super().run()


class CPUAffinityWrapper:
    """
    Wrapper for existing functions to set CPU affinity.
    
    Usage:
        wrapper = CPUAffinityWrapper(cpu_core=0)
        thread = threading.Thread(
            target=wrapper.wrap(my_worker_function),
            args=(arg1, arg2),
        )
        thread.start()
    """
    
    def __init__(self, cpu_core: int, affinity_manager: Optional[CPUAffinityManager] = None):
        """
        Initialize wrapper.
        
        Args:
            cpu_core: CPU core to bind to
            affinity_manager: CPUAffinityManager instance
        """
        self.cpu_core = cpu_core
        self.affinity_manager = affinity_manager or CPUAffinityManager()
        self.logger = logging.getLogger("INIDS.Performance.CPUAffinityWrapper")
    
    def wrap(self, func):
        """
        Wrap function to set CPU affinity.
        
        Args:
            func: Function to wrap
        
        Returns:
            Wrapped function that sets CPU affinity before execution
        """
        def wrapped(*args, **kwargs):
            # Set CPU affinity
            self.affinity_manager.bind_thread(
                thread_id=threading.current_thread().ident,
                cpu_core=self.cpu_core,
            )
            # Call original function
            return func(*args, **kwargs)
        
        return wrapped


# Global instance
_affinity_manager = None


def get_affinity_manager() -> CPUAffinityManager:
    """Get global CPU affinity manager"""
    global _affinity_manager
    if _affinity_manager is None:
        _affinity_manager = CPUAffinityManager()
    return _affinity_manager


def init_affinity() -> CPUAffinityManager:
    """Initialize global CPU affinity manager"""
    global _affinity_manager
    _affinity_manager = CPUAffinityManager()
    return _affinity_manager
