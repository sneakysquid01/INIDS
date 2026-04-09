"""
Phase C: Multi-Threaded Detection Engine
Lock-free worker pool with flow-based packet distribution
Inspired by Suricata's autofp (auto flow pinning) mode
"""

import threading
import queue
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Worker thread state"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class WorkerStats:
    """Statistics for a worker thread"""
    worker_id: int
    packets_processed: int = 0
    packets_with_flows: int = 0
    detections_made: int = 0
    flow_contexts_cached: int = 0
    
    # Timing (in seconds)
    uptime: float = 0.0
    total_processing_time: float = 0.0
    total_detection_time: float = 0.0
    
    # Performance
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # State
    state: WorkerState = WorkerState.STOPPED
    
    def get_throughput(self) -> float:
        """Get packets per second"""
        if self.uptime > 0:
            return self.packets_processed / self.uptime
        return 0.0
    
    def __repr__(self):
        return (f"WorkerStats(id={self.worker_id}, packets={self.packets_processed}, "
                f"throughput={self.get_throughput():.0f} pps)")


@dataclass
class WorkerPacketBatch:
    """Batch of packets for a worker"""
    batch_id: int
    packets: List[tuple] = field(default_factory=list)  # (flow_id, decoded_packet)
    flow_contexts: Dict[str, 'FlowContext'] = field(default_factory=dict)
    timestamps_received: float = 0.0
    
    def __len__(self):
        return len(self.packets)


class FlowHasher:
    """
    Deterministic flow hashing for worker assignment
    Maps 5-tuple to worker ID
    """
    
    @staticmethod
    def compute_flow_partition(flow_id: str, num_workers: int) -> int:
        """
        Compute worker partition for a flow ID
        
        Args:
            flow_id: 5-tuple flow identifier (from Phase A)
            num_workers: Number of worker threads
        
        Returns:
            Worker ID (0 to num_workers-1)
        """
        if num_workers <= 0:
            return 0
        
        # Use MD5 hash of flow ID (already computed in Phase A)
        # Convert hex to integer, modulo worker count
        hash_value = int(flow_id, 16) if len(flow_id) <= 16 else int(hashlib.md5(flow_id.encode()).hexdigest(), 16)
        return hash_value % num_workers
    
    @staticmethod
    def compute_flow_partition_from_tuple(src_ip: str, dst_ip: str, src_port: int, 
                                         dst_port: int, protocol: str, num_workers: int) -> int:
        """
        Compute worker partition from raw tuple (if flow_id not available)
        
        Args:
            src_ip, dst_ip, src_port, dst_port, protocol: 5-tuple components
            num_workers: Number of worker threads
        
        Returns:
            Worker ID
        """
        if num_workers <= 0:
            return 0
        
        # Create normalized 5-tuple
        if src_ip.lower() > dst_ip.lower():
            tuple_str = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        else:
            tuple_str = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        
        hash_value = int(hashlib.md5(tuple_str.encode()).hexdigest(), 16)
        return hash_value % num_workers


class DetectionWorker:
    """
    Single detection worker thread
    Processes packets from assigned flows (lock-free)
    """
    
    def __init__(self, worker_id: int, input_queue: queue.Queue, 
                 packet_decoder_class, flow_table_class,
                 detection_callbacks: List[Callable] = None):
        """
        Initialize detection worker
        
        Args:
            worker_id: Worker ID (0 to num_workers-1)
            input_queue: Queue for receiving packet batches
            packet_decoder_class: PacketDecoder class
            flow_table_class: FlowTable class
            detection_callbacks: Optional detection callbacks
        """
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.packet_decoder = packet_decoder_class()
        self.flow_table = flow_table_class()
        self.detection_callbacks = detection_callbacks or []
        
        # State
        self.state = WorkerState.INITIALIZING
        self.stats = WorkerStats(worker_id=worker_id)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Timing
        self.start_time: Optional[float] = None
        
        # Per-worker results
        self.detections_results = []
        self.results_lock = threading.Lock()
    
    def run(self):
        """Main worker loop (runs in thread)"""
        try:
            self.start_time = time.time()
            self.state = WorkerState.RUNNING
            self.running = True
            self.stats.state = WorkerState.RUNNING
            
            logger.info(f"Worker {self.worker_id} started")
            
            while self.running:
                try:
                    # Get packet batch with timeout
                    batch = self.input_queue.get(timeout=0.1)
                    
                    if batch is None:
                        # Sentinel value = stop signal
                        break
                    
                    # Process batch
                    self._process_batch(batch)
                    
                except queue.Empty:
                    # No packets available, continue waiting
                    continue
                
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} error: {e}")
                    continue
            
            self.state = WorkerState.STOPPED
            self.running = False
            logger.info(f"Worker {self.worker_id} stopped")
        
        except Exception as e:
            logger.error(f"Worker {self.worker_id} fatal error: {e}")
            self.state = WorkerState.STOPPED
    
    def _process_batch(self, batch: WorkerPacketBatch) -> None:
        """Process a batch of packets"""
        batch_start_time = time.time()
        
        for flow_id, decoded_packet in batch.packets:
            packet_start = time.time()
            
            try:
                # Get or create flow context
                flow_context = batch.flow_contexts.get(flow_id)
                
                if not flow_context:
                    # This shouldn't happen if batching is correct, but handle gracefully
                    logger.warning(f"Worker {self.worker_id}: Missing flow context for {flow_id}")
                    continue
                
                # Decode packet (already done in Phase A, but validate)
                if not decoded_packet:
                    continue
                
                self.stats.packets_processed += 1
                self.stats.packets_with_flows += 1
                
                # Update flow tracking
                self.flow_table.update_packet_stats(
                    flow_context,
                    len(decoded_packet.payload_data) if hasattr(decoded_packet, 'payload_data') else 0
                )
                
                # Run detection callbacks (protocol analysis + ML detection)
                detection_score = 0.0
                alert_reason = None
                
                for callback in self.detection_callbacks:
                    try:
                        score, reason = callback(flow_context, decoded_packet)
                        if score > detection_score:
                            detection_score = score
                            alert_reason = reason
                    except Exception as e:
                        logger.debug(f"Detection callback error: {e}")
                
                # Store results if detection made
                if detection_score > 0.5:
                    self.stats.detections_made += 1
                    
                    result = {
                        'flow_id': flow_id,
                        'detection_score': detection_score,
                        'alert_reason': alert_reason,
                        'timestamp': time.time(),
                        'worker_id': self.worker_id
                    }
                    
                    with self.results_lock:
                        self.detections_results.append(result)
                
                # Timing stats
                packet_latency = (time.time() - packet_start) * 1000  # ms
                if packet_latency > self.stats.max_latency_ms:
                    self.stats.max_latency_ms = packet_latency
                
            except Exception as e:
                logger.debug(f"Worker {self.worker_id} packet error: {e}")
                continue
        
        # Update batch timing
        batch_latency = time.time() - batch_start_time
        self.stats.total_processing_time += batch_latency
        if self.stats.packets_processed > 0:
            self.stats.avg_latency_ms = (self.stats.total_processing_time / self.stats.packets_processed) * 1000
        
        self.stats.flow_contexts_cached = len(self.flow_table.flows)
    
    def get_detections(self) -> List[Dict]:
        """Retrieve detections and clear buffer"""
        with self.results_lock:
            results = self.detections_results.copy()
            self.detections_results.clear()
        return results
    
    def start(self) -> None:
        """Start worker thread"""
        if self.thread is None:
            self.thread = threading.Thread(target=self.run, name=f"DetectionWorker-{self.worker_id}", daemon=False)
            self.thread.start()
    
    def stop(self) -> None:
        """Stop worker thread"""
        self.running = False
        self.state = WorkerState.STOPPING
        
        if self.thread:
            # Send sentinel
            self.input_queue.put(None)
            # Wait for thread to finish
            self.thread.join(timeout=5.0)
        
        self.state = WorkerState.STOPPED
    
    def get_stats(self) -> WorkerStats:
        """Get worker statistics"""
        if self.start_time:
            self.stats.uptime = time.time() - self.start_time
        return self.stats


class WorkerPool:
    """
    Thread pool for lock-free packet processing
    Distributes packets to workers based on flow hash
    """
    
    def __init__(self, num_workers: int = 4, queue_size: int = 1000,
                 packet_decoder_class=None, flow_table_class=None,
                 detection_callbacks: List[Callable] = None):
        """
        Initialize worker pool
        
        Args:
            num_workers: Number of worker threads
            queue_size: Size of input queue per worker
            packet_decoder_class: PacketDecoder for workers
            flow_table_class: FlowTable for workers
            detection_callbacks: Callbacks for detection layer
        """
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.detection_callbacks = detection_callbacks or []
        
        # Import Phase A classes if not provided
        if packet_decoder_class is None:
            from src.decoding import PacketDecoder
            packet_decoder_class = PacketDecoder
        
        if flow_table_class is None:
            from src.flow_tracking import FlowTable
            flow_table_class = FlowTable
        
        # Create worker threads
        self.workers: List[DetectionWorker] = []
        
        for i in range(num_workers):
            worker = DetectionWorker(
                worker_id=i,
                input_queue=queue.Queue(maxsize=queue_size),
                packet_decoder_class=packet_decoder_class,
                flow_table_class=flow_table_class,
                detection_callbacks=self.detection_callbacks
            )
            self.workers.append(worker)
        
        # Statistics
        self.pool_stats = {
            'total_packets': 0,
            'total_detections': 0,
            'batches_distributed': 0,
            'start_time': None,
        }
    
    def start(self) -> None:
        """Start all workers"""
        self.pool_stats['start_time'] = time.time()
        for worker in self.workers:
            worker.start()
        logger.info(f"Worker pool started with {self.num_workers} workers")
    
    def stop(self) -> None:
        """Stop all workers"""
        for worker in self.workers:
            worker.stop()
        logger.info("Worker pool stopped")
    
    def distribute_batch(self, packets_with_flows: List[Tuple[str, 'DecodedPacket']], 
                        flow_contexts: Dict[str, 'FlowContext']) -> None:
        """
        Distribute packets to workers based on flow hash
        
        Args:
            packets_with_flows: List of (flow_id, decoded_packet) tuples
            flow_contexts: Dict mapping flow_id to FlowContext
        """
        # Group packets by worker
        worker_packets: Dict[int, List[Tuple]] = {i: [] for i in range(self.num_workers)}
        
        for flow_id, decoded_packet in packets_with_flows:
            worker_id = FlowHasher.compute_flow_partition(flow_id, self.num_workers)
            worker_packets[worker_id].append((flow_id, decoded_packet))
        
        # Send batches to workers
        for worker_id, packets in worker_packets.items():
            if len(packets) > 0:
                batch = WorkerPacketBatch(
                    batch_id=self.pool_stats['batches_distributed'],
                    packets=packets,
                    flow_contexts={fid: flow_contexts[fid] for fid, _ in packets if fid in flow_contexts},
                    timestamps_received=time.time()
                )
                
                try:
                    self.workers[worker_id].input_queue.put_nowait(batch)
                    self.pool_stats['total_packets'] += len(packets)
                except queue.Full:
                    logger.warning(f"Worker {worker_id} queue full, dropping batch")
        
        self.pool_stats['batches_distributed'] += 1
    
    def collect_detections(self) -> List[Dict]:
        """Collect detections from all workers"""
        all_detections = []
        
        for worker in self.workers:
            detections = worker.get_detections()
            all_detections.extend(detections)
            self.pool_stats['total_detections'] += len(detections)
        
        return all_detections
    
    def get_stats(self) -> Dict:
        """Get pool statistics"""
        worker_stats = [worker.get_stats() for worker in self.workers]
        
        total_threads_packets = sum(w.packets_processed for w in worker_stats)
        total_threads_detections = sum(w.detections_made for w in worker_stats)
        avg_latency = sum(w.avg_latency_ms for w in worker_stats) / len(worker_stats) if worker_stats else 0
        
        return {
            'num_workers': self.num_workers,
            'pool_packets': self.pool_stats['total_packets'],
            'pool_detections': self.pool_stats['total_detections'],
            'batches_distributed': self.pool_stats['batches_distributed'],
            'worker_stats': worker_stats,
            'total_worker_packets': total_threads_packets,
            'total_worker_detections': total_threads_detections,
            'avg_worker_latency_ms': avg_latency,
            'uptime_seconds': (time.time() - self.pool_stats['start_time']) if self.pool_stats['start_time'] else 0,
        }
    
    def print_stats(self) -> None:
        """Pretty-print statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("WORKER POOL STATISTICS")
        print("="*70)
        print(f"Workers: {stats['num_workers']}")
        print(f"Uptime: {stats['uptime_seconds']:.1f}s")
        print(f"Pool packets: {stats['pool_packets']}")
        print(f"Total detections: {stats['pool_detections']}")
        print(f"Batches distributed: {stats['batches_distributed']}")
        print(f"Avg worker latency: {stats['avg_worker_latency_ms']:.2f}ms")
        print()
        
        for ws in stats['worker_stats']:
            print(f"  Worker {ws.worker_id}: {ws.packets_processed} packets, "
                  f"{ws.detections_made} detections, "
                  f"{ws.get_throughput():.0f} pps")
        
        print("="*70 + "\n")
