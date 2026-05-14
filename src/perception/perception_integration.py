"""Perception Layer Real-time Integration

Connects perception engines to the EventBus for real-time updates.
Manages backpressure, queuing, and latency optimization.
"""

import time
import logging
from typing import Any, Callable, Optional
from threading import Thread, Lock, Event
from queue import Queue, Full
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from src.core.event_bus import DetectionEvent, EventBus
from src.perception.attack_story import AttackStoryEngine
from src.perception.confidence_breakdown import ConfidenceBreakdownEngine
from src.perception.live_pulse import LiveSystemPulse

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Track real-time latency from detection to perception update"""
    event_received_at: float  # timestamp
    processing_started_at: float
    processing_completed_at: float
    perception_updated_at: float
    
    @property
    def total_latency_ms(self) -> float:
        """Total time from event receipt to perception update (ms)"""
        return (self.perception_updated_at - self.event_received_at) * 1000
    
    @property
    def processing_latency_ms(self) -> float:
        """Time spent processing the event (ms)"""
        return (self.processing_completed_at - self.processing_started_at) * 1000
    
    @property
    def perception_latency_ms(self) -> float:
        """Time to update perception engines (ms)"""
        return (self.perception_updated_at - self.processing_completed_at) * 1000


class PerceptionIntegration:
    """Real-time integration of perception engines with EventBus
    
    Responsibilities:
    - Subscribe to detection events
    - Feed detections to perception engines
    - Manage event queue (backpressure handling)
    - Track latency metrics
    - Graceful degradation under load
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        attack_story_engine: AttackStoryEngine,
        confidence_breakdown_engine: ConfidenceBreakdownEngine,
        live_system_pulse: LiveSystemPulse,
        queue_size: int = 1000,
        batch_size: int = 10,
        worker_threads: int = 2
    ):
        self.event_bus = event_bus
        self.attack_story_engine = attack_story_engine
        self.confidence_breakdown_engine = confidence_breakdown_engine
        self.live_system_pulse = live_system_pulse
        
        # Event queue with backpressure management
        self.event_queue: Queue = Queue(maxsize=queue_size)
        self.batch_size = batch_size
        self.worker_threads = worker_threads
        self.stop_event = Event()
        
        # Metrics tracking
        self._latency_metrics: list[LatencyMetrics] = []
        self._metrics_lock = Lock()
        self._events_processed = 0
        self._events_dropped = 0
        self._processing_start_time = time.time()
        
        # Worker threads
        self._workers = []
        
    def start(self) -> None:
        """Start perception integration (subscribe to EventBus, start workers)"""
        logger.info("Starting perception integration...")
        
        # Subscribe to detection events
        self.event_bus.subscribe(DetectionEvent, self._handle_detection_event)
        logger.info("Subscribed to DetectionEvent from EventBus")
        
        # Start worker threads for processing events
        self.stop_event.clear()
        for i in range(self.worker_threads):
            worker = Thread(
                target=self._worker_loop,
                name=f"PerceptionWorker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {self.worker_threads} perception worker threads")
    
    def stop(self, timeout_seconds: int = 10) -> None:
        """Stop perception integration gracefully"""
        logger.info("Stopping perception integration...")
        
        # Signal workers to stop
        self.stop_event.set()
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=timeout_seconds / len(self._workers))
        
        logger.info(f"Perception integration stopped. Processed {self._events_processed} events")
    
    def _handle_detection_event(self, event: DetectionEvent) -> None:
        """Handle incoming detection event (from EventBus)
        
        Called by EventBus when a detection occurs. Queues event for processing
        with backpressure handling.
        """
        event_received_at = time.time()
        
        try:
            # Try to add event to queue (non-blocking with timeout)
            self.event_queue.put(
                (event, event_received_at),
                block=False,
                timeout=0.1
            )
        except Full:
            # Queue is full - apply backpressure (sample/drop)
            with self._metrics_lock:
                self._events_dropped += 1
            logger.warning(
                f"Perception queue full. Dropped event from {event.source_ip}. "
                f"(Dropped: {self._events_dropped})"
            )
    
    def _worker_loop(self) -> None:
        """Worker thread main loop - processes events from queue"""
        logger.debug(f"Perception worker started")
        
        batch = []
        while not self.stop_event.is_set():
            try:
                # Collect batch of events (with timeout for graceful shutdown)
                try:
                    event, event_received_at = self.event_queue.get(timeout=0.5)
                    batch.append((event, event_received_at))
                except:
                    # Queue timeout - process partial batch
                    if not batch:
                        continue
                
                # Process batch when full or on timeout
                if len(batch) >= self.batch_size or (
                    self.stop_event.is_set() and batch
                ):
                    self._process_batch(batch)
                    batch = []
                    
            except Exception:
                logger.exception("Error in perception worker loop")
    
    def _process_batch(self, batch: list[tuple[DetectionEvent, float]]) -> None:
        """Process a batch of events and update perception engines
        
        Converts DetectionEvents to format expected by perception engines,
        updates all three engines, and tracks latency.
        """
        processing_started_at = time.time()
        
        try:
            for event, event_received_at in batch:
                # Convert DetectionEvent to detection dict for perception engines
                detection_data = {
                    'id': f"{event.source_ip}_{event.timestamp}",
                    'timestamp': event.timestamp,
                    'source_ip': event.source_ip,
                    'prediction': event.prediction,
                    'confidence': event.confidence,
                    'attack_type': event.attack_type,
                    'severity': event.severity,
                    'features': event.features,
                    'reason': event.reason,
                }
                
                # Update confidence breakdown engine
                self.confidence_breakdown_engine.analyze_detection(detection_data)
                
                # Update attack story engine
                self.attack_story_engine.store_story(event.source_ip, detection_data)
                
                # Update live pulse metrics
                self.live_system_pulse.record_alert()
                
                # Record latency
                processing_completed_at = time.time()
                perception_updated_at = time.time()
                
                metrics = LatencyMetrics(
                    event_received_at=event_received_at,
                    processing_started_at=processing_started_at,
                    processing_completed_at=processing_completed_at,
                    perception_updated_at=perception_updated_at,
                )
                
                with self._metrics_lock:
                    self._latency_metrics.append(metrics)
                    self._events_processed += 1
                    
                    # Keep only last 1000 metrics to avoid memory growth
                    if len(self._latency_metrics) > 1000:
                        self._latency_metrics = self._latency_metrics[-1000:]
                
                if self._events_processed % 100 == 0:
                    avg_latency = self.get_average_latency_ms()
                    logger.debug(
                        f"Perception integration: {self._events_processed} events processed, "
                        f"avg latency {avg_latency:.1f}ms"
                    )
        
        except Exception:
            logger.exception("Error processing perception batch")
    
    def get_average_latency_ms(self) -> float:
        """Get average latency from recent events (ms)"""
        with self._metrics_lock:
            if not self._latency_metrics:
                return 0.0
            avg = sum(m.total_latency_ms for m in self._latency_metrics) / len(self._latency_metrics)
            return avg
    
    def get_p95_latency_ms(self) -> float:
        """Get 95th percentile latency (ms)"""
        with self._metrics_lock:
            if not self._latency_metrics:
                return 0.0
            sorted_latencies = sorted([m.total_latency_ms for m in self._latency_metrics])
            idx = int(len(sorted_latencies) * 0.95)
            return sorted_latencies[idx] if idx < len(sorted_latencies) else 0.0
    
    def get_p99_latency_ms(self) -> float:
        """Get 99th percentile latency (ms)"""
        with self._metrics_lock:
            if not self._latency_metrics:
                return 0.0
            sorted_latencies = sorted([m.total_latency_ms for m in self._latency_metrics])
            idx = int(len(sorted_latencies) * 0.99)
            return sorted_latencies[idx] if idx < len(sorted_latencies) else 0.0
    
    def get_status(self) -> dict[str, Any]:
        """Get current integration status including metrics"""
        uptime_seconds = time.time() - self._processing_start_time
        queue_size = self.event_queue.qsize()
        
        with self._metrics_lock:
            throughput = self._events_processed / uptime_seconds if uptime_seconds > 0 else 0
            
        return {
            'status': 'running' if not self.stop_event.is_set() else 'stopped',
            'events_processed': self._events_processed,
            'events_dropped': self._events_dropped,
            'queue_size': queue_size,
            'queue_max_size': self.event_queue.maxsize,
            'throughput_events_per_second': throughput,
            'uptime_seconds': uptime_seconds,
            'latency_ms': {
                'average': self.get_average_latency_ms(),
                'p95': self.get_p95_latency_ms(),
                'p99': self.get_p99_latency_ms(),
            },
            'worker_threads': self.worker_threads,
            'batch_size': self.batch_size,
        }
