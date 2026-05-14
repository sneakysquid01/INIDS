"""
INIDS Flow Aggregator Module

Aggregates alerts per flow, providing:
- Alert deduplication within time windows
- Flow-level statistics
- Alert batching for efficiency
- Automatic flow expiration
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .eve_json import EVEEvent, EventType


class AggregationMode(Enum):
    """Alert aggregation strategies"""
    PASS_THROUGH = "pass_through"     # All alerts
    UNIQUE_PER_MINUTE = "unique_per_minute"  # One alert per flow per minute
    UNIQUE_PER_HOUR = "unique_per_hour"      # One alert per flow per hour
    TOP_ALERT_PER_FLOW = "top_alert_per_flow"  # Highest-scoring alert per flow


@dataclass
class FlowAlertWindow:
    """Alert window for a flow"""
    flow_id: int
    alerts: List[EVEEvent] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    total_alerts: int = 0  # Count of all alerts (including deduplicated)
    seen_signatures: set = field(default_factory=set)
    max_score: float = 0.0
    max_score_alert: Optional[EVEEvent] = None
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if window has expired"""
        return (time.time() - self.last_update) > ttl_seconds
    
    def add_alert(self, event: EVEEvent) -> bool:
        """
        Add alert to window.
        
        Returns:
            True if alert should be forwarded (not deduplicated)
        """
        self.last_update = time.time()
        self.total_alerts += 1
        
        # Extract signature if alert
        sig_id = None
        if event.alert:
            sig_id = event.alert.signature_id

        # Track highest scoring alert
        score = event.metadata.get("detection_score", 0.0) if event.metadata else 0.0
        if score > self.max_score:
            self.max_score = score
            self.max_score_alert = event

        # Deduplicate by signature (if available)
        if sig_id and sig_id in self.seen_signatures:
            # Already seen this alert type
            return False
        if sig_id:
            self.seen_signatures.add(sig_id)

        self.alerts.append(event)
        return True


class FlowAggregator:
    """
    Aggregates alerts per flow with configurable deduplication.
    
    Thread-safe aggregation of events for efficient output.
    Prevents alert storms from overwhelming downstream consumers.
    """
    
    def __init__(
        self,
        mode: AggregationMode = AggregationMode.PASS_THROUGH,
        window_ttl_seconds: int = 3600,
        max_flows: int = 100000,
    ):
        self.mode = mode
        self.window_ttl = window_ttl_seconds
        self.max_flows = max_flows
        self.flows: Dict[int, FlowAlertWindow] = {}
        self.lock = threading.Lock()
        
        # Statistics
        self.total_events_in = 0
        self.total_events_out = 0
        self.total_deduplicated = 0
    
    def add_event(self, event: EVEEvent) -> bool:
        """
        Add event to aggregator.
        
        Args:
            event: EVE JSON event
        
        Returns:
            True if event should be forwarded (not deduplicated)
        """
        with self.lock:
            self.total_events_in += 1
            
            # Flow events and others always pass through
            if event.event_type != EventType.ALERT:
                self.total_events_out += 1
                return True
            
            # Get or create flow window
            flow_id = event.flow_id
            if flow_id not in self.flows:
                if len(self.flows) >= self.max_flows:
                    self._evict_expired_flows()
                self.flows[flow_id] = FlowAlertWindow(flow_id=flow_id)
            
            window = self.flows[flow_id]
            
            # Apply aggregation mode
            if self.mode == AggregationMode.PASS_THROUGH:
                self.total_events_out += 1
                window.add_alert(event)
                return True
            
            elif self.mode == AggregationMode.UNIQUE_PER_MINUTE:
                # Check if window expired
                if (time.time() - window.last_update) > 60:
                    window.seen_signatures.clear()
                    window.total_alerts = 0
                
                # Add if new signature
                if window.add_alert(event):
                    self.total_events_out += 1
                    return True
                else:
                    self.total_deduplicated += 1
                    return False
            
            elif self.mode == AggregationMode.UNIQUE_PER_HOUR:
                # Check if window expired
                if (time.time() - window.last_update) > 3600:
                    window.seen_signatures.clear()
                    window.total_alerts = 0
                
                # Add if new signature
                if window.add_alert(event):
                    self.total_events_out += 1
                    return True
                else:
                    self.total_deduplicated += 1
                    return False
            
            elif self.mode == AggregationMode.TOP_ALERT_PER_FLOW:
                # Only keep highest scoring alert
                max_event = window.max_score_alert
                
                if window.add_alert(event):
                    # Return if this is the new max
                    if window.max_score_alert is event or max_event is None:
                        self.total_events_out += 1
                        return True
                    else:
                        self.total_deduplicated += 1
                        return False
                else:
                    self.total_deduplicated += 1
                    return False
            
            else:
                # Unknown mode, pass through
                self.total_events_out += 1
                window.add_alert(event)
                return True
    
    def get_flow_stats(self, flow_id: int) -> Optional[Dict]:
        """Get aggregation stats for a flow"""
        with self.lock:
            if flow_id not in self.flows:
                return None
            
            window = self.flows[flow_id]
            return {
                "flow_id": flow_id,
                "alert_count": len(window.alerts),
                "total_alerts_seen": window.total_alerts,
                "unique_signatures": len(window.seen_signatures),
                "max_score": window.max_score,
                "last_update": window.last_update,
            }
    
    def get_aggregation_stats(self) -> Dict:
        """Get overall aggregation statistics"""
        with self.lock:
            return {
                "mode": self.mode.value,
                "total_events_in": self.total_events_in,
                "total_events_out": self.total_events_out,
                "total_deduplicated": self.total_deduplicated,
                "dedup_ratio": (
                    self.total_deduplicated / self.total_events_in
                    if self.total_events_in > 0 else 0.0
                ),
                "active_flows": len(self.flows),
                "max_flows": self.max_flows,
            }
    
    def get_pending_alerts(self, flow_id: int) -> List[EVEEvent]:
        """Get pending alerts for a flow"""
        with self.lock:
            if flow_id not in self.flows:
                return []
            
            alerts = self.flows[flow_id].alerts.copy()
            self.flows[flow_id].alerts.clear()
            return alerts
    
    def flush_all_alerts(self) -> List[EVEEvent]:
        """Flush all pending alerts from all flows"""
        with self.lock:
            alerts = []
            for window in self.flows.values():
                alerts.extend(window.alerts)
                window.alerts.clear()
            return alerts
    
    def cleanup_expired(self) -> int:
        """Remove expired flow windows"""
        with self.lock:
            return self._evict_expired_flows()
    
    def _evict_expired_flows(self) -> int:
        """Internal: Remove expired flows"""
        expired_flows = [
            flow_id for flow_id, window in self.flows.items()
            if window.is_expired(self.window_ttl)
        ]
        
        for flow_id in expired_flows:
            del self.flows[flow_id]
        
        return len(expired_flows)
    
    def reset_stats(self) -> None:
        """Reset statistics counters"""
        with self.lock:
            self.total_events_in = 0
            self.total_events_out = 0
            self.total_deduplicated = 0


class OutputPipeline:
    """
    Complete output pipeline:
    Aggregation → EVE JSON formatting → Backend delivery
    
    Coordinates aggregation, formatting, and delivery of alerts
    from detection pipeline to output backends.
    """
    
    def __init__(
        self,
        aggregator: FlowAggregator,
        output_aggregator,  # OutputAggregator from output_backends
        batch_size: int = 100,
        batch_timeout_seconds: float = 5.0,
    ):
        self.aggregator = aggregator
        self.output_aggregator = output_aggregator
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_seconds
        
        self.event_batch = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
        
        # Statistics
        self.events_processed = 0
        self.events_aggregated = 0
        self.events_dropped = 0
    
    def process_event(self, event: EVEEvent) -> bool:
        """
        Process and output event.
        
        Args:
            event: EVE JSON event from detection pipeline
        
        Returns:
            True if event was accepted
        """
        with self.lock:
            self.events_processed += 1
            
            # Apply aggregation
            if not self.aggregator.add_event(event):
                self.events_dropped += 1
                return False
            
            # Add to batch
            self.event_batch.append(event)
            
            # Check if should flush
            should_flush = (
                len(self.event_batch) >= self.batch_size or
                (time.time() - self.last_flush) > self.batch_timeout
            )
            
            if should_flush:
                return self._flush_batch()
            
            return True
    
    def _flush_batch(self) -> bool:
        """Internal: Flush current batch"""
        if not self.event_batch:
            return True
        
        batch = self.event_batch.copy()
        self.event_batch.clear()
        self.last_flush = time.time()
        
        # Send to output backends
        sent = self.output_aggregator.send_events(batch)
        self.events_aggregated += sent
        
        return sent > 0
    
    def flush(self) -> bool:
        """Flush any pending events"""
        with self.lock:
            return self._flush_batch()
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        with self.lock:
            return {
                "events_processed": self.events_processed,
                "events_aggregated": self.events_aggregated,
                "events_dropped": self.events_dropped,
                "pending_in_batch": len(self.event_batch),
                "aggregator_stats": self.aggregator.get_aggregation_stats(),
                "output_stats": self.output_aggregator.get_stats(),
            }
    
    def close(self) -> None:
        """Close and flush pipeline"""
        self.flush()


class AlertThrottler:
    """
    Throttles alerts to prevent storms.
    
    Limits alerts per flow and globally.
    """
    
    def __init__(
        self,
        max_alerts_per_flow_per_second: int = 10,
        max_alerts_per_second: int = 1000,
    ):
        self.max_per_flow = max_alerts_per_flow_per_second
        self.max_global = max_alerts_per_second
        
        self.flow_counters: Dict[int, Tuple[int, float]] = {}  # flow_id -> (count, timestamp)
        self.global_counter = 0
        self.global_timestamp = time.time()
        self.lock = threading.Lock()
    
    def should_rate_limit(self, event: EVEEvent) -> bool:
        """
        Check if event should be rate limited.
        
        Returns:
            True if event should be dropped
        """
        with self.lock:
            current_time = time.time()
            flow_id = event.flow_id
            
            # Reset global counter if needed
            if (current_time - self.global_timestamp) >= 1.0:
                self.global_counter = 0
                self.global_timestamp = current_time
            
            # Check global limit
            if self.global_counter >= self.max_global:
                return True
            
            # Check per-flow limit
            if flow_id in self.flow_counters:
                flow_count, flow_time = self.flow_counters[flow_id]
                if (current_time - flow_time) >= 1.0:
                    # Reset per-flow counter
                    self.flow_counters[flow_id] = (1, current_time)
                else:
                    if flow_count >= self.max_per_flow:
                        return True
                    self.flow_counters[flow_id] = (flow_count + 1, flow_time)
            else:
                self.flow_counters[flow_id] = (1, current_time)
            
            self.global_counter += 1
            return False
    
    def get_stats(self) -> Dict:
        """Get throttler statistics"""
        with self.lock:
            return {
                "global_per_second": self.global_counter,
                "max_per_flow_per_second": self.max_per_flow,
                "max_global_per_second": self.max_global,
                "active_flows": len(self.flow_counters),
            }
