"""
Packet Distribution Engine
Lock-free distribution of packets to worker threads
Flow-pinned (autofp) mode: each flow always goes to same worker
"""

import threading
import logging
from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class DistributionStats:
    """Statistics for packet distribution"""
    total_packets_queued: int = 0
    total_batches_created: int = 0
    packets_in_batch: int = 0
    
    flow_count: int = 0
    flow_unique_seen: set = field(default_factory=set)
    
    average_batch_size: float = 0.0
    max_batch_size: int = 0
    
    distribution_time_ms: float = 0.0
    
    def __repr__(self):
        return (f"DistributionStats(packets={self.total_packets_queued}, "
                f"batches={self.total_batches_created}, "
                f"flows={self.flow_count})")


class PacketDistributor:
    """
    Distributes packets to worker pool based on flow affinity
    Lock-free distribution (packets with same flow → same worker)
    """
    
    def __init__(self, worker_pool, batch_timeout_ms: int = 100, batch_size: int = 128):
        """
        Initialize packet distributor
        
        Args:
            worker_pool: WorkerPool instance
            batch_timeout_ms: Max time to wait before sending batch
            batch_size: Max packets per batch before sending
        """
        self.worker_pool = worker_pool
        self.batch_timeout_ms = batch_timeout_ms
        self.batch_size = batch_size
        
        # Current batch being accumulated
        self.current_batch: List[Tuple[str, 'DecodedPacket']] = []
        self.current_batch_flow_contexts: Dict[str, 'FlowContext'] = {}
        self.batch_start_time: Optional[float] = None
        
        # Batch lock (only one thread builds batch at a time)
        self.batch_lock = threading.Lock()
        
        # Statistics
        self.stats = DistributionStats()
    
    def queue_packet(self, flow_id: str, decoded_packet: 'DecodedPacket', 
                    flow_context: 'FlowContext') -> None:
        """
        Queue a packet for distribution
        
        Args:
            flow_id: Flow identifier (5-tuple hash)
            decoded_packet: DecodedPacket from Phase A
            flow_context: FlowContext from Phase A (contains flow state + features)
        """
        with self.batch_lock:
            # Initialize batch on first packet
            if not self.current_batch:
                self.batch_start_time = time.time()
            
            # Add packet to batch
            self.current_batch.append((flow_id, decoded_packet))
            self.current_batch_flow_contexts[flow_id] = flow_context
            self.stats.total_packets_queued += 1
            self.stats.flow_unique_seen.add(flow_id)
            
            # Check if batch is full or timed out
            should_flush = self._should_flush_batch()
            
            if should_flush:
                self._flush_batch()
    
    def _should_flush_batch(self) -> bool:
        """Check if batch should be sent to workers"""
        if not self.current_batch:
            return False
        
        # Check size threshold
        if len(self.current_batch) >= self.batch_size:
            return True
        
        # Check timeout
        if self.batch_start_time:
            elapsed_ms = (time.time() - self.batch_start_time) * 1000
            if elapsed_ms >= self.batch_timeout_ms:
                return True
        
        return False
    
    def _flush_batch(self) -> None:
        """Send batch to worker pool"""
        if not self.current_batch:
            return
        
        # Distribute to workers
        self.worker_pool.distribute_batch(self.current_batch, self.current_batch_flow_contexts)
        
        # Update stats
        batch_size = len(self.current_batch)
        self.stats.total_batches_created += 1
        self.stats.packets_in_batch = batch_size
        self.stats.flow_count = len(self.current_batch_flow_contexts)
        self.stats.max_batch_size = max(self.stats.max_batch_size, batch_size)
        
        if self.stats.total_batches_created > 1:
            self.stats.average_batch_size = (
                (self.stats.average_batch_size * (self.stats.total_batches_created - 1) + batch_size)
                / self.stats.total_batches_created
            )
        else:
            self.stats.average_batch_size = batch_size
        
        # Clear batch
        self.current_batch = []
        self.current_batch_flow_contexts = {}
        self.batch_start_time = None
        
        logger.debug(f"Batch flushed: {batch_size} packets to {self.worker_pool.num_workers} workers")
    
    def flush_pending(self) -> None:
        """Flush any pending packets"""
        with self.batch_lock:
            self._flush_batch()
    
    def get_stats(self) -> DistributionStats:
        """Get distributor statistics"""
        self.stats.flow_count = len(self.stats.flow_unique_seen)
        return self.stats


class MultiLayerFeatureAggregator:
    """
    Aggregate features from all packets and flows
    Combines Phase A (packet/flow) + Phase B (protocol) + Phase C (detection) features
    """
    
    @staticmethod
    def aggregate_packet_features(decoded_packet) -> Dict[str, any]:
        """Extract features from decoded packet"""
        features = {}
        
        if not decoded_packet:
            return features
        
        # L2 features
        if decoded_packet.l2_info:
            features['pkt_has_vlan'] = decoded_packet.l2_info.vlan_tag is not None
            features['pkt_ethertype'] = decoded_packet.l2_info.ethertype
        
        # L3 features
        if decoded_packet.l3_info:
            features['pkt_src_ip'] = decoded_packet.l3_info.src_ip
            features['pkt_dst_ip'] = decoded_packet.l3_info.dst_ip
            features['pkt_ttl'] = decoded_packet.l3_info.ttl
            features['pkt_ip_flags'] = decoded_packet.l3_info.flags
        
        # L4 features
        if decoded_packet.l4_info:
            features['pkt_src_port'] = decoded_packet.l4_info.src_port
            features['pkt_dst_port'] = decoded_packet.l4_info.dst_port
            features['pkt_protocol'] = decoded_packet.l4_info.protocol
            
            if hasattr(decoded_packet.l4_info, 'tcp_flags'):
                features['pkt_tcp_flags'] = decoded_packet.l4_info.tcp_flags
        
        # Payload
        features['pkt_payload_size'] = len(decoded_packet.payload_data) if decoded_packet.payload_data else 0
        
        return features
    
    @staticmethod
    def aggregate_flow_features(flow_context) -> Dict[str, any]:
        """Extract features from flow context (Phase A)"""
        features = {}
        
        if not flow_context:
            return features
        
        # Flow identification
        features['flow_src_ip'] = flow_context.src_ip if hasattr(flow_context, 'src_ip') else ""
        features['flow_dst_ip'] = flow_context.dst_ip if hasattr(flow_context, 'dst_ip') else ""
        features['flow_src_port'] = flow_context.src_port if hasattr(flow_context, 'src_port') else 0
        features['flow_dst_port'] = flow_context.dst_port if hasattr(flow_context, 'dst_port') else 0
        features['flow_protocol'] = flow_context.protocol if hasattr(flow_context, 'protocol') else ""
        
        # Flow state
        features['flow_state'] = str(flow_context.state) if hasattr(flow_context, 'state') else ""
        features['flow_packets_count'] = flow_context.packet_count if hasattr(flow_context, 'packet_count') else 0
        features['flow_bytes_toserver'] = flow_context.bytes_toserver if hasattr(flow_context, 'bytes_toserver') else 0
        features['flow_bytes_toclient'] = flow_context.bytes_toclient if hasattr(flow_context, 'bytes_toclient') else 0
        
        # Flow action
        features['flow_action'] = str(flow_context.action) if hasattr(flow_context, 'action') else "ALLOW"
        features['flow_escalation'] = flow_context.escalation_level if hasattr(flow_context, 'escalation_level') else 0
        
        # Flow duration
        if hasattr(flow_context, 'get_duration'):
            features['flow_duration_seconds'] = flow_context.get_duration()
        
        return features
    
    @staticmethod
    def aggregate_protocol_features(flow_context) -> Dict[str, any]:
        """Extract protocol features from flow context (Phase B)"""
        features = {}
        
        if not flow_context or not hasattr(flow_context, 'features_cache'):
            return features
        
        proto_analysis = flow_context.features_cache.get('protocol_analysis')
        if not proto_analysis:
            return features
        
        # Protocol identification
        if hasattr(proto_analysis, 'detected_protocol'):
            features['proto_protocol'] = str(proto_analysis.detected_protocol)
            features['proto_confidence'] = proto_analysis.classification_confidence
        
        # Protocol-specific features
        if hasattr(proto_analysis, 'ml_features'):
            features.update(proto_analysis.ml_features)
        
        # Suspicious indicators
        if hasattr(proto_analysis, 'is_suspicious'):
            features['proto_is_suspicious'] = proto_analysis.is_suspicious
            if hasattr(proto_analysis, 'suspicious_indicators'):
                features['proto_indicators_count'] = len(proto_analysis.suspicious_indicators)
                features['proto_indicators'] = proto_analysis.suspicious_indicators[:5]  # Top 5
        
        return features
    
    @staticmethod
    def aggregate_all_features(decoded_packet, flow_context) -> Dict[str, any]:
        """Combine all feature types"""
        features = {}
        
        # Layer by layer
        features.update(MultiLayerFeatureAggregator.aggregate_packet_features(decoded_packet))
        features.update(MultiLayerFeatureAggregator.aggregate_flow_features(flow_context))
        features.update(MultiLayerFeatureAggregator.aggregate_protocol_features(flow_context))
        
        return features


class WorkerDetectionCallback:
    """
    Detection callback factory for worker threads
    Executed for each packet in worker context
    """
    
    @staticmethod
    def create_detection_callback(ml_models=None, protocol_analyzer=None) -> Callable:
        """
        Create a detection callback for workers
        
        Args:
            ml_models: Tuple of (ensemble_model, risk_scorer) or None
            protocol_analyzer: Protocol analyzer instance or None
        
        Returns:
            Callback function: (flow_context, decoded_packet) → (score, reason)
        """
        def detection_callback(flow_context, decoded_packet) -> Tuple[float, Optional[str]]:
            """
            Detect threats in packet
            
            Args:
                flow_context: Flow context with state
                decoded_packet: Decoded packet
            
            Returns:
                Tuple of (detection_score, alert_reason) or (0.0, None)
            """
            try:
                # Run protocol-layer analysis (Phase B)
                if protocol_analyzer:
                    from src.protocol_parsers.phase_b_integration import ProtocolAnalyzer
                    proto_ctx = ProtocolAnalyzer.analyze_packet_protocol(flow_context, decoded_packet)
                    
                    if proto_ctx and proto_ctx.is_suspicious:
                        # Protocol layer detected threat
                        return min(0.99, len(proto_ctx.suspicious_indicators) * 0.2), \
                               f"Protocol anomaly: {', '.join(proto_ctx.suspicious_indicators[:2])}"
                
                # Aggregate features from all layers
                features = MultiLayerFeatureAggregator.aggregate_all_features(decoded_packet, flow_context)
                
                # Run ML detection if available
                if ml_models:
                    ensemble_model, risk_scorer = ml_models
                    
                    # Predict risk
                    try:
                        risk_score = ensemble_model.predict([features])[0] if hasattr(ensemble_model, 'predict') else 0.0
                        
                        if risk_score > 0.5:
                            return risk_score, f"ML detection: score={risk_score:.2f}"
                    except Exception as e:
                        logger.debug(f"ML prediction error: {e}")
                
                return 0.0, None
            
            except Exception as e:
                logger.debug(f"Detection callback error: {e}")
                return 0.0, None
        
        return detection_callback
