"""
Packet Processing Pipeline
Orchestrates packet ingestion → decoding → flow tracking → detection
Bridges packet-level analysis with existing ML detection engine
"""

import logging
from typing import Optional, Generator, Callable, Dict, Any
from datetime import datetime
import time

from src.packet_capture import Packet, PacketSource
from src.decoding import PacketDecoder, DecodedPacket
from src.flow_tracking import FlowTable, FlowContext, FlowAction

logger = logging.getLogger(__name__)


class PacketProcessingPipeline:
    """
    Main pipeline: Packet → Decode → Flow Track → Detection
    
    This bridges Suricata-like packet processing with INIDS ML detection.
    """
    
    def __init__(self, packet_source: PacketSource, flow_table: Optional[FlowTable] = None,
                 detection_callback: Optional[Callable] = None):
        """
        Initialize packet processing pipeline
        
        Args:
            packet_source: PacketSource instance (PCAP, live, etc.)
            flow_table: FlowTable for state tracking (auto-created if None)
            detection_callback: Function to call with each flow for detection
                               Signature: detection_callback(flow_context: FlowContext) -> dict
        """
        self.packet_source = packet_source
        self.flow_table = flow_table or FlowTable(window_seconds=300, max_flows=100000)
        self.detection_callback = detection_callback
        
        # Statistics
        self.packets_processed = 0
        self.packets_decoded = 0
        self.packets_failed = 0
        self.flows_created = 0
        self.detections_made = 0
        
        # Performance tracking
        self.decode_time_ms = 0
        self.detection_time_ms = 0
        
        logger.info("PacketProcessingPipeline initialized")
    
    def run(self, max_packets: int = 0, cleanup_interval: int = 100) -> Dict[str, Any]:
        """
        Process packets from source
        
        Args:
            max_packets: Max packets to process (0 = unlimited)
            cleanup_interval: Run flow cleanup every N packets
        
        Returns:
            Statistics dict
        """
        logger.info(f"Pipeline starting: max_packets={max_packets}")
        
        try:
            packet_count = 0
            
            for packet in self.packet_source.read_packets():
                try:
                    # Process flow
                    result = self._process_packet(packet)
                    
                    packet_count += 1
                    self.packets_processed += 1
                    
                    # Periodic cleanup
                    if packet_count % cleanup_interval == 0:
                        self.flow_table.cleanup_expired_flows()
                        logger.debug(f"Processed {packet_count} packets, "
                                   f"active flows: {len(self.flow_table.get_active_flows())}")
                    
                    # Stop if max reached
                    if max_packets > 0 and packet_count >= max_packets:
                        logger.info(f"Reached max_packets limit: {max_packets}")
                        break
                
                except Exception as e:
                    logger.warning(f"Error processing packet {packet_count}: {e}")
                    self.packets_failed += 1
                    continue
            
            self.packet_source.close()
            return self._get_stats()
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return self._get_stats()
    
    def _process_packet(self, packet: Packet) -> Optional[Dict[str, Any]]:
        """
        Process single packet through full pipeline
        
        Returns:
            Detection result dict (if detection made) or None
        """
        start_time = time.time()
        
        # Step 1: Decode packet
        decoded = PacketDecoder.decode(packet.packet_data, packet.timestamp)
        decode_time = (time.time() - start_time) * 1000
        self.decode_time_ms = decode_time
        
        if not decoded or not decoded.l3:
            logger.debug(f"Failed to decode packet")
            self.packets_failed += 1
            return None
        
        self.packets_decoded += 1
        
        # Step 2: Extract flow info
        flow_id = decoded.flow_id
        flow = self.flow_table.get_or_create_flow(
            flow_id=flow_id,
            src_ip=decoded.l3.src_ip,
            dst_ip=decoded.l3.dst_ip,
            src_port=decoded.l4.src_port if decoded.l4 else 0,
            dst_port=decoded.l4.dst_port if decoded.l4 else 0,
            protocol=decoded.l4.protocol if decoded.l4 else "other"
        )
        
        # Step 3: Update flow state
        direction = "toserver"  # Simplified: could enhance with reverse flow detection
        self.flow_table.update_packet_stats(flow_id, direction, 
                                           decoded.payload_len, packet.timestamp)
        
        if decoded.l4 and decoded.l4.flags:
            self.flow_table.update_tcp_state(flow_id, decoded.l4.flags)
        
        # Step 4: Call detection callback if provided
        detection_result = None
        if self.detection_callback:
            det_start = time.time()
            try:
                detection_result = self.detection_callback(flow)
                detection_time = (time.time() - det_start) * 1000
                self.detection_time_ms = detection_time
                
                if detection_result:
                    self.detections_made += 1
                    logger.info(f"Detection: {flow} → {detection_result}")
                    
                    # Update flow with detection results
                    if "risk_score" in detection_result:
                        self.flow_table.set_risk_score(flow_id, detection_result["risk_score"])
                    
                    if "action" in detection_result:
                        action_str = detection_result["action"].upper()
                        try:
                            action = FlowAction[action_str]
                            self.flow_table.set_flow_action(flow_id, action)
                        except KeyError:
                            logger.debug(f"Unknown action: {action_str}")
            
            except Exception as e:
                logger.warning(f"Detection error: {e}")
        
        return detection_result
    
    def _get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        flow_stats = self.flow_table.get_stats()
        
        return {
            "packets_processed": self.packets_processed,
            "packets_decoded": self.packets_decoded,
            "packets_failed": self.packets_failed,
            "detections_made": self.detections_made,
            "decode_time_ms_avg": self.decode_time_ms,
            "detection_time_ms_avg": self.detection_time_ms,
            "flow_stats": {
                "total_flows_seen": flow_stats.total_flows,
                "active_flows": flow_stats.active_flows,
                "closed_flows": flow_stats.closed_flows,
                "evicted_flows": self.flow_table.evicted_count,
                "memory_usage_kb": flow_stats.memory_usage_kb,
            },
            "success_rate": (self.packets_decoded / self.packets_processed * 100) 
                           if self.packets_processed > 0 else 0.0
        }
    
    def get_flow_context(self, flow_id: str) -> Optional[FlowContext]:
        """Get flow by ID (for debugging/inspection)"""
        return self.flow_table.get_flow(flow_id)
    
    def get_all_flows(self) -> Dict[str, FlowContext]:
        """Get all flows"""
        return self.flow_table.get_all_flows()
    
    def print_summary(self):
        """Print processing summary"""
        stats = self._get_stats()
        logger.info("=" * 80)
        logger.info("PACKET PROCESSING PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Packets processed:     {stats['packets_processed']}")
        logger.info(f"Packets decoded:       {stats['packets_decoded']}")
        logger.info(f"Decode failures:       {stats['packets_failed']}")
        logger.info(f"Success rate:          {stats['success_rate']:.1f}%")
        logger.info(f"")
        logger.info(f"Detections made:       {stats['detections_made']}")
        logger.info(f"Avg decode time:       {stats['decode_time_ms_avg']:.2f}ms")
        logger.info(f"Avg detection time:    {stats['detection_time_ms_avg']:.2f}ms")
        logger.info(f"")
        logger.info(f"Flows created:         {stats['flow_stats']['total_flows_seen']}")
        logger.info(f"Active flows:          {stats['flow_stats']['active_flows']}")
        logger.info(f"Closed flows:          {stats['flow_stats']['closed_flows']}")
        logger.info(f"Evicted flows:         {stats['flow_stats']['evicted_flows']}")
        logger.info(f"Memory usage:          {stats['flow_stats']['memory_usage_kb']:.1f}KB")
        logger.info("=" * 80)
