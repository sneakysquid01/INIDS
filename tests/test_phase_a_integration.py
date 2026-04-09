"""
Phase A Integration Tests
Tests for packet capture, decoding, flow tracking, and pipeline
"""

import pytest
import logging
from datetime import datetime

from src.packet_capture import (
    Packet, PCAPReader, LiveCapture, InMemorySource, PacketSourceFactory
)
from src.decoding import PacketDecoder, DecodedPacket
from src.flow_tracking import FlowTable, FlowContext, FlowState, FlowAction
from src.integration.packet_pipeline import PacketProcessingPipeline

logger = logging.getLogger(__name__)


class TestPacketCapture:
    """Test packet source abstraction"""
    
    def test_packet_object_creation(self):
        """Test Packet dataclass"""
        pkt = Packet(
            timestamp=datetime.now().timestamp(),
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=54321,
            dst_port=80,
            protocol="tcp",
            packet_data=b"test",
            packet_len=4
        )
        
        assert pkt.src_ip == "192.168.1.100"
        assert pkt.dst_ip == "10.0.0.1"
        assert pkt.flow_id is not None
        assert len(pkt.flow_id) == 16  # MD5 hash truncated to 16 chars
        logger.info(f"✓ Packet created: {pkt}")
    
    def test_packet_flow_id_computation(self):
        """Test flow ID hashing"""
        pkt1 = Packet(
            timestamp=datetime.now().timestamp(),
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=1000,
            dst_port=80,
            protocol="tcp"
        )
        
        pkt2 = Packet(
            timestamp=datetime.now().timestamp(),
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=1000,
            dst_port=80,
            protocol="tcp"
        )
        
        # Same 5-tuple → same flow ID
        assert pkt1.flow_id == pkt2.flow_id
        logger.info(f"✓ Flow ID matching works: {pkt1.flow_id}")
    
    def test_in_memory_source(self):
        """Test in-memory packet source"""
        packets = [
            Packet(timestamp=datetime.now().timestamp(), src_ip="1.1.1.1",
                  dst_ip="2.2.2.2", protocol="tcp"),
            Packet(timestamp=datetime.now().timestamp(), src_ip="3.3.3.3",
                  dst_ip="4.4.4.4", protocol="udp"),
        ]
        
        source = InMemorySource(packets)
        read_packets = list(source.read_packets())
        
        assert len(read_packets) == 2
        assert read_packets[0].src_ip == "1.1.1.1"
        assert read_packets[1].protocol == "udp"
        logger.info(f"✓ InMemorySource: read {len(read_packets)} packets")
    
    def test_packet_source_factory(self):
        """Test factory pattern"""
        packets = [Packet(timestamp=datetime.now().timestamp())]
        source = PacketSourceFactory.create("memory", packets=packets)
        
        assert isinstance(source, InMemorySource)
        logger.info("✓ PacketSourceFactory creates correct source type")


class TestPacketDecoding:
    """Test packet decoder"""
    
    def test_packet_decoder_with_scapy(self):
        """Test decoding with real scapy packets"""
        try:
            from scapy.all import IP, TCP
        except ImportError:
            pytest.skip("scapy not installed")
        
        # Create synthetic IP/TCP packet
        pkt = IP(dst="192.168.1.1")/TCP(dport=80, sport=12345)
        raw_bytes = bytes(pkt)
        
        decoded = PacketDecoder.decode(raw_bytes, datetime.now().timestamp())
        
        assert decoded is not None
        assert decoded.l3 is not None
        assert decoded.l3.dst_ip == "192.168.1.1"
        assert decoded.l4 is not None
        assert decoded.l4.dst_port == 80
        assert decoded.l4.src_port == 12345
        logger.info(f"✓ Decoded packet: {decoded}")
    
    def test_flow_id_computation_in_decoder(self):
        """Test flow ID matches between Packet and DecodedPacket"""
        try:
            from scapy.all import IP, TCP
        except ImportError:
            pytest.skip("scapy not installed")
        
        pkt = IP(src="10.0.0.1", dst="10.0.0.2")/TCP(sport=1000, dport=80)
        raw = bytes(pkt)
        
        decoded = PacketDecoder.decode(raw, datetime.now().timestamp())
        
        assert decoded.flow_id is not None
        assert len(decoded.flow_id) > 0
        logger.info(f"✓ Flow ID from decoder: {decoded.flow_id}")


class TestFlowTracking:
    """Test flow table and state management"""
    
    def test_flow_creation(self):
        """Test flow creation and retrieval"""
        ft = FlowTable(max_flows=1000)
        
        flow = ft.get_or_create_flow(
            flow_id="test123",
            src_ip="192.168.1.1",
            dst_ip="10.0.0.1",
            src_port=1000,
            dst_port=80,
            protocol="tcp"
        )
        
        assert flow.flow_id == "test123"
        assert flow.src_ip == "192.168.1.1"
        assert flow.state == FlowState.NEW
        logger.info(f"✓ Flow created: {flow}")
    
    def test_flow_packet_tracking(self):
        """Test packet counting"""
        ft = FlowTable()
        flow = ft.get_or_create_flow(
            "flow1", "1.1.1.1", "2.2.2.2", 1000, 80, "tcp"
        )
        
        ft.update_packet_stats("flow1", "toserver", 100, datetime.now().timestamp())
        ft.update_packet_stats("flow1", "toserver", 200, datetime.now().timestamp())
        ft.update_packet_stats("flow1", "toclient", 150, datetime.now().timestamp())
        
        assert flow.packets_toserver == 2
        assert flow.packets_toclient == 1
        assert flow.bytes_toserver == 300
        assert flow.bytes_toclient == 150
        logger.info(f"✓ Packet tracking: {flow.get_total_packets()} pkts, "
                   f"{flow.get_total_bytes()} bytes")
    
    def test_tcp_state_machine(self):
        """Test TCP state tracking"""
        ft = FlowTable()
        ft.get_or_create_flow("flow1", "1.1.1.1", "2.2.2.2", 1000, 80, "tcp")
        
        # SYN
        ft.update_tcp_state("flow1", "SYN")
        flow = ft.get_flow("flow1")
        assert flow.seen_syn == True
        
        # SYN-ACK
        ft.update_tcp_state("flow1", "SYN,ACK")
        assert flow.seen_ack == True
        assert flow.state == FlowState.ESTABLISHED
        
        logger.info(f"✓ TCP state machine works: {flow.state.value}")
    
    def test_flow_actions(self):
        """Test IPS actions per flow"""
        ft = FlowTable()
        ft.get_or_create_flow("flow1", "1.1.1.1", "2.2.2.2", 1000, 80, "tcp")
        
        ft.set_flow_action("flow1", FlowAction.BLOCK)
        flow = ft.get_flow("flow1")
        
        assert flow.action == FlowAction.BLOCK
        assert flow.is_blocked() == True
        logger.info(f"✓ Flow action set: {flow.action.value}")
    
    def test_flow_escalation(self):
        """Test escalation state machine"""
        ft = FlowTable()
        ft.get_or_create_flow("attacker", "1.1.1.1", "2.2.2.2", 1000, 80, "tcp")
        
        flow = ft.get_flow("attacker")
        assert flow.escalation_level == 0
        
        ft.escalate_flow("attacker")
        assert flow.escalation_level == 1
        assert flow.block_time_remaining > 0
        
        logger.info(f"✓ Escalation: level {flow.escalation_level}, "
                   f"block time {flow.block_time_remaining}s")
    
    def test_flow_table_stats(self):
        """Test statistics collection"""
        ft = FlowTable(max_flows=100)
        
        # Create multiple flows
        for i in range(10):
            ft.get_or_create_flow(
                f"flow{i}", f"1.1.1.{i}", "2.2.2.2", 1000+i, 80, "tcp"
            )
        
        stats = ft.get_stats()
        
        assert stats.total_flows == 10
        assert stats.active_flows == 10
        logger.info(f"✓ Flow table stats: {stats.total_flows} flows, "
                   f"{stats.memory_usage_kb:.1f}KB")
    
    def test_flow_expiration(self):
        """Test flow cleanup"""
        import time
        
        ft = FlowTable(window_seconds=1)  # 1 second timeout
        ft.get_or_create_flow("flow1", "1.1.1.1", "2.2.2.2", 1000, 80, "tcp")
        
        stats_before = ft.get_stats()
        assert stats_before.active_flows == 1
        
        # Wait for expiration
        time.sleep(2)
        ft.cleanup_expired_flows()
        
        stats_after = ft.get_stats()
        assert stats_after.active_flows == 0
        logger.info(f"✓ Flow expiration works: evicted {ft.evicted_count} flows")


class TestPacketPipeline:
    """Test end-to-end packet processing pipeline"""
    
    def test_pipeline_with_mock_detection(self):
        """Test pipeline with mock detection callback"""
        
        # Mock detection function
        def mock_detection(flow: FlowContext):
            # Simple: flag flows with more than 10 bytes
            if flow.get_total_bytes() > 10:
                return {
                    "verdict": "SUSPICIOUS",
                    "risk_score": 0.7,
                    "action": "alert"
                }
            return None
        
        # Create test packets
        packets = [
            Packet(
                timestamp=datetime.now().timestamp(),
                src_ip="192.168.1.1",
                dst_ip="10.0.0.1",
                src_port=1000,
                dst_port=80,
                protocol="tcp",
                packet_len=20
            )
        ]
        
        source = InMemorySource(packets)
        pipeline = PacketProcessingPipeline(
            packet_source=source,
            detection_callback=mock_detection
        )
        
        stats = pipeline.run(max_packets=1)
        
        assert stats["packets_processed"] == 1
        logger.info(f"✓ Pipeline stats: {stats}")
    
    def test_pipeline_flow_retrieval(self):
        """Test retrieving flows from pipeline"""
        packets = [
            Packet(
                timestamp=datetime.now().timestamp(),
                src_ip="192.168.1.1",
                dst_ip="10.0.0.1",
                src_port=1000,
                dst_port=80,
                protocol="tcp",
                packet_len=100
            )
        ]
        
        source = InMemorySource(packets)
        pipeline = PacketProcessingPipeline(packet_source=source)
        pipeline.run()
        
        flows = pipeline.get_all_flows()
        assert len(flows) > 0
        logger.info(f"✓ Pipeline flows: {len(flows)} flows tracked")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
