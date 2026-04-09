"""
Unified Packet Source Abstraction
Supports PCAP, live capture, NetFlow, and in-memory sources
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator, Optional, List
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class Packet:
    """Unified packet representation across all sources"""
    timestamp: float        # Unix timestamp
    src_mac: Optional[str] = None
    dst_mac: Optional[str] = None
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: str = "tcp"   # 'tcp', 'udp', 'icmp'
    packet_data: bytes = b""
    packet_len: int = 0
    flow_id: Optional[str] = None
    vlan_id: Optional[int] = None
    
    def __post_init__(self):
        """Auto-compute flow_id if not set"""
        if self.flow_id is None and self.src_ip and self.dst_ip:
            self.flow_id = self._compute_flow_id()
    
    def _compute_flow_id(self) -> str:
        """Compute 5-tuple hash"""
        key = f"{self.src_ip}:{self.src_port}-{self.dst_ip}:{self.dst_port}-{self.protocol}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def __repr__(self):
        return (f"Packet({self.src_ip}:{self.src_port} → "
                f"{self.dst_ip}:{self.dst_port} {self.protocol.upper()} "
                f"@ {datetime.fromtimestamp(self.timestamp).isoformat()})")


class PacketSource(ABC):
    """Abstract base class for all packet sources"""
    
    @abstractmethod
    def read_packets(self) -> Generator[Packet, None, None]:
        """
        Yield packets one at a time
        Subclasses implement for different sources (PCAP, live, NetFlow, etc.)
        """
        pass
    
    @abstractmethod
    def close(self):
        """Cleanup resources"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PCAPReader(PacketSource):
    """Read PCAP files (offline analysis)"""
    
    def __init__(self, filepath: str):
        """
        Initialize PCAP reader
        
        Args:
            filepath: Path to .pcap or .pcapng file
        """
        self.filepath = filepath
        self.packet_count = 0
        logger.info(f"PCAPReader initialized: {filepath}")
    
    def read_packets(self) -> Generator[Packet, None, None]:
        """Read packets from PCAP file using scapy"""
        try:
            from scapy.all import rdpcap
            from scapy.layers.l2 import Ether
            from scapy.layers.inet import IP, ICMP
            from scapy.layers.inet6 import IPv6
            from scapy.layers.inet import TCP, UDP
        except ImportError:
            logger.error("scapy not installed. Install with: pip install scapy")
            return
        
        try:
            packets = rdpcap(self.filepath)
            logger.info(f"Opened PCAP file: {self.filepath} ({len(packets)} packets)")
            
            for scapy_pkt in packets:
                try:
                    packet = self._parse_scapy_packet(scapy_pkt)
                    if packet:
                        self.packet_count += 1
                        yield packet
                except Exception as e:
                    logger.warning(f"Failed to parse packet {self.packet_count}: {e}")
                    continue
            
            logger.info(f"Finished reading PCAP: {self.packet_count} packets")
        
        except FileNotFoundError:
            logger.error(f"PCAP file not found: {self.filepath}")
        except Exception as e:
            logger.error(f"Error reading PCAP: {e}")
    
    def _parse_scapy_packet(self, pkt) -> Optional[Packet]:
        """Convert scapy packet to Packet object"""
        from scapy.layers.l2 import Ether
        from scapy.layers.inet import IP, ICMP, TCP, UDP
        
        try:
            timestamp = float(pkt.time)
            src_mac = dst_mac = None
            src_ip = dst_ip = src_port = dst_port = protocol = None
            payload_len = len(pkt)
            
            # L2 - Ethernet
            if Ether in pkt:
                eth = pkt[Ether]
                src_mac = eth.src
                dst_mac = eth.dst
            
            # L3 - IP
            if IP in pkt:
                ip = pkt[IP]
                src_ip = ip.src
                dst_ip = ip.dst
                protocol = "icmp" if ICMP in pkt else protocol
            
            # L4 - TCP/UDP
            if TCP in pkt:
                tcp = pkt[TCP]
                src_port = tcp.sport
                dst_port = tcp.dport
                protocol = "tcp"
            elif UDP in pkt:
                udp = pkt[UDP]
                src_port = udp.sport
                dst_port = udp.dport
                protocol = "udp"
            elif ICMP in pkt:
                protocol = "icmp"
            
            if src_ip and dst_ip:
                return Packet(
                    timestamp=timestamp,
                    src_mac=src_mac,
                    dst_mac=dst_mac,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port or 0,
                    dst_port=dst_port or 0,
                    protocol=protocol or "unknown",
                    packet_data=bytes(pkt),
                    packet_len=payload_len
                )
        except Exception as e:
            logger.debug(f"Failed to parse packet: {e}")
        
        return None
    
    def close(self):
        """No resources to cleanup for PCAP reader"""
        pass


class LiveCapture(PacketSource):
    """Capture packets from network interface in real-time"""
    
    def __init__(self, interface: str, packet_count: int = 0, 
                 filter_expr: str = None, timeout: int = None):
        """
        Initialize live packet capture
        
        Args:
            interface: Network interface name (e.g., 'eth0', 'Wi-Fi')
            packet_count: Max packets to capture (0 = unlimited)
            filter_expr: BPF filter expression (e.g., 'tcp port 80')
            timeout: Timeout in seconds
        """
        self.interface = interface
        self.packet_count = packet_count
        self.filter_expr = filter_expr
        self.timeout = timeout
        self.packets_captured = 0
        logger.info(f"LiveCapture initialized: {interface}")
    
    def read_packets(self) -> Generator[Packet, None, None]:
        """Capture live packets from network interface"""
        try:
            from scapy.all import sniff
            from scapy.layers.l2 import Ether
            from scapy.layers.inet import IP, TCP, UDP, ICMP
        except ImportError:
            logger.error("scapy not installed. Install with: pip install scapy")
            return
        
        try:
            logger.info(f"Starting live capture on {self.interface}")
            
            def packet_callback(pkt):
                try:
                    packet = self._parse_scapy_packet(pkt)
                    if packet:
                        self.packets_captured += 1
                        yield packet
                except Exception as e:
                    logger.warning(f"Failed to parse captured packet: {e}")
            
            # Use sniff for live capture
            sniff(
                iface=self.interface,
                prn=lambda pkt: next(packet_callback(pkt), None),
                filter=self.filter_expr,
                store=False,
                timeout=self.timeout,
                count=self.packet_count if self.packet_count > 0 else None,
            )
            
            logger.info(f"Capture stopped: {self.packets_captured} packets captured")
        
        except PermissionError:
            logger.error(f"Permission denied. Run with elevated privileges (sudo/admin)")
        except Exception as e:
            logger.error(f"Error capturing packets: {e}")
    
    def _parse_scapy_packet(self, pkt) -> Optional[Packet]:
        """Convert scapy packet to Packet object (same as PCAPReader)"""
        from scapy.layers.l2 import Ether
        from scapy.layers.inet import IP, ICMP, TCP, UDP
        
        try:
            timestamp = float(pkt.time)
            src_mac = dst_mac = None
            src_ip = dst_ip = src_port = dst_port = protocol = None
            
            if Ether in pkt:
                eth = pkt[Ether]
                src_mac = eth.src
                dst_mac = eth.dst
            
            if IP in pkt:
                ip = pkt[IP]
                src_ip = ip.src
                dst_ip = ip.dst
            
            if TCP in pkt:
                tcp = pkt[TCP]
                src_port = tcp.sport
                dst_port = tcp.dport
                protocol = "tcp"
            elif UDP in pkt:
                udp = pkt[UDP]
                src_port = udp.sport
                dst_port = udp.dport
                protocol = "udp"
            elif ICMP in pkt:
                protocol = "icmp"
            
            if src_ip and dst_ip:
                return Packet(
                    timestamp=timestamp,
                    src_mac=src_mac,
                    dst_mac=dst_mac,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port or 0,
                    dst_port=dst_port or 0,
                    protocol=protocol or "unknown",
                    packet_data=bytes(pkt),
                    packet_len=len(pkt)
                )
        except Exception as e:
            logger.debug(f"Packet parsing error: {e}")
        
        return None
    
    def close(self):
        """Cleanup capture (nothing needed for scapy)"""
        logger.info(f"Live capture closed: {self.packets_captured} packets captured")


class InMemorySource(PacketSource):
    """In-memory packet source for testing"""
    
    def __init__(self, packets: List[Packet]):
        """
        Initialize with pre-built packets
        
        Args:
            packets: List of Packet objects
        """
        self.packets = packets
        logger.info(f"InMemorySource initialized with {len(packets)} packets")
    
    def read_packets(self) -> Generator[Packet, None, None]:
        """Yield packets from memory"""
        for packet in self.packets:
            yield packet
    
    def close(self):
        """Cleanup"""
        pass


class PacketSourceFactory:
    """Factory for creating appropriate packet sources"""
    
    @staticmethod
    def create(source_type: str, **kwargs) -> PacketSource:
        """
        Create packet source by type
        
        Args:
            source_type: 'pcap', 'live', 'memory'
            **kwargs: Type-specific arguments
        
        Returns:
            PacketSource instance
        """
        if source_type == "pcap":
            return PCAPReader(filepath=kwargs.get("filepath"))
        elif source_type == "live":
            return LiveCapture(
                interface=kwargs.get("interface"),
                packet_count=kwargs.get("packet_count", 0),
                filter_expr=kwargs.get("filter_expr"),
                timeout=kwargs.get("timeout")
            )
        elif source_type == "memory":
            return InMemorySource(packets=kwargs.get("packets", []))
        else:
            raise ValueError(f"Unknown source type: {source_type}")
