"""
Multi-Layer Packet Decoder
Parses L2 (Ethernet), L3 (IP), and L4 (TCP/UDP/ICMP) headers
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import struct
import logging

logger = logging.getLogger(__name__)


class Protocol(Enum):
    """L4 Protocol types"""
    TCP = 6
    UDP = 17
    ICMP = 1
    ICMPV6 = 58
    OTHER = 0


@dataclass
class Layer2Info:
    """Ethernet layer information"""
    src_mac: str
    dst_mac: str
    ethertype: int              # 0x0800 (IPv4), 0x86DD (IPv6), 0x8100 (VLAN)
    vlan_id: Optional[int] = None
    vlan_priority: Optional[int] = None


@dataclass
class Layer3Info:
    """IP layer information"""
    version: int                # 4 or 6
    src_ip: str
    dst_ip: str
    protocol: int              # TCP (6), UDP (17), ICMP (1), etc.
    ttl: int
    total_length: int
    identification: int
    flags: str                 # 'DF', 'MF', ''
    fragment_offset: int
    checksum: int


@dataclass
class Layer4Info:
    """TCP/UDP layer information"""
    src_port: int
    dst_port: int
    protocol: str              # 'tcp', 'udp', 'icmp'
    
    # TCP specific
    seq_num: Optional[int] = None
    ack_num: Optional[int] = None
    flags: Optional[str] = None    # 'SYN', 'ACK', 'FIN', 'RST', etc.
    window_size: Optional[int] = None
    
    # UDP specific
    length: Optional[int] = None
    checksum: Optional[int] = None


@dataclass
class DecodedPacket:
    """Complete decoded packet with all layers"""
    raw: bytes
    timestamp: float
    l2: Optional[Layer2Info]
    l3: Optional[Layer3Info]
    l4: Optional[Layer4Info]
    payload: bytes              # L7 data (after headers)
    payload_len: int
    flow_id: str                # 5-tuple hash
    
    def __repr__(self):
        if self.l3 and self.l4:
            return (f"DecodedPacket({self.l3.src_ip}:{self.l4.src_port} → "
                   f"{self.l3.dst_ip}:{self.l4.dst_port} {self.l4.protocol.upper()})")
        return f"DecodedPacket({len(self.raw)} bytes)"


class PacketDecoder:
    """Decode packets into layers"""
    
    @staticmethod
    def decode(raw_bytes: bytes, timestamp: float) -> Optional[DecodedPacket]:
        """
        Decode raw packet bytes into layers
        
        Args:
            raw_bytes: Raw packet data
            timestamp: Packet timestamp
        
        Returns:
            DecodedPacket with parsed layers, or None if decode failed
        """
        try:
            if not raw_bytes:
                logger.debug("Packet is empty")
                return None

            version_nibble = (raw_bytes[0] >> 4) & 0x0F
            l2 = None
            payload = raw_bytes
            offset = 0

            # Prefer Ethernet decoding when available, but accept raw IP packets too.
            if len(raw_bytes) >= 14:
                l2_candidate, payload_candidate, offset_candidate = PacketDecoder._decode_l2(raw_bytes)
                if l2_candidate and l2_candidate.ethertype in {0x0800, 0x86DD, 0x8100}:
                    l2 = l2_candidate
                    payload = payload_candidate
                    offset = offset_candidate
                elif version_nibble not in {4, 6}:
                    return None
            elif version_nibble not in {4, 6}:
                logger.debug("Packet too short for L2 and not a raw IP packet")
                return None
            
            # Decode L3: IP
            l3 = None
            if (l2 and l2.ethertype == 0x0800) or version_nibble == 4:  # IPv4
                l3, payload, offset = PacketDecoder._decode_ipv4(payload)
            elif (l2 and l2.ethertype == 0x86DD) or version_nibble == 6:  # IPv6
                l3, payload, offset = PacketDecoder._decode_ipv6(payload)
            
            # Decode L4: TCP/UDP
            l4 = None
            if l3:
                if l3.protocol == Protocol.TCP.value:
                    l4, payload, offset = PacketDecoder._decode_tcp(payload)
                elif l3.protocol == Protocol.UDP.value:
                    l4, payload, offset = PacketDecoder._decode_udp(payload)
                elif l3.protocol == Protocol.ICMP.value:
                    l4 = PacketDecoder._decode_icmp()
            
            # Compute flow ID
            flow_id = PacketDecoder._compute_flow_id(l3, l4)
            
            return DecodedPacket(
                raw=raw_bytes,
                timestamp=timestamp,
                l2=l2,
                l3=l3,
                l4=l4,
                payload=payload,
                payload_len=len(payload),
                flow_id=flow_id
            )
        
        except Exception as e:
            logger.debug(f"Packet decode error: {e}")
            return None
    
    @staticmethod
    def _decode_l2(data: bytes) -> Tuple[Optional[Layer2Info], bytes, int]:
        """Decode Ethernet layer"""
        if len(data) < 14:
            return None, data, 0
        
        try:
            dst_mac, src_mac, ethertype = struct.unpack("!6s6sH", data[:14])
            
            src_mac_str = ":".join(f"{b:02x}" for b in src_mac)
            dst_mac_str = ":".join(f"{b:02x}" for b in dst_mac)
            
            l2 = Layer2Info(
                src_mac=src_mac_str,
                dst_mac=dst_mac_str,
                ethertype=ethertype
            )
            
            # Handle VLAN tagging
            vlan_id = None
            offset = 14
            if ethertype == 0x8100:  # VLAN tag
                if len(data) >= 18:
                    vlan_tci, ethertype = struct.unpack("!HH", data[14:18])
                    vlan_id = vlan_tci & 0x0FFF
                    l2.vlan_id = vlan_id
                    l2.ethertype = ethertype
                    offset = 18
            
            return l2, data[offset:], offset
        
        except Exception as e:
            logger.debug(f"L2 decode error: {e}")
            return None, data, 0
    
    @staticmethod
    def _decode_ipv4(data: bytes) -> Tuple[Optional[Layer3Info], bytes, int]:
        """Decode IPv4 layer"""
        if len(data) < 20:
            return None, data, 0
        
        try:
            version_ihl, dscp_ecn, total_len, ident, flags_offset, ttl, \
            protocol, checksum, src_ip_int, dst_ip_int = \
                struct.unpack("!BBHHHBBH4s4s", data[:20])
            
            version = (version_ihl >> 4) & 0x0F
            ihl = (version_ihl & 0x0F) * 4
            
            src_ip = ".".join(str(b) for b in src_ip_int)
            dst_ip = ".".join(str(b) for b in dst_ip_int)
            
            flags = ""
            if flags_offset & 0x4000:
                flags += "DF"
            if flags_offset & 0x2000:
                flags += "MF"
            
            fragment_offset = (flags_offset & 0x1FFF) * 8
            
            l3 = Layer3Info(
                version=version,
                src_ip=src_ip,
                dst_ip=dst_ip,
                protocol=protocol,
                ttl=ttl,
                total_length=total_len,
                identification=ident,
                flags=flags,
                fragment_offset=fragment_offset,
                checksum=checksum
            )
            
            # Payload starts after IP header (ihl includes IP options)
            payload = data[ihl:total_len] if total_len <= len(data) else data[ihl:]
            
            return l3, payload, ihl
        
        except Exception as e:
            logger.debug(f"IPv4 decode error: {e}")
            return None, data, 0
    
    @staticmethod
    def _decode_ipv6(data: bytes) -> Tuple[Optional[Layer3Info], bytes, int]:
        """Decode IPv6 layer"""
        if len(data) < 40:
            return None, data, 0
        
        try:
            version_class_label, payload_len, next_header, hop_limit, \
            src_ip_bytes, dst_ip_bytes = \
                struct.unpack("!I HBB 16s16s", data[:40])
            
            version = (version_class_label >> 28) & 0x0F
            
            src_ip = ":".join(f"{int.from_bytes(src_ip_bytes[i:i+2], 'big'):x}" 
                             for i in range(0, 16, 2))
            dst_ip = ":".join(f"{int.from_bytes(dst_ip_bytes[i:i+2], 'big'):x}" 
                             for i in range(0, 16, 2))
            
            l3 = Layer3Info(
                version=version,
                src_ip=src_ip,
                dst_ip=dst_ip,
                protocol=next_header,
                ttl=hop_limit,
                total_length=payload_len + 40,
                identification=0,
                flags="",
                fragment_offset=0,
                checksum=0
            )
            
            return l3, data[40:payload_len+40], 40
        
        except Exception as e:
            logger.debug(f"IPv6 decode error: {e}")
            return None, data, 0
    
    @staticmethod
    def _decode_tcp(data: bytes) -> Tuple[Optional[Layer4Info], bytes, int]:
        """Decode TCP layer"""
        if len(data) < 20:
            return None, data, 0
        
        try:
            src_port, dst_port, seq, ack, offset_flags, window, checksum, urgent = \
                struct.unpack("!HHIIHHHH", data[:20])
            
            offset = (offset_flags >> 12) * 4
            flags = ""
            
            if offset_flags & 0x0001:  # FIN
                flags += "FIN"
            if offset_flags & 0x0002:  # SYN
                flags += "SYN" if not flags else ",SYN"
            if offset_flags & 0x0004:  # RST
                flags += "RST" if not flags else ",RST"
            if offset_flags & 0x0008:  # PSH
                flags += "PSH" if not flags else ",PSH"
            if offset_flags & 0x0010:  # ACK
                flags += "ACK" if not flags else ",ACK"
            if offset_flags & 0x0020:  # URG
                flags += "URG" if not flags else ",URG"
            
            l4 = Layer4Info(
                src_port=src_port,
                dst_port=dst_port,
                protocol="tcp",
                seq_num=seq,
                ack_num=ack,
                flags=flags,
                window_size=window
            )
            
            payload = data[offset:]
            return l4, payload, offset
        
        except Exception as e:
            logger.debug(f"TCP decode error: {e}")
            return None, data, 0
    
    @staticmethod
    def _decode_udp(data: bytes) -> Tuple[Optional[Layer4Info], bytes, int]:
        """Decode UDP layer"""
        if len(data) < 8:
            return None, data, 0
        
        try:
            src_port, dst_port, length, checksum = struct.unpack("!HHHH", data[:8])
            
            l4 = Layer4Info(
                src_port=src_port,
                dst_port=dst_port,
                protocol="udp",
                length=length,
                checksum=checksum
            )
            
            payload = data[8:length] if length > 8 else data[8:]
            return l4, payload, 8
        
        except Exception as e:
            logger.debug(f"UDP decode error: {e}")
            return None, data, 0
    
    @staticmethod
    def _decode_icmp() -> Optional[Layer4Info]:
        """Decode ICMP layer"""
        return Layer4Info(
            src_port=0,
            dst_port=0,
            protocol="icmp"
        )
    
    @staticmethod
    def _compute_flow_id(l3: Optional[Layer3Info], 
                        l4: Optional[Layer4Info]) -> str:
        """Compute 5-tuple flow ID"""
        if not l3 or not l4:
            return "unknown"
        
        import hashlib
        key = f"{l3.src_ip}:{l4.src_port}-{l3.dst_ip}:{l4.dst_port}-{l4.protocol}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
