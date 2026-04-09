"""
INIDS EVE JSON Output Module

Implements Suricata-compatible EVE JSON format for structured alert output.
Supports:
- Alert events (detections)
- HTTP events (protocol payloads)
- DNS events (protocol payloads)
- TLS/SSL events (protocol payloads)
- Flow events (flow start/end)

EVE JSON format enables integration with:
- ELK Stack (Elasticsearch + Kibana + Logstash)
- Splunk
- ArcSight
- Graylog
- Custom SOC workflows
"""

import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import uuid4


class EventType(Enum):
    """EVE event types (Suricata-compatible)"""
    ALERT = "alert"
    HTTP = "http"
    DNS = "dns"
    TLS = "tls"
    SSH = "ssh"
    FLOW = "flow"
    FILEINFO = "fileinfo"
    STATS = "stats"


class AlertSeverity(Enum):
    """Alert severity levels (1-highest to 3-lowest)"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


@dataclass
class FlowTuple:
    """5-tuple network flow identifier"""
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: str  # tcp, udp, icmp
    vlan_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = {
            "src_ip": self.src_ip,
            "src_port": self.src_port,
            "dest_ip": self.dst_ip,
            "dest_port": self.dst_port,
            "proto": self.proto
        }
        if self.vlan_id is not None:
            d["vlan_id"] = self.vlan_id
        return d


@dataclass
class AlertPayload:
    """Alert event payload (detection)"""
    action: str  # allow, drop, reject, alert
    gid: int = 0  # Generator ID (0 = default)
    signature_id: int = 0  # Signature/rule ID
    signature: str = ""  # Signature/rule name
    category: str = ""  # Alert category (e.g., "Attempted Denial of Service")
    severity: int = 3  # 1-5, lower = more severe
    source: Optional[str] = None  # Alert source (INIDS, Suricata, etc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class HTTPPayload:
    """HTTP protocol payload"""
    http_method: Optional[str] = None
    http_uri: Optional[str] = None
    http_version: Optional[str] = None
    http_host: Optional[str] = None
    http_user_agent: Optional[str] = None
    http_content_type: Optional[str] = None
    http_response_code: Optional[int] = None
    http_response_body: Optional[str] = None
    http_request_body_printable: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering out None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DNSPayload:
    """DNS protocol payload"""
    dns_type: Optional[str] = None  # query, answer
    dns_id: Optional[int] = None
    dns_records_count: Optional[int] = None
    dns_queries: Optional[List[Dict[str, Any]]] = None
    dns_answers: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering out None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TLSPayload:
    """TLS/SSL protocol payload"""
    tls_version: Optional[str] = None
    tls_cipher: Optional[str] = None
    tls_ja3: Optional[str] = None
    tls_ja3s: Optional[str] = None
    tls_subject: Optional[str] = None
    tls_issuer: Optional[str] = None
    tls_fingerprint: Optional[str] = None
    tls_sni: Optional[str] = None
    tls_certificate_serial: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering out None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class EVEEvent:
    """
    EVE JSON Event (Suricata-compatible)
    
    Complete structured event for output to various backends.
    All fields optional except timestamp, flow, and event_type.
    """
    # Mandatory fields
    timestamp: str  # ISO 8601: 2025-04-09T14:32:15.123456+00:00
    event_type: EventType
    flow_id: int
    in_iface: str = "unknown"
    out_iface: str = "unknown"
    
    # Flow info
    src_ip: str = ""
    src_port: int = 0
    dest_ip: str = ""
    dest_port: int = 0
    proto: str = ""
    vlan_id: Optional[int] = None
    
    # Packet info
    pcap_cnt: int = 0  # Packet count in flow
    packet_info: Optional[Dict[str, Any]] = None
    
    # Alert info (for alert events)
    alert: Optional[AlertPayload] = None
    
    # Protocol-specific payloads
    http: Optional[HTTPPayload] = None
    dns: Optional[DNSPayload] = None
    tls: Optional[TLSPayload] = None
    ssh: Optional[Dict[str, Any]] = None
    
    # Metadata
    community_id: Optional[str] = None  # Community ID fingerprint
    app_layer: Optional[Dict[str, Any]] = None
    
    # Flow metadata
    flow: Optional[Dict[str, Any]] = None  # Flow state/stats if flow event
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    # Auto-generated fields
    event_id: str = field(default_factory=lambda: str(uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization"""
        result = {
            "timestamp": self.timestamp,
            "flow_id": self.flow_id,
            "event_type": self.event_type.value,
            "in_iface": self.in_iface,
            "out_iface": self.out_iface,
            "event_id": self.event_id,
        }
        
        # Flow tuple
        if any([self.src_ip, self.src_port, self.dest_ip, self.dest_port, self.proto]):
            flow_tuple = {
                "src_ip": self.src_ip,
                "src_port": self.src_port,
                "dest_ip": self.dest_ip,
                "dest_port": self.dest_port,
                "proto": self.proto,
            }
            if self.vlan_id is not None:
                flow_tuple["vlan_id"] = self.vlan_id
            result["flow"] = flow_tuple
        
        # Packet info
        if self.pcap_cnt > 0:
            result["pcap_cnt"] = self.pcap_cnt
        if self.packet_info:
            result["packet_info"] = self.packet_info
        
        # Alert
        if self.alert:
            result["alert"] = self.alert.to_dict()
        
        # Protocol payloads
        if self.http:
            result["http"] = self.http.to_dict()
        if self.dns:
            result["dns"] = self.dns.to_dict()
        if self.tls:
            result["tls"] = self.tls.to_dict()
        if self.ssh:
            result["ssh"] = self.ssh
        
        # Metadata
        if self.community_id:
            result["community_id"] = self.community_id
        if self.app_layer:
            result["app_layer"] = self.app_layer
        if self.flow:
            result["flow"] = self.flow
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class EVEEventBuilder:
    """
    Builder for constructing EVE events from detection results.
    
    Converts INIDS detections (flow context + detection score) into
    structured EVE JSON events for output.
    """
    
    def __init__(self, source: str = "INIDS", facility_id: int = 101):
        self.source = source
        self.facility_id = facility_id
        self._event_counter = 0
    
    def create_alert_event(
        self,
        flow_id: int,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        proto: str,
        detection_reason: str,
        detection_score: float,
        payload: Optional[Dict[str, Any]] = None,
        vlan_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> EVEEvent:
        """
        Create an alert event from detection result.
        
        Args:
            flow_id: Flow identifier
            src_ip: Source IP address
            src_port: Source port
            dst_ip: Destination IP address
            dst_port: Destination port
            proto: Protocol (tcp, udp, icmp)
            detection_reason: Human-readable reason for detection
            detection_score: Detection confidence (0.0-1.0)
            payload: Optional protocol-specific payload
            vlan_id: Optional VLAN ID
            timestamp: ISO 8601 timestamp (auto-generated if not provided)
        
        Returns:
            EVEEvent: Alert event ready for output
        """
        if timestamp is None:
            timestamp = self._get_iso_timestamp()
        
        # Map detection score to severity (lower score = higher severity)
        severity = self._score_to_severity(detection_score)
        
        # Create alert payload
        alert = AlertPayload(
            action="alert",
            signature_id=self.facility_id + self._event_counter,
            signature=detection_reason,
            category="Anomaly Detection",
            severity=severity,
            source=self.source
        )
        self._event_counter += 1
        
        # Create event
        event = EVEEvent(
            timestamp=timestamp,
            event_type=EventType.ALERT,
            flow_id=flow_id,
            src_ip=src_ip,
            src_port=src_port,
            dest_ip=dst_ip,
            dest_port=dst_port,
            proto=proto,
            vlan_id=vlan_id,
            alert=alert,
            metadata={
                "detection_score": detection_score,
                "detection_confidence": f"{detection_score*100:.1f}%"
            }
        )
        
        # Add protocol-specific payload if provided
        if payload:
            if "http" in payload:
                event.http = self._dict_to_http_payload(payload["http"])
            if "dns" in payload:
                event.dns = self._dict_to_dns_payload(payload["dns"])
            if "tls" in payload:
                event.tls = self._dict_to_tls_payload(payload["tls"])
        
        return event
    
    def create_flow_event(
        self,
        flow_id: int,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        proto: str,
        flow_state: Dict[str, Any],
        vlan_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> EVEEvent:
        """
        Create a flow event (flow start/end).
        
        Args:
            flow_id: Flow identifier
            src_ip: Source IP
            src_port: Source port
            dst_ip: Destination IP
            dst_port: Destination port
            proto: Protocol
            flow_state: Flow statistics (packets, bytes, duration, etc)
            vlan_id: Optional VLAN ID
            timestamp: ISO 8601 timestamp (auto-generated if not provided)
        
        Returns:
            EVEEvent: Flow event
        """
        if timestamp is None:
            timestamp = self._get_iso_timestamp()
        
        event = EVEEvent(
            timestamp=timestamp,
            event_type=EventType.FLOW,
            flow_id=flow_id,
            src_ip=src_ip,
            src_port=src_port,
            dest_ip=dst_ip,
            dest_port=dst_port,
            proto=proto,
            vlan_id=vlan_id,
            flow=flow_state
        )
        
        return event
    
    def create_http_event(
        self,
        flow_id: int,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        proto: str,
        http_data: Dict[str, Any],
        vlan_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> EVEEvent:
        """Create HTTP protocol event"""
        if timestamp is None:
            timestamp = self._get_iso_timestamp()
        
        event = EVEEvent(
            timestamp=timestamp,
            event_type=EventType.HTTP,
            flow_id=flow_id,
            src_ip=src_ip,
            src_port=src_port,
            dest_ip=dst_ip,
            dest_port=dst_port,
            proto=proto,
            vlan_id=vlan_id,
            http=self._dict_to_http_payload(http_data)
        )
        
        return event
    
    def create_dns_event(
        self,
        flow_id: int,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        proto: str,
        dns_data: Dict[str, Any],
        vlan_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> EVEEvent:
        """Create DNS protocol event"""
        if timestamp is None:
            timestamp = self._get_iso_timestamp()
        
        event = EVEEvent(
            timestamp=timestamp,
            event_type=EventType.DNS,
            flow_id=flow_id,
            src_ip=src_ip,
            src_port=src_port,
            dest_ip=dst_ip,
            dest_port=dst_port,
            proto=proto,
            vlan_id=vlan_id,
            dns=self._dict_to_dns_payload(dns_data)
        )
        
        return event
    
    def create_tls_event(
        self,
        flow_id: int,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        proto: str,
        tls_data: Dict[str, Any],
        vlan_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> EVEEvent:
        """Create TLS/SSL protocol event"""
        if timestamp is None:
            timestamp = self._get_iso_timestamp()
        
        event = EVEEvent(
            timestamp=timestamp,
            event_type=EventType.TLS,
            flow_id=flow_id,
            src_ip=src_ip,
            src_port=src_port,
            dest_ip=dst_ip,
            dest_port=dst_port,
            proto=proto,
            vlan_id=vlan_id,
            tls=self._dict_to_tls_payload(tls_data)
        )
        
        return event
    
    @staticmethod
    def _get_iso_timestamp() -> str:
        """Get current ISO 8601 timestamp"""
        return datetime.now(timezone.utc).isoformat(timespec='microseconds')
    
    @staticmethod
    def _score_to_severity(score: float) -> int:
        """
        Convert detection score (0.0-1.0) to EVE severity (1-5).
        
        - 0.9-1.0 → 1 (Critical)
        - 0.8-0.9 → 2 (High)
        - 0.6-0.8 → 3 (Medium)
        - 0.4-0.6 → 4 (Low)
        - 0.0-0.4 → 5 (Info)
        """
        if score >= 0.9:
            return 1
        elif score >= 0.8:
            return 2
        elif score >= 0.6:
            return 3
        elif score >= 0.4:
            return 4
        else:
            return 5
    
    @staticmethod
    def _dict_to_http_payload(data: Dict[str, Any]) -> HTTPPayload:
        """Convert dictionary to HTTPPayload"""
        return HTTPPayload(
            http_method=data.get("http_method"),
            http_uri=data.get("http_uri"),
            http_version=data.get("http_version"),
            http_host=data.get("http_host"),
            http_user_agent=data.get("http_user_agent"),
            http_content_type=data.get("http_content_type"),
            http_response_code=data.get("http_response_code"),
            http_response_body=data.get("http_response_body"),
            http_request_body_printable=data.get("http_request_body_printable"),
        )
    
    @staticmethod
    def _dict_to_dns_payload(data: Dict[str, Any]) -> DNSPayload:
        """Convert dictionary to DNSPayload"""
        return DNSPayload(
            dns_type=data.get("dns_type"),
            dns_id=data.get("dns_id"),
            dns_records_count=data.get("dns_records_count"),
            dns_queries=data.get("dns_queries"),
            dns_answers=data.get("dns_answers"),
        )
    
    @staticmethod
    def _dict_to_tls_payload(data: Dict[str, Any]) -> TLSPayload:
        """Convert dictionary to TLSPayload"""
        return TLSPayload(
            tls_version=data.get("tls_version"),
            tls_cipher=data.get("tls_cipher"),
            tls_ja3=data.get("tls_ja3"),
            tls_ja3s=data.get("tls_ja3s"),
            tls_subject=data.get("tls_subject"),
            tls_issuer=data.get("tls_issuer"),
            tls_fingerprint=data.get("tls_fingerprint"),
            tls_sni=data.get("tls_sni"),
            tls_certificate_serial=data.get("tls_certificate_serial"),
        )


# Example usage and schema documentation
EXAMPLE_ALERT_EVENT = {
    "timestamp": "2025-04-09T14:32:15.123456+00:00",
    "flow_id": 42,
    "event_type": "alert",
    "event_id": "550e8400-e29b-41d4-a716-446655440000",
    "in_iface": "eth0",
    "out_iface": "unknown",
    "flow": {
        "src_ip": "192.168.1.100",
        "src_port": 54321,
        "dest_ip": "8.8.8.8",
        "dest_port": 443,
        "proto": "tcp"
    },
    "alert": {
        "action": "alert",
        "gid": 0,
        "signature_id": 101,
        "signature": "Potential SQL injection attempt",
        "category": "Anomaly Detection",
        "severity": 2,
        "source": "INIDS"
    },
    "metadata": {
        "detection_score": 0.92,
        "detection_confidence": "92.0%"
    }
}

EXAMPLE_DNS_EVENT = {
    "timestamp": "2025-04-09T14:32:15.123456+00:00",
    "flow_id": 43,
    "event_type": "dns",
    "event_id": "550e8400-e29b-41d4-a716-446655440001",
    "in_iface": "eth0",
    "out_iface": "unknown",
    "flow": {
        "src_ip": "192.168.1.100",
        "src_port": 53218,
        "dest_ip": "8.8.8.8",
        "dest_port": 53,
        "proto": "udp"
    },
    "dns": {
        "dns_type": "query",
        "dns_queries": [
            {
                "rrname": "evil.example.com",
                "rrtype": "A"
            }
        ]
    }
}

EXAMPLE_HTTP_EVENT = {
    "timestamp": "2025-04-09T14:32:15.123456+00:00",
    "flow_id": 44,
    "event_type": "http",
    "event_id": "550e8400-e29b-41d4-a716-446655440002",
    "in_iface": "eth0",
    "out_iface": "unknown",
    "flow": {
        "src_ip": "192.168.1.100",
        "src_port": 54322,
        "dest_ip": "93.184.216.34",
        "dest_port": 80,
        "proto": "tcp"
    },
    "http": {
        "http_method": "GET",
        "http_uri": "/index.php?id=1' OR '1'='1",
        "http_host": "example.com",
        "http_user_agent": "Mozilla/5.0",
        "http_response_code": 200
    }
}
