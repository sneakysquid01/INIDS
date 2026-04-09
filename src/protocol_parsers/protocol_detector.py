"""
Protocol Detector
Routes packets to appropriate protocol parsers based on port, payload patterns
Integrates with Phase A pipeline flow context
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ApplicationProtocol(Enum):
    """Application layer protocols"""
    HTTP = "HTTP"
    HTTPS = "HTTPS"
    DNS = "DNS"
    TLS = "TLS"
    SSH = "SSH"
    FTP = "FTP"
    SMTP = "SMTP"
    POP3 = "POP3"
    IMAP = "IMAP"
    TELNET = "TELNET"
    UNKNOWN = "UNKNOWN"


@dataclass
class ProtocolClassification:
    """Result of protocol detection"""
    protocol: ApplicationProtocol
    confidence: float                  # 0.0 - 1.0 confidence score
    detection_method: str              # "port", "payload_pattern", "tls_hello"
    payload_indicators: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"ProtocolClassification({self.protocol.value}, {self.confidence:.2f})"


@dataclass
class ParsedProtocolData:
    """Container for parsed protocol data"""
    protocol: ApplicationProtocol
    
    # HTTP
    http_request: Optional['HTTPRequest'] = None
    http_response: Optional['HTTPResponse'] = None
    
    # DNS
    dns_query: Optional['DNSQuery'] = None
    dns_response: Optional['DNSResponse'] = None
    
    # TLS
    tls_client_hello: Optional['TLSClientHello'] = None
    tls_server_hello: Optional['TLSServerHello'] = None
    
    # Raw payload (if parsing failed)
    raw_payload: bytes = b""
    
    def __repr__(self):
        summary = f"ParsedProtocolData({self.protocol.value})"
        if self.http_request:
            summary += f" [HTTP {self.http_request.method}]"
        elif self.dns_query:
            summary += f" [DNS {self.dns_query.domain}]"
        elif self.tls_client_hello:
            summary += f" [TLS ClientHello]"
        return summary


class ProtocolDetector:
    """Detect and parse application layer protocols"""
    
    # Port-to-protocol mapping
    WELL_KNOWN_PORTS = {
        80: ApplicationProtocol.HTTP,
        8080: ApplicationProtocol.HTTP,
        8000: ApplicationProtocol.HTTP,
        8888: ApplicationProtocol.HTTP,
        
        443: ApplicationProtocol.HTTPS,
        8443: ApplicationProtocol.HTTPS,
        
        53: ApplicationProtocol.DNS,
        
        22: ApplicationProtocol.SSH,
        
        21: ApplicationProtocol.FTP,
        
        25: ApplicationProtocol.SMTP,
        587: ApplicationProtocol.SMTP,
        
        110: ApplicationProtocol.POP3,
        995: ApplicationProtocol.POP3,
        
        143: ApplicationProtocol.IMAP,
        993: ApplicationProtocol.IMAP,
        
        23: ApplicationProtocol.TELNET,
    }
    
    @staticmethod
    def classify_protocol(src_ip: str, dst_ip: str, src_port: int, dst_port: int,
                         protocol: str, payload: bytes = b"") -> ProtocolClassification:
        """
        Classify application protocol from packet details
        
        Args:
            src_ip: Source IP
            dst_ip: Destination IP
            src_port: Source port
            dst_port: Destination port
            protocol: L4 protocol (TCP, UDP)
            payload: Packet payload
        
        Returns:
            ProtocolClassification with detected protocol and confidence
        """
        
        # Analyze payload FIRST for definitive patterns (TLS, HTTP, DNS)
        # This prevents port-based misclassification
        if payload and len(payload) > 0:
            payload_result = ProtocolDetector._classify_by_payload(payload, protocol, dst_port, src_port)
            if payload_result.confidence >= 0.95:  # High confidence payload match
                return payload_result
        
        # Check well-known ports second (fallback for conn without payload analysis yet)
        if dst_port in ProtocolDetector.WELL_KNOWN_PORTS:
            proto = ProtocolDetector.WELL_KNOWN_PORTS[dst_port]
            return ProtocolClassification(
                protocol=proto,
                confidence=0.85,
                detection_method="well_known_port"
            )
        
        # Check source port (for responses)
        if src_port in ProtocolDetector.WELL_KNOWN_PORTS:
            proto = ProtocolDetector.WELL_KNOWN_PORTS[src_port]
            return ProtocolClassification(
                protocol=proto,
                confidence=0.80,
                detection_method="well_known_port_src"
            )
        
        # Default to unknown
        return ProtocolClassification(
            protocol=ApplicationProtocol.UNKNOWN,
            confidence=0.0,
            detection_method="unknown"
        )
    
    @staticmethod
    def _classify_by_payload(payload: bytes, protocol: str, dst_port: int, 
                            src_port: int) -> ProtocolClassification:
        """Detect protocol from payload patterns"""
        
        if not payload or len(payload) < 4:
            return ProtocolClassification(
                protocol=ApplicationProtocol.UNKNOWN,
                confidence=0.0,
                detection_method="payload_too_short"
            )
        
        # Get first bytes as string (with error handling)
        try:
            text_payload = payload[:100].decode('utf-8', errors='ignore').upper()
        except:
            text_payload = ""
        
        # HTTP detection
        if any(text_payload.startswith(method) for method in 
               ['GET ', 'POST ', 'PUT ', 'DELETE ', 'HEAD ', 'OPTIONS ', 'PATCH ', 'TRACE ', 'CONNECT ']):
            return ProtocolClassification(
                protocol=ApplicationProtocol.HTTP,
                confidence=0.99,
                detection_method="http_method_pattern",
                payload_indicators=["http_request_line"]
            )
        
        if 'HTTP/' in text_payload:
            return ProtocolClassification(
                protocol=ApplicationProtocol.HTTP,
                confidence=0.95,
                detection_method="http_version_pattern",
                payload_indicators=["http_version"]
            )
        
        # HTTPS/TLS detection (0x16 = Handshake record type)
        if payload[0] == 0x16 and len(payload) > 5:
            if payload[1:3] in [b'\x03\x01', b'\x03\x02', b'\x03\x03', b'\x03\x04']:
                return ProtocolClassification(
                    protocol=ApplicationProtocol.TLS,
                    confidence=0.99,
                    detection_method="tls_record_type",
                    payload_indicators=["tls_handshake_record"]
                )
        
        # DNS detection (port 53, UDP)
        if (dst_port == 53 or src_port == 53) and protocol.upper() == "UDP":
            # DNS queries are usually short and start with transaction ID
            if len(payload) >= 12:
                return ProtocolClassification(
                    protocol=ApplicationProtocol.DNS,
                    confidence=0.90,
                    detection_method="dns_port_protocol",
                    payload_indicators=["dns_on_port_53"]
                )
        
        # SSH detection (SSH-2.0 or SSH-1.99)
        if text_payload.startswith('SSH-'):
            return ProtocolClassification(
                protocol=ApplicationProtocol.SSH,
                confidence=0.99,
                detection_method="ssh_banner",
                payload_indicators=["ssh_version_string"]
            )
        
        # FTP detection
        if text_payload.startswith('220 ') or 'FTP' in text_payload:
            return ProtocolClassification(
                protocol=ApplicationProtocol.FTP,
                confidence=0.90,
                detection_method="ftp_banner",
                payload_indicators=["ftp_response_code"]
            )
        
        # SMTP detection
        if text_payload.startswith('220 ') and ('SMTP' in text_payload or 'MAIL' in text_payload):
            return ProtocolClassification(
                protocol=ApplicationProtocol.SMTP,
                confidence=0.95,
                detection_method="smtp_banner",
                payload_indicators=["smtp_response"]
            )
        
        # Telnet detection (hard to distinguish, use port as hint)
        if dst_port == 23 or src_port == 23:
            return ProtocolClassification(
                protocol=ApplicationProtocol.TELNET,
                confidence=0.75,
                detection_method="telnet_port",
                payload_indicators=["telnet_port_23"]
            )
        
        # Default
        return ProtocolClassification(
            protocol=ApplicationProtocol.UNKNOWN,
            confidence=0.0,
            detection_method="no_pattern_match"
        )
    
    @staticmethod
    def parse_protocol_payload(classification: ProtocolClassification, 
                               payload: bytes, src_port: int = 0, dst_port: int = 0) -> ParsedProtocolData:
        """
        Parse protocol-specific data from payload
        
        Args:
            classification: Protocol classification result
            payload: Packet payload
            src_port: Source port
            dst_port: Destination port
        
        Returns:
            ParsedProtocolData with parsed protocol-specific structures
        """
        from . import http_parser, dns_parser, tls_parser
        
        parsed = ParsedProtocolData(protocol=classification.protocol, raw_payload=payload)
        
        try:
            if classification.protocol == ApplicationProtocol.HTTP:
                # Try to parse as HTTP request
                http_req = http_parser.HTTPParser.parse_request(payload)
                if http_req:
                    parsed.http_request = http_req
                else:
                    # Try to parse as HTTP response
                    http_resp = http_parser.HTTPParser.parse_response(payload)
                    if http_resp:
                        parsed.http_response = http_resp
            
            elif classification.protocol == ApplicationProtocol.HTTPS or \
                 classification.protocol == ApplicationProtocol.TLS:
                # Try to parse as TLS ClientHello
                tls_ch = tls_parser.TLSParser.parse_client_hello(payload)
                if tls_ch:
                    parsed.tls_client_hello = tls_ch
                else:
                    # Try to parse as TLS ServerHello
                    tls_sh = tls_parser.TLSParser.parse_server_hello(payload)
                    if tls_sh:
                        parsed.tls_server_hello = tls_sh
            
            elif classification.protocol == ApplicationProtocol.DNS:
                # Try to parse as DNS query
                dns_q = dns_parser.DNSParser.parse_dns_query(payload, src_port, dst_port)
                if dns_q:
                    parsed.dns_query = dns_q
                else:
                    # Try to parse as DNS response
                    dns_r = dns_parser.DNSParser.parse_dns_response(payload)
                    if dns_r:
                        parsed.dns_response = dns_r
        
        except Exception as e:
            logger.debug(f"Protocol parse error: {e}")
        
        return parsed
    
    @staticmethod
    def extract_ml_features(parsed_data: ParsedProtocolData) -> Dict[str, any]:
        """
        Extract combined ML features from parsed protocol data
        
        Args:
            parsed_data: Parsed protocol structures
        
        Returns:
            Dict of ML features
        """
        from . import http_parser, dns_parser, tls_parser
        
        features = {
            "protocol": parsed_data.protocol.value,
            "payload_size": len(parsed_data.raw_payload),
        }
        
        # Add protocol-specific features
        if parsed_data.http_request:
            features.update(http_parser.HTTPParser.extract_features(parsed_data.http_request))
        
        elif parsed_data.http_response:
            features.update(http_parser.HTTPParser.extract_features_response(parsed_data.http_response))
        
        elif parsed_data.dns_query:
            features.update(dns_parser.DNSParser.extract_features(parsed_data.dns_query))
        
        elif parsed_data.dns_response:
            features.update(dns_parser.DNSParser.extract_features_response(parsed_data.dns_response))
        
        elif parsed_data.tls_client_hello:
            features.update(tls_parser.TLSParser.extract_features(parsed_data.tls_client_hello))
        
        elif parsed_data.tls_server_hello:
            features.update(tls_parser.TLSParser.extract_features_server(parsed_data.tls_server_hello))
        
        return features
    
    @staticmethod
    def is_protocol_suspicious(parsed_data: ParsedProtocolData) -> bool:
        """Check if parsed protocol data shows suspicious indicators"""
        
        if parsed_data.http_request and parsed_data.http_request.is_suspicious:
            return True
        
        if parsed_data.http_response and parsed_data.http_response.is_error:
            return True
        
        if parsed_data.dns_query and parsed_data.dns_query.is_suspicious:
            return True
        
        if parsed_data.dns_response and parsed_data.dns_response.is_suspicious:
            return True
        
        if parsed_data.tls_client_hello and parsed_data.tls_client_hello.is_suspicious:
            return True
        
        if parsed_data.tls_server_hello and parsed_data.tls_server_hello.is_suspicious:
            return True
        
        return False
    
    @staticmethod
    def get_suspicious_indicators(parsed_data: ParsedProtocolData) -> List[str]:
        """Get list of all suspicious indicators"""
        
        indicators = []
        
        if parsed_data.http_request:
            indicators.extend(parsed_data.http_request.suspicious_indicators)
        
        if parsed_data.http_response:
            if parsed_data.http_response.is_error:
                indicators.append(f"http_error_{parsed_data.http_response.status_code}")
        
        if parsed_data.dns_query:
            indicators.extend(parsed_data.dns_query.suspicious_indicators)
        
        if parsed_data.dns_response:
            indicators.extend(parsed_data.dns_response.suspicious_indicators)
        
        if parsed_data.tls_client_hello:
            indicators.extend(parsed_data.tls_client_hello.suspicious_indicators)
        
        if parsed_data.tls_server_hello:
            indicators.extend(parsed_data.tls_server_hello.suspicious_indicators)
        
        return indicators
