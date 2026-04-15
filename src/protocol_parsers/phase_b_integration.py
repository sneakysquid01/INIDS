"""
Phase B Integration
Integrate protocol parsers with Phase A pipeline
Updates flow context with parsed protocol data and ML features
"""

from typing import Optional, Dict, Any
from dataclasses import field
import logging

logger = logging.getLogger(__name__)


class ProtocolAnalysisContext:
    """
    Per-flow protocol analysis context
    Stored in FlowContext.features_cache['protocol_analysis']
    """
    
    def __init__(self):
        self.detected_protocol = None           # ApplicationProtocol enum
        self.classification_confidence = 0.0
        
        # Parsed protocol objects
        self.http_request = None
        self.http_response = None
        self.dns_query = None
        self.dns_response = None
        self.tls_client_hello = None
        self.tls_server_hello = None
        
        # Extracted features
        self.ml_features = {}
        
        # Suspicious indicators
        self.is_suspicious = False
        self.suspicious_indicators = []
        
        # Statistics
        self.packets_analyzed = 0
        self.detections_made = 0
    
    def __repr__(self):
        return (f"ProtocolAnalysisContext("
                f"protocol={self.detected_protocol}, "
                f"confidence={self.classification_confidence:.2f}, "
                f"suspicious={self.is_suspicious})")


class ProtocolAnalyzer:
    """
    Analyze packets at protocol layer
    Integrates with Phase A pipeline
    """
    
    @staticmethod
    def analyze_packet_protocol(flow_context, decoded_packet) -> Optional[ProtocolAnalysisContext]:
        """
        Analyze packet protocol layer
        
        Args:
            flow_context: FlowContext from Phase A
            decoded_packet: DecodedPacket from Phase A
        
        Returns:
            ProtocolAnalysisContext or None
        """
        from . import (
            ProtocolDetector,
            ApplicationProtocol
        )
        
        if not decoded_packet or not decoded_packet.l4_info:
            return None
        
        # Get or create protocol analysis context
        proto_ctx = flow_context.features_cache.get('protocol_analysis')
        if not proto_ctx:
            proto_ctx = ProtocolAnalysisContext()
            flow_context.features_cache['protocol_analysis'] = proto_ctx
        
        proto_ctx.packets_analyzed += 1
        
        # Extract packet info
        src_ip = decoded_packet.l3_info.src_ip if decoded_packet.l3_info else ""
        dst_ip = decoded_packet.l3_info.dst_ip if decoded_packet.l3_info else ""
        src_port = decoded_packet.l4_info.src_port if decoded_packet.l4_info else 0
        dst_port = decoded_packet.l4_info.dst_port if decoded_packet.l4_info else 0
        l4_protocol = decoded_packet.l4_info.protocol if decoded_packet.l4_info else "TCP"
        payload = decoded_packet.payload_data or b""
        
        # Classify protocol
        classification = ProtocolDetector.classify_protocol(
            src_ip, dst_ip, src_port, dst_port, l4_protocol, payload
        )
        
        # Update context if we have new/better classification
        if classification.confidence > proto_ctx.classification_confidence:
            proto_ctx.detected_protocol = classification.protocol
            proto_ctx.classification_confidence = classification.confidence
            
            # Parse protocol-specific data
            parsed_data = ProtocolDetector.parse_protocol_payload(
                classification, payload, src_port, dst_port
            )
            
            # Store parsed objects
            if parsed_data.http_request:
                proto_ctx.http_request = parsed_data.http_request
            if parsed_data.http_response:
                proto_ctx.http_response = parsed_data.http_response
            if parsed_data.dns_query:
                proto_ctx.dns_query = parsed_data.dns_query
            if parsed_data.dns_response:
                proto_ctx.dns_response = parsed_data.dns_response
            if parsed_data.tls_client_hello:
                proto_ctx.tls_client_hello = parsed_data.tls_client_hello
            if parsed_data.tls_server_hello:
                proto_ctx.tls_server_hello = parsed_data.tls_server_hello
            
            # Extract ML features
            proto_ctx.ml_features = ProtocolDetector.extract_ml_features(parsed_data)
            
            # Check for suspicious patterns
            proto_ctx.is_suspicious = ProtocolDetector.is_protocol_suspicious(parsed_data)
            proto_ctx.suspicious_indicators = ProtocolDetector.get_suspicious_indicators(parsed_data)
            
            proto_ctx.detections_made += 1
            
            logger.debug(f"Protocol analysis: {proto_ctx}")
        
        return proto_ctx
    
    @staticmethod
    def get_protocol_features(flow_context) -> Dict[str, Any]:
        """
        Extract protocol-layer features for ML detection
        
        Args:
            flow_context: FlowContext with protocol analysis
        
        Returns:
            Dict of protocol features
        """
        features = {}
        
        proto_ctx = flow_context.features_cache.get('protocol_analysis')
        if proto_ctx:
            features.update(proto_ctx.ml_features)
            features['protocol_is_suspicious'] = proto_ctx.is_suspicious
            features['protocol_suspicious_count'] = len(proto_ctx.suspicious_indicators)
            features['protocol_detected'] = proto_ctx.detected_protocol.value if proto_ctx.detected_protocol else "UNKNOWN"
            features['protocol_confidence'] = proto_ctx.classification_confidence
        
        return features
    
    @staticmethod
    def augment_flow_context(flow_context) -> None:
        """
        Augment flow context with protocol analysis
        Modifies features_cache and IPS action if needed
        
        Args:
            flow_context: FlowContext to augment
        """
        from . import ApplicationProtocol
        
        proto_ctx = flow_context.features_cache.get('protocol_analysis')
        if not proto_ctx:
            return
        
        # Add protocol features to flow context
        if 'protocol_analysis' not in flow_context.features_cache:
            flow_context.features_cache['protocol_analysis'] = {}
        
        proto_data = flow_context.features_cache['protocol_analysis']
        proto_data.update(ProtocolAnalyzer.get_protocol_features(flow_context))
        
        # If protocol analysis found suspicious patterns, escalate flow action
        if proto_ctx.is_suspicious and len(proto_ctx.suspicious_indicators) > 0:
            # Escalate to ALERT if not already blocking
            current_action = flow_context.action
            if current_action != flow_context.FlowAction.BLOCK and current_action != flow_context.FlowAction.RATE_LIMIT:
                flow_context.action = flow_context.FlowAction.ALERT
                logger.warning(f"Flow escalated to ALERT due to protocol analysis: {proto_ctx.suspicious_indicators}")


class PhaseABIntegrationAdapter:
    """
    Adapter to integrate Phase B protocol parsers with Phase A pipeline
    This enables protocol parsing within the existing packet processing loop
    """
    
    @staticmethod
    def create_protocol_detection_callback():
        """
        Create a detection callback for Phase A pipeline
        Enables protocol analysis as optional detection layer
        """
        def protocol_detection_callback(flow_context, decoded_packet):
            """
            Callback to run protocol analysis on each packet
            
            Args:
                flow_context: FlowContext from Phase A
                decoded_packet: DecodedPacket from Phase A
            
            Returns:
                Tuple of (detection_score, alert_reason) or (0.0, None)
            """
            try:
                proto_ctx = ProtocolAnalyzer.analyze_packet_protocol(flow_context, decoded_packet)
                
                if proto_ctx and proto_ctx.is_suspicious:
                    # Return score based on suspicious indicators
                    score = min(0.99, len(proto_ctx.suspicious_indicators) * 0.15)
                    reason = f"Protocol anomaly: {', '.join(proto_ctx.suspicious_indicators[:3])}"
                    return score, reason
                
                return 0.0, None
            
            except Exception as e:
                logger.debug(f"Protocol detection callback error: {e}")
                return 0.0, None
        
        return protocol_detection_callback
    
    @staticmethod
    def integrate_with_pipeline(pipeline):
        """
        Integrate protocol detection with Phase A pipeline
        
        Args:
            pipeline: PacketProcessingPipeline from Phase A
        """
        # Register protocol detection callback
        protocol_callback = PhaseABIntegrationAdapter.create_protocol_detection_callback()
        
        # Store in pipeline for use during packet processing
        if not hasattr(pipeline, 'protocol_detection_callback'):
            pipeline.protocol_detection_callback = protocol_callback
            logger.info("Protocol detection integrated with Phase A pipeline")
    
    @staticmethod
    def enhance_packet_processing(flow_context, decoded_packet, pipeline=None):
        """
        Enhanced packet processing with protocol analysis
        Call this from Phase A _process_packet method
        
        Args:
            flow_context: FlowContext
            decoded_packet: DecodedPacket
            pipeline: Optional PacketProcessingPipeline reference
        
        Returns:
            Optional (detection_score, alert_reason)
        """
        # Analyze protocol
        proto_ctx = ProtocolAnalyzer.analyze_packet_protocol(flow_context, decoded_packet)
        
        # Augment flow context with protocol features
        ProtocolAnalyzer.augment_flow_context(flow_context)
        
        # Run protocol detection
        if hasattr(pipeline or {}, 'protocol_detection_callback'):
            callback = pipeline.protocol_detection_callback
            return callback(flow_context, decoded_packet)
        
        return 0.0, None


# Convenience functions for integration

def analyze_flow_protocol(flow_context) -> Optional[ProtocolAnalysisContext]:
    """Get protocol analysis for a flow"""
    return flow_context.features_cache.get('protocol_analysis')


def get_flow_protocol(flow_context) -> str:
    """Get detected protocol name for a flow"""
    proto_ctx = flow_context.features_cache.get('protocol_analysis')
    return proto_ctx.detected_protocol.value if proto_ctx and proto_ctx.detected_protocol else "UNKNOWN"


def is_flow_protocol_suspicious(flow_context) -> bool:
    """Check if protocol analysis found suspicious indicators"""
    proto_ctx = flow_context.features_cache.get('protocol_analysis')
    return proto_ctx.is_suspicious if proto_ctx else False


def get_protocol_suspicious_indicators(flow_context) -> list:
    """Get protocol suspicious indicators for a flow"""
    proto_ctx = flow_context.features_cache.get('protocol_analysis')
    return proto_ctx.suspicious_indicators if proto_ctx else []
