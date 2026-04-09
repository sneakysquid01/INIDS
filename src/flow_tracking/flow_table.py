"""
Flow Tracking Module
Maintains 5-tuple flows with state, feature caching, and per-flow IPS actions
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class FlowState(Enum):
    """TCP flow state"""
    NEW = "new"
    ESTABLISHED = "established"
    CLOSING = "closing"
    CLOSED = "closed"
    TIMEOUT = "timeout"


class FlowAction(Enum):
    """IPS actions per flow"""
    ALLOW = "allow"
    ALERT = "alert"
    BLOCK = "block"
    RATE_LIMIT = "rate_limit"


@dataclass
class FlowContext:
    """
    Per-flow state container
    Enables stateful detection across multiple packets
    """
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    
    # Packet tracking
    packets_toserver: int = 0
    packets_toclient: int = 0
    bytes_toserver: int = 0
    bytes_toclient: int = 0
    
    # Timing
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())
    last_seen: float = field(default_factory=lambda: datetime.now().timestamp())
    first_packet_time: Optional[float] = None
    last_packet_time: Optional[float] = None
    
    # State
    state: FlowState = FlowState.NEW
    
    # Detection state
    features_cache: Dict = field(default_factory=dict)
    
    # IPS state
    action: FlowAction = FlowAction.ALLOW
    triggered_models: List[str] = field(default_factory=list)
    triggered_rules: List[str] = field(default_factory=list)
    model_votes: Dict = field(default_factory=dict)
    risk_score: float = 0.0
    
    # TCP sequence tracking (for state machine)
    tcp_state: str = "NEW"
    seen_syn: bool = False
    seen_ack: bool = False
    seen_fin: bool = False
    seen_rst: bool = False
    
    # Escalation tracking
    escalation_level: int = 0      # 0=normal, 1=low, 2=medium, 3=high, 4=max
    block_time_remaining: int = 0  # Seconds remaining on block
    
    def __repr__(self):
        return (f"Flow({self.src_ip}:{self.src_port} → "
                f"{self.dst_ip}:{self.dst_port} {self.protocol.upper()} "
                f"state={self.state.value} action={self.action.value})")
    
    def add_packet(self, direction: str, packet_len: int, timestamp: float):
        """Record packet"""
        if not self.first_packet_time:
            self.first_packet_time = timestamp
        self.last_packet_time = timestamp
        self.last_seen = timestamp
        
        if direction == "toserver":
            self.packets_toserver += 1
            self.bytes_toserver += packet_len
        else:  # toclient
            self.packets_toclient += 1
            self.bytes_toclient += packet_len
    
    def get_duration(self) -> float:
        """Duration since flow start (seconds)"""
        return datetime.now().timestamp() - self.start_time
    
    def get_total_packets(self) -> int:
        """Total packets in both directions"""
        return self.packets_toserver + self.packets_toclient
    
    def get_total_bytes(self) -> int:
        """Total bytes in both directions"""
        return self.bytes_toserver + self.bytes_toclient
    
    def is_empty(self) -> bool:
        """Check if flow has any packets"""
        return self.get_total_packets() == 0
    
    def is_established(self) -> bool:
        """Check if flow is TCP established"""
        return self.state == FlowState.ESTABLISHED
    
    def is_blocked(self) -> bool:
        """Check if flow is currently blocked"""
        return self.action == FlowAction.BLOCK


@dataclass
class FlowStats:
    """Flow statistics snapshot"""
    total_flows: int
    active_flows: int
    closed_flows: int
    avg_packets_per_flow: float
    avg_bytes_per_flow: float
    total_bytes_tracked: int
    memory_usage_kb: float


class FlowTable:
    """
    Flow table with 5-tuple hashing
    Enables stateful detection and per-flow actions
    """
    
    def __init__(self, window_seconds: int = 300, max_flows: int = 100000):
        """
        Initialize flow table
        
        Args:
            window_seconds: Flow idle timeout (seconds)
            max_flows: Maximum flows to track (LRU eviction after this)
        """
        self.flows: Dict[str, FlowContext] = {}
        self.window = window_seconds
        self.max_flows = max_flows
        self.evicted_count = 0
        self.total_flows_seen = 0
        logger.info(f"FlowTable initialized: max={max_flows}, timeout={window_seconds}s")
    
    def get_or_create_flow(self, flow_id: str, src_ip: str, dst_ip: str, 
                          src_port: int, dst_port: int, protocol: str) -> FlowContext:
        """
        Get existing flow or create new one
        
        Args:
            flow_id: 5-tuple hash
            src_ip, dst_ip: IP addresses
            src_port, dst_port: Ports
            protocol: 'tcp', 'udp', 'icmp'
        
        Returns:
            FlowContext object
        """
        if flow_id not in self.flows:
            if len(self.flows) >= self.max_flows:
                self._evict_oldest()
            
            flow = FlowContext(
                flow_id=flow_id,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol
            )
            self.flows[flow_id] = flow
            self.total_flows_seen += 1
            logger.debug(f"New flow created: {flow}")
        
        return self.flows[flow_id]
    
    def update_packet_stats(self, flow_id: str, direction: str, 
                          packet_len: int, timestamp: float):
        """Update packet count and bytes for flow"""
        if flow_id in self.flows:
            flow = self.flows[flow_id]
            flow.add_packet(direction, packet_len, timestamp)
    
    def update_tcp_state(self, flow_id: str, flags: str):
        """Update TCP state machine based on flags"""
        if flow_id not in self.flows:
            return
        
        flow = self.flows[flow_id]
        
        if not flags:
            return
        
        # Simple TCP state tracking
        if "SYN" in flags and "ACK" not in flags:
            flow.seen_syn = True
            flow.tcp_state = "SYN_SENT"
        elif "SYN" in flags and "ACK" in flags:
            flow.seen_ack = True
            flow.state = FlowState.ESTABLISHED
            flow.tcp_state = "ESTABLISHED"
        elif "ACK" in flags and flow.seen_syn:
            flow.seen_ack = True
            if flow.state != FlowState.ESTABLISHED:
                flow.state = FlowState.ESTABLISHED
                flow.tcp_state = "ESTABLISHED"
        elif "FIN" in flags:
            flow.seen_fin = True
            flow.state = FlowState.CLOSING
            flow.tcp_state = "CLOSING"
        elif "RST" in flags:
            flow.seen_rst = True
            flow.state = FlowState.CLOSED
            flow.tcp_state = "RESET"
    
    def set_flow_action(self, flow_id: str, action: FlowAction):
        """Set IPS action for this flow"""
        if flow_id in self.flows:
            self.flows[flow_id].action = action
            logger.info(f"Flow {flow_id}: action set to {action.value}")
    
    def add_model_vote(self, flow_id: str, model_name: str, score: float):
        """Record model detection vote"""
        if flow_id in self.flows:
            self.flows[flow_id].model_votes[model_name] = score
            if model_name not in self.flows[flow_id].triggered_models:
                self.flows[flow_id].triggered_models.append(model_name)
    
    def set_risk_score(self, flow_id: str, risk_score: float):
        """Set overall risk score for flow"""
        if flow_id in self.flows:
            self.flows[flow_id].risk_score = risk_score
    
    def escalate_flow(self, flow_id: str, levels: int = 1):
        """Escalate threat level for repeat offenders"""
        if flow_id in self.flows:
            flow = self.flows[flow_id]
            flow.escalation_level = min(4, flow.escalation_level + levels)
            # Map escalation to block time
            block_times = {0: 0, 1: 300, 2: 1800, 3: 3600, 4: 86400}
            flow.block_time_remaining = block_times[flow.escalation_level]
            logger.info(f"Flow {flow_id} escalated to level {flow.escalation_level}")
    
    def cleanup_expired_flows(self):
        """Remove flows idle > window_seconds"""
        now = datetime.now().timestamp()
        expired = []
        
        for fid, flow in self.flows.items():
            idle_time = now - flow.last_seen
            if idle_time > self.window:
                expired.append(fid)
        
        for fid in expired:
            del self.flows[fid]
            self.evicted_count += 1
        
        if expired:
            logger.debug(f"Cleaned {len(expired)} expired flows")
        
        return len(expired)
    
    def _evict_oldest(self):
        """Evict oldest flow when at capacity (LRU policy)"""
        if not self.flows:
            return
        
        oldest_id = min(self.flows.keys(), 
                       key=lambda fid: self.flows[fid].last_seen)
        del self.flows[oldest_id]
        self.evicted_count += 1
        logger.debug(f"Evicted oldest flow {oldest_id}")
    
    def get_flow(self, flow_id: str) -> Optional[FlowContext]:
        """Get flow by flow_id"""
        return self.flows.get(flow_id)
    
    def get_all_flows(self) -> Dict[str, FlowContext]:
        """Get all flows"""
        return self.flows.copy()
    
    def get_active_flows(self) -> List[FlowContext]:
        """Get flows still active (not closed/timeout)"""
        return [f for f in self.flows.values() 
                if f.state not in [FlowState.CLOSED, FlowState.TIMEOUT]]
    
    def get_blocked_flows(self) -> List[FlowContext]:
        """Get currently blocked flows"""
        return [f for f in self.flows.values() if f.is_blocked()]
    
    def get_stats(self) -> FlowStats:
        """Get current flow table statistics"""
        closed_count = sum(1 for f in self.flows.values() 
                          if f.state in [FlowState.CLOSED, FlowState.TIMEOUT])
        active_count = len(self.flows) - closed_count
        
        total_bytes = sum(f.get_total_bytes() for f in self.flows.values())
        avg_packets = sum(f.get_total_packets() for f in self.flows.values()) / len(self.flows) if self.flows else 0
        avg_bytes = total_bytes / len(self.flows) if self.flows else 0
        
        # Rough memory estimate (each FlowContext ~1KB)
        memory_kb = len(self.flows) * 1
        
        return FlowStats(
            total_flows=self.total_flows_seen,
            active_flows=active_count,
            closed_flows=closed_count,
            avg_packets_per_flow=avg_packets,
            avg_bytes_per_flow=avg_bytes,
            total_bytes_tracked=total_bytes,
            memory_usage_kb=memory_kb
        )
    
    def print_stats(self):
        """Log flow table statistics"""
        stats = self.get_stats()
        logger.info(
            f"FlowTable stats: active={stats.active_flows}, "
            f"closed={stats.closed_flows}, total_seen={stats.total_flows}, "
            f"evicted={self.evicted_count}, memory={stats.memory_usage_kb}KB"
        )
