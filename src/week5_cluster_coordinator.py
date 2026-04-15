"""
Week 5: Multi-Node Clustering & Distributed Coordination

Enables INIDS to operate as a distributed system with:
1. Leader election for coordination
2. Node health monitoring
3. Distributed event aggregation
4. Model synchronization across nodes
5. Automatic failover

Architecture:
- Etcd/Redis: Distributed state store
- Leader: Coordinates model updates and policy changes
- Workers: Process events and make predictions
- Heartbeat: Health monitoring every 5 seconds
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class NodeState(str, Enum):
    """Node health state."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class NodeRole(str, Enum):
    """Node role in cluster."""
    LEADER = "leader"
    WORKER = "worker"
    STANDBY = "standby"


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    hostname: str
    port: int
    role: NodeRole = NodeRole.WORKER
    state: NodeState = NodeState.HEALTHY
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_version: str = ""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    processed_events: int = 0
    cached_predictions: int = 0
    
    def is_healthy(self, heartbeat_timeout_sec: int = 15) -> bool:
        """Check if node is healthy based on heartbeat."""
        elapsed = (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()
        return elapsed < heartbeat_timeout_sec and self.state == NodeState.HEALTHY
    
    def update_heartbeat(self):
        """Update heartbeat timestamp."""
        self.last_heartbeat = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "port": self.port,
            "role": self.role.value,
            "state": self.state.value,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "model_version": self.model_version,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "processed_events": self.processed_events,
            "cached_predictions": self.cached_predictions,
            "is_healthy": self.is_healthy()
        }


@dataclass
class ClusterConfig:
    """Cluster configuration."""
    cluster_name: str = "inids-cluster"
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hostname: str = "localhost"
    port: int = 5000
    
    # Distributed state store
    state_store_url: str = "redis://localhost:6379"  # or etcd://localhost:2379
    state_store_type: str = "redis"  # redis or etcd
    
    # Clustering
    heartbeat_interval_sec: int = 5
    heartbeat_timeout_sec: int = 15
    leader_election_timeout_sec: int = 10
    
    # Model synchronization
    model_sync_interval_sec: int = 60
    
    # Event aggregation
    event_batch_size: int = 100
    event_batch_timeout_sec: int = 5


class LeaderElectionManager:
    """
    Manages leader election using distributed consensus.
    
    Algorithm:
    1. All nodes compete for leadership
    2. Winner is elected and holds lock
    3. Lock has TTL - automatically released on timeout
    4. Other nodes monitor leader and fallback if needed
    """
    
    def __init__(self, config: ClusterConfig, state_store):
        """Initialize leader election manager.
        
        Args:
            config: Cluster configuration
            state_store: Distributed state store (Redis/Etcd)
        """
        self.config = config
        self.state_store = state_store
        self.is_leader = False
        self.leader_node_id: Optional[str] = None
        self.election_lock_key = f"{config.cluster_name}:leader"
        self.election_lock_ttl = config.leader_election_timeout_sec
    
    async def attempt_leadership(self) -> bool:
        """Attempt to become leader.
        
        Returns:
            True if elected leader, False otherwise
        """
        try:
            # Try to acquire leadership lock with TTL
            acquired = await self.state_store.acquire_lock(
                self.election_lock_key,
                self.config.node_id,
                ttl=self.election_lock_ttl
            )
            
            if acquired:
                self.is_leader = True
                self.leader_node_id = self.config.node_id
                logger.info(f"[{self.config.node_id}] Became cluster leader")
                return True
            else:
                # Get current leader
                self.leader_node_id = await self.state_store.get(self.election_lock_key)
                self.is_leader = False
                logger.debug(f"[{self.config.node_id}] Leader is {self.leader_node_id}")
                return False
        
        except Exception as e:
            logger.error(f"Leader election error: {e}")
            return False
    
    async def renew_leadership(self) -> bool:
        """Renew leadership lock (leader calls this periodically).
        
        Returns:
            True if renewed, False if lost leadership
        """
        if not self.is_leader:
            return False
        
        try:
            renewed = await self.state_store.renew_lock(
                self.election_lock_key,
                self.config.node_id,
                ttl=self.election_lock_ttl
            )
            
            if not renewed:
                logger.warning(f"[{self.config.node_id}] Lost leadership")
                self.is_leader = False
            
            return renewed
        
        except Exception as e:
            logger.error(f"Leadership renewal error: {e}")
            self.is_leader = False
            return False
    
    async def release_leadership(self):
        """Release leadership lock."""
        if self.is_leader:
            try:
                await self.state_store.release_lock(self.election_lock_key, self.config.node_id)
                self.is_leader = False
                logger.info(f"[{self.config.node_id}] Released leadership")
            except Exception as e:
                logger.error(f"Error releasing leadership: {e}")


class ClusterMembershipManager:
    """
    Manages cluster membership - tracking alive nodes.
    
    Features:
    - Node discovery and registration
    - Heartbeat monitoring
    - Automatic node removal on timeout
    - Member list updates to all nodes
    """
    
    def __init__(self, config: ClusterConfig, state_store):
        """Initialize membership manager.
        
        Args:
            config: Cluster configuration
            state_store: Distributed state store
        """
        self.config = config
        self.state_store = state_store
        self.local_node: NodeInfo = NodeInfo(
            node_id=config.node_id,
            hostname=config.hostname,
            port=config.port
        )
        self.members: Dict[str, NodeInfo] = {}
        self.node_registry_key = f"{config.cluster_name}:members"
    
    async def register_node(self) -> bool:
        """Register this node in cluster.
        
        Returns:
            True if registered successfully
        """
        try:
            # Store node info in distributed state
            await self.state_store.set_hash(
                self.node_registry_key,
                self.config.node_id,
                self.local_node.to_dict()
            )
            logger.info(f"Registered node {self.config.node_id} in cluster")
            self.members[self.config.node_id] = self.local_node
            return True
        
        except Exception as e:
            logger.error(f"Error registering node: {e}")
            return False
    
    async def send_heartbeat(self) -> bool:
        """Send heartbeat to cluster.
        
        Returns:
            True if heartbeat sent successfully
        """
        try:
            self.local_node.update_heartbeat()
            await self.state_store.set_hash(
                self.node_registry_key,
                self.config.node_id,
                self.local_node.to_dict(),
                ttl=self.config.heartbeat_timeout_sec * 2  # Allow 2x timeout before removal
            )
            return True
        
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
            return False
    
    async def get_cluster_members(self) -> List[NodeInfo]:
        """Get all healthy cluster members.
        
        Returns:
            List of healthy NodeInfo objects
        """
        try:
            members_dict = await self.state_store.get_hash(self.node_registry_key)
            healthy_members = []
            
            for node_id, node_data in members_dict.items():
                node = NodeInfo(**node_data)
                if node.is_healthy(self.config.heartbeat_timeout_sec):
                    healthy_members.append(node)
                    self.members[node_id] = node
            
            return healthy_members
        
        except Exception as e:
            logger.error(f"Error getting cluster members: {e}")
            return []
    
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics.
        
        Returns:
            Cluster statistics
        """
        members = await self.get_cluster_members()
        
        total_processed = sum(m.processed_events for m in members)
        total_cached = sum(m.cached_predictions for m in members)
        avg_cpu = sum(m.cpu_usage for m in members) / len(members) if members else 0
        avg_memory = sum(m.memory_usage for m in members) / len(members) if members else 0
        
        return {
            "cluster_name": self.config.cluster_name,
            "total_members": len(members),
            "total_processed_events": total_processed,
            "total_cached_predictions": total_cached,
            "avg_cpu_usage": avg_cpu,
            "avg_memory_usage": avg_memory,
            "members": [m.to_dict() for m in members]
        }


class EventAggregator:
    """
    Aggregates and correlates detection events across cluster.
    
    Features:
    - Central event collection from all nodes
    - Consensus-based alerting (multiple nodes detecting same attack)
    - Cross-node attack correlation
    - Distributed attack timeline
    """
    
    def __init__(self, config: ClusterConfig, state_store):
        """Initialize event aggregator.
        
        Args:
            config: Cluster configuration
            state_store: Distributed state store
        """
        self.config = config
        self.state_store = state_store
        self.event_queue_key = f"{config.cluster_name}:events"
        self.correlation_window_sec = 30  # Correlation window for same attack
        self.consensus_threshold = 2  # Nodes needed to confirm attack
    
    async def emit_event(self, event: Dict[str, Any]) -> bool:
        """Emit detection event to cluster event stream.
        
        Args:
            event: Detection event
            
        Returns:
            True if event published successfully
        """
        try:
            # Add metadata
            event["node_id"] = self.config.node_id
            event["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Publish to distributed event stream
            await self.state_store.push_to_stream(self.event_queue_key, event)
            return True
        
        except Exception as e:
            logger.error(f"Error emitting event: {e}")
            return False
    
    async def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events from cluster.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        try:
            events = await self.state_store.read_stream(
                self.event_queue_key,
                count=limit
            )
            return events
        
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []
    
    async def check_consensus_alert(
        self,
        attack_signature: str,
        source_ip: str
    ) -> bool:
        """Check if multiple nodes detected same attack (consensus).
        
        Args:
            attack_signature: Attack signature/pattern
            source_ip: Source IP address
            
        Returns:
            True if consensus reached (multiple detections)
        """
        try:
            correlation_key = f"{attack_signature}:{source_ip}"
            start_time = datetime.now(timezone.utc) - timedelta(
                seconds=self.correlation_window_sec
            )
            
            # Get recent events matching this attack
            recent_events = await self.get_recent_events(limit=1000)
            matching_events = [
                e for e in recent_events
                if e.get("attack_signature") == attack_signature
                and e.get("source_ip") == source_ip
                and datetime.fromisoformat(e.get("timestamp", "")) > start_time
            ]
            
            # Consensus: multiple nodes detected
            node_ids = set(e.get("node_id") for e in matching_events)
            return len(node_ids) >= self.consensus_threshold
        
        except Exception as e:
            logger.error(f"Error checking consensus: {e}")
            return False


class ClusterCoordinator:
    """
    Orchestrates cluster operations: leadership, membership, events.
    """
    
    def __init__(self, config: ClusterConfig, state_store):
        """Initialize cluster coordinator.
        
        Args:
            config: Cluster configuration
            state_store: Distributed state store
        """
        self.config = config
        self.state_store = state_store
        
        self.leader_manager = LeaderElectionManager(config, state_store)
        self.membership_manager = ClusterMembershipManager(config, state_store)
        self.event_aggregator = EventAggregator(config, state_store)
        
        self.is_running = False
    
    async def start(self):
        """Start cluster coordination."""
        self.is_running = True
        logger.info(f"Starting cluster coordinator for node {self.config.node_id}")
        
        # Register node
        await self.membership_manager.register_node()
        
        # Start background tasks
        await asyncio.gather(
            self._leadership_loop(),
            self._heartbeat_loop(),
            self._health_check_loop(),
            return_exceptions=True
        )
    
    async def stop(self):
        """Stop cluster coordination."""
        self.is_running = False
        await self.leader_manager.release_leadership()
        logger.info("Cluster coordinator stopped")
    
    async def _leadership_loop(self):
        """Periodically attempt/renew leadership."""
        while self.is_running:
            try:
                if self.leader_manager.is_leader:
                    # Renew leadership if leader
                    await self.leader_manager.renew_leadership()
                else:
                    # Try to become leader
                    await self.leader_manager.attempt_leadership()
                
                await asyncio.sleep(self.config.leader_election_timeout_sec / 2)
            
            except Exception as e:
                logger.error(f"Leadership loop error: {e}")
                await asyncio.sleep(1)
    
    async def _heartbeat_loop(self):
        """Periodically send heartbeat."""
        while self.is_running:
            try:
                await self.membership_manager.send_heartbeat()
                await asyncio.sleep(self.config.heartbeat_interval_sec)
            
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(1)
    
    async def _health_check_loop(self):
        """Periodically check cluster health."""
        while self.is_running:
            try:
                members = await self.membership_manager.get_cluster_members()
                logger.debug(f"Cluster health: {len(members)} healthy members")
                await asyncio.sleep(self.config.heartbeat_interval_sec * 2)
            
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(1)


# Global cluster coordinator instance
_cluster_coordinator: Optional[ClusterCoordinator] = None


def get_cluster_coordinator(config: ClusterConfig = None, state_store=None):
    """Get or create global cluster coordinator."""
    global _cluster_coordinator
    
    if _cluster_coordinator is None and config and state_store:
        _cluster_coordinator = ClusterCoordinator(config, state_store)
    
    return _cluster_coordinator


async def stop_cluster_coordinator():
    """Stop global cluster coordinator."""
    global _cluster_coordinator
    if _cluster_coordinator:
        await _cluster_coordinator.stop()
        _cluster_coordinator = None
