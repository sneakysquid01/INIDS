#!/usr/bin/env python3
"""Week 5 Validation Tests - Multi-Node Clustering"""

import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from week5_cluster_coordinator import (
    NodeState, NodeRole, NodeInfo, ClusterConfig, LeaderElectionManager,
    ClusterMembershipManager, EventAggregator, ClusterCoordinator,
    get_cluster_coordinator
)
from distributed_state_store import RedisStateStore, EtcdStateStore, create_state_store
from distributed_model_manager import ModelVersion, DistributedModelManager, ModelSyncService

def test_node_info_creation():
    """Test NodeInfo creation and health check."""
    print("Testing NodeInfo Creation... ", end="")
    
    node = NodeInfo(
        node_id="node-1",
        hostname="localhost",
        port=5000,
        role=NodeRole.LEADER
    )
    
    assert node.node_id == "node-1"
    assert node.role == NodeRole.LEADER
    assert node.is_healthy() == True
    
    print("✓")


def test_cluster_config():
    """Test cluster configuration."""
    print("Testing Cluster Config... ", end="")
    
    config = ClusterConfig(
        cluster_name="test-cluster",
        hostname="10.0.0.1",
        port=5001,
        state_store_type="redis"
    )
    
    assert config.cluster_name == "test-cluster"
    assert config.hostname == "10.0.0.1"
    assert config.state_store_type == "redis"
    
    print("✓")


def test_node_role_enum():
    """Test NodeRole enumeration."""
    print("Testing NodeRole Enum... ", end="")
    
    assert NodeRole.LEADER.value == "leader"
    assert NodeRole.WORKER.value == "worker"
    assert NodeRole.STANDBY.value == "standby"
    
    print("✓")


def test_node_state_enum():
    """Test NodeState enumeration."""
    print("Testing NodeState Enum... ", end="")
    
    assert NodeState.HEALTHY.value == "healthy"
    assert NodeState.DEGRADED.value == "degraded"
    assert NodeState.UNHEALTHY.value == "unhealthy"
    assert NodeState.OFFLINE.value == "offline"
    
    print("✓")


def test_node_info_to_dict():
    """Test NodeInfo serialization."""
    print("Testing NodeInfo Serialization... ", end="")
    
    node = NodeInfo(
        node_id="node-1",
        hostname="localhost",
        port=5000,
        model_version="v1.0"
    )
    
    node_dict = node.to_dict()
    
    assert node_dict["node_id"] == "node-1"
    assert node_dict["hostname"] == "localhost"
    assert node_dict["port"] == 5000
    assert "is_healthy" in node_dict
    
    print("✓")


def test_distributed_state_store_factory():
    """Test state store factory."""
    print("Testing State Store Factory... ", end="")
    
    redis_store = create_state_store("redis")
    assert isinstance(redis_store, RedisStateStore)
    
    etcd_store = create_state_store("etcd")
    assert isinstance(etcd_store, EtcdStateStore)
    
    print("✓")


def test_model_version():
    """Test ModelVersion dataclass."""
    print("Testing ModelVersion... ", end="")
    
    model = ModelVersion(
        version_id="anomaly-v1",
        model_name="anomaly_detection",
        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        checksum="abc123",
        source_node_id="node-1",
        size_bytes=1024,
        compatible_nodes=["node-1", "node-2"]
    )
    
    assert model.version_id == "anomaly-v1"
    assert model.model_name == "anomaly_detection"
    assert len(model.compatible_nodes) == 2
    
    print("✓")


def test_cluster_membership_initialization():
    """Test cluster membership manager initialization."""
    print("Testing Cluster Membership Init... ", end="")
    
    config = ClusterConfig(node_id="node-1", hostname="localhost")
    mock_store = AsyncMock()
    
    membership = ClusterMembershipManager(config, mock_store)
    
    assert membership.config == config
    assert membership.local_node.node_id == "node-1"
    assert len(membership.members) == 0
    
    print("✓")


def test_event_aggregator_initialization():
    """Test event aggregator initialization."""
    print("Testing Event Aggregator Init... ", end="")
    
    config = ClusterConfig(cluster_name="test")
    mock_store = AsyncMock()
    
    aggregator = EventAggregator(config, mock_store)
    
    assert aggregator.config == config
    assert aggregator.consensus_threshold == 2
    assert aggregator.correlation_window_sec == 30
    
    print("✓")


def test_leader_election_manager_initialization():
    """Test leader election manager."""
    print("Testing Leader Election Manager Init... ", end="")
    
    config = ClusterConfig(node_id="node-1")
    mock_store = AsyncMock()
    
    leader_mgr = LeaderElectionManager(config, mock_store)
    
    assert leader_mgr.is_leader == False
    assert leader_mgr.leader_node_id is None
    assert leader_mgr.election_lock_key == "inids-cluster:leader"
    
    print("✓")


def test_cluster_coordinator_initialization():
    """Test cluster coordinator."""
    print("Testing Cluster Coordinator Init... ", end="")
    
    config = ClusterConfig(node_id="node-1")
    mock_store = AsyncMock()
    
    coordinator = ClusterCoordinator(config, mock_store)
    
    assert coordinator.config == config
    assert coordinator.is_running == False
    assert coordinator.leader_manager is not None
    assert coordinator.membership_manager is not None
    assert coordinator.event_aggregator is not None
    
    print("✓")


def test_model_sync_service_initialization():
    """Test model sync service."""
    print("Testing Model Sync Service Init... ", end="")
    
    mock_manager = AsyncMock()
    config = ClusterConfig()
    
    sync_service = ModelSyncService(mock_manager, config)
    
    assert sync_service.model_manager == mock_manager
    assert len(sync_service.local_models) == 0
    
    print("✓")


def test_distributed_model_manager_initialization():
    """Test distributed model manager."""
    print("Testing Distributed Model Manager Init... ", end="")
    
    mock_store = AsyncMock()
    config = ClusterConfig(cluster_name="test")
    
    model_mgr = DistributedModelManager(mock_store, config)
    
    assert model_mgr.state_store == mock_store
    assert model_mgr.model_registry_key == "test:models"
    assert model_mgr.model_updates_stream == "test:model_updates"
    
    print("✓")


if __name__ == "__main__":
    print("=" * 60)
    print("Week 5 Validation Tests - Multi-Node Clustering")
    print("=" * 60)
    print()
    
    tests = [
        test_node_info_creation,
        test_cluster_config,
        test_node_role_enum,
        test_node_state_enum,
        test_node_info_to_dict,
        test_distributed_state_store_factory,
        test_model_version,
        test_cluster_membership_initialization,
        test_event_aggregator_initialization,
        test_leader_election_manager_initialization,
        test_cluster_coordinator_initialization,
        test_model_sync_service_initialization,
        test_distributed_model_manager_initialization,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)
