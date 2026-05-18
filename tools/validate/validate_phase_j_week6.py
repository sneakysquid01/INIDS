#!/usr/bin/env python3
"""Week 6 Validation Tests - Cloud Integrations"""

import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cloud_adapters import (
    CloudMetric, CloudAlert, CloudStorageAdapter, CloudMonitoringAdapter,
    CloudMessagingAdapter, AWSAdapter, AzureAdapter, GCPAdapter,
    get_cloud_adapter
)
from multi_cloud_orchestration import (
    CloudProvider, CloudDeploymentConfig, ProviderHealth,
    MultiCloudOrchestrator, get_multi_cloud_orchestrator
)
import datetime


def test_cloud_metric():
    """Test CloudMetric dataclass."""
    print("Testing CloudMetric... ", end="")
    
    metric = CloudMetric(
        name="cpu_usage",
        value=75.5,
        unit="Percent",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        tags={"region": "us-east-1", "instance": "node-1"}
    )
    
    assert metric.name == "cpu_usage"
    assert metric.value == 75.5
    assert len(metric.tags) == 2
    
    print("✓")


def test_cloud_alert():
    """Test CloudAlert dataclass."""
    print("Testing CloudAlert... ", end="")
    
    alert = CloudAlert(
        alert_id="alert-123",
        service="inids",
        severity="CRITICAL",
        message="Potential DDoS attack detected",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        metadata={"src_ip": "192.168.1.1", "port": 80}
    )
    
    assert alert.alert_id == "alert-123"
    assert alert.severity == "CRITICAL"
    assert alert.metadata["src_ip"] == "192.168.1.1"
    
    print("✓")


def test_aws_adapter_initialization():
    """Test AWS adapter initialization."""
    print("Testing AWS Adapter Init... ", end="")
    
    adapter = AWSAdapter(
        region="us-west-2",
        s3_bucket="inids-models",
        access_key="fake_key",
        secret_key="fake_secret"
    )
    
    assert adapter.region == "us-west-2"
    assert adapter.s3_bucket == "inids-models"
    
    print("✓")


def test_azure_adapter_initialization():
    """Test Azure adapter initialization."""
    print("Testing Azure Adapter Init... ", end="")
    
    adapter = AzureAdapter(
        connection_string="DefaultEndpointsProtocol=https;...",
        account_name="iniidsaccount",
        container_name="models"
    )
    
    assert adapter.account_name == "iniidsaccount"
    assert adapter.container_name == "models"
    
    print("✓")


def test_gcp_adapter_initialization():
    """Test GCP adapter initialization."""
    print("Testing GCP Adapter Init... ", end="")
    
    adapter = GCPAdapter(
        project_id="inids-project",
        bucket_name="inids-models"
    )
    
    assert adapter.project_id == "inids-project"
    assert adapter.bucket_name == "inids-models"
    
    print("✓")


def test_cloud_adapter_factory():
    """Test cloud adapter factory."""
    print("Testing Cloud Adapter Factory... ", end="")
    
    aws = get_cloud_adapter("aws", region="us-east-1", s3_bucket="test")
    assert isinstance(aws, AWSAdapter)
    
    azure = get_cloud_adapter("azure", connection_string="test")
    assert isinstance(azure, AzureAdapter)
    
    gcp = get_cloud_adapter("gcp", project_id="test")
    assert isinstance(gcp, GCPAdapter)
    
    print("✓")


def test_cloud_provider_enum():
    """Test CloudProvider enumeration."""
    print("Testing CloudProvider Enum... ", end="")
    
    assert CloudProvider.AWS.value == "aws"
    assert CloudProvider.AZURE.value == "azure"
    assert CloudProvider.GCP.value == "gcp"
    
    print("✓")


def test_deployment_config():
    """Test CloudDeploymentConfig."""
    print("Testing Deployment Config... ", end="")
    
    config = CloudDeploymentConfig(
        primary_provider=CloudProvider.AWS,
        fallback_providers=[CloudProvider.AZURE, CloudProvider.GCP],
        enable_multi_region=True,
        cost_optimization=True
    )
    
    assert config.primary_provider == CloudProvider.AWS
    assert len(config.fallback_providers) == 2
    assert config.enable_multi_region == True
    
    print("✓")


def test_provider_health():
    """Test ProviderHealth dataclass."""
    print("Testing Provider Health... ", end="")
    
    health = ProviderHealth(
        provider=CloudProvider.AWS,
        is_healthy=True,
        last_check=datetime.datetime.now(datetime.timezone.utc),
        error_count=0,
        response_time_ms=45.2,
        available_capacity_percent=95.0
    )
    
    assert health.provider == CloudProvider.AWS
    assert health.is_healthy == True
    assert health.response_time_ms == 45.2
    
    print("✓")


def test_multi_cloud_orchestrator_initialization():
    """Test MultiCloudOrchestrator initialization."""
    print("Testing Multi-Cloud Orchestrator Init... ", end="")
    
    config = CloudDeploymentConfig(
        primary_provider=CloudProvider.AWS,
        fallback_providers=[CloudProvider.AZURE]
    )
    
    orchestrator = MultiCloudOrchestrator(config)
    
    assert orchestrator.config == config
    assert orchestrator.current_provider == CloudProvider.AWS
    assert len(orchestrator.providers) == 0  # Not connected yet
    
    print("✓")


def test_orchestrator_health_tracking():
    """Test health status tracking."""
    print("Testing Health Tracking... ", end="")
    
    config = CloudDeploymentConfig(
        primary_provider=CloudProvider.AWS,
        fallback_providers=[]
    )
    
    orchestrator = MultiCloudOrchestrator(config)
    
    # Manually add health status
    orchestrator.health_status[CloudProvider.AWS] = ProviderHealth(
        provider=CloudProvider.AWS,
        is_healthy=True,
        last_check=datetime.datetime.now(datetime.timezone.utc)
    )
    
    assert CloudProvider.AWS in orchestrator.health_status
    assert orchestrator.health_status[CloudProvider.AWS].is_healthy == True
    
    print("✓")


def test_orchestrator_operation_logging():
    """Test operation logging."""
    print("Testing Operation Logging... ", end="")
    
    config = CloudDeploymentConfig(
        primary_provider=CloudProvider.AWS
    )
    
    orchestrator = MultiCloudOrchestrator(config)
    
    orchestrator._log_operation(
        operation="upload_model",
        provider="aws",
        status="success",
        details={"model_name": "detector"}
    )
    
    assert len(orchestrator.operation_log) == 1
    assert orchestrator.operation_log[0]["operation"] == "upload_model"
    
    print("✓")


def test_orchestrator_get_operation_log():
    """Test getting operation logs."""
    print("Testing Get Operation Log... ", end="")
    
    config = CloudDeploymentConfig(
        primary_provider=CloudProvider.AWS
    )
    
    orchestrator = MultiCloudOrchestrator(config)
    
    # Add multiple operations
    for i in range(5):
        orchestrator._log_operation(
            operation=f"op_{i}",
            provider="aws",
            status="success",
            details={}
        )
    
    logs = orchestrator.get_operation_log(limit=3)
    assert len(logs) == 3
    
    print("✓")


def test_global_orchestrator_singleton():
    """Test global orchestrator singleton."""
    print("Testing Global Orchestrator Singleton... ", end="")
    
    config = CloudDeploymentConfig(
        primary_provider=CloudProvider.AWS
    )
    
    orchestrator1 = get_multi_cloud_orchestrator(config)
    orchestrator2 = get_multi_cloud_orchestrator()
    
    assert orchestrator1 is orchestrator2
    
    print("✓")


if __name__ == "__main__":
    print("=" * 60)
    print("Week 6 Validation Tests - Cloud Integrations")
    print("=" * 60)
    print()
    
    tests = [
        test_cloud_metric,
        test_cloud_alert,
        test_aws_adapter_initialization,
        test_azure_adapter_initialization,
        test_gcp_adapter_initialization,
        test_cloud_adapter_factory,
        test_cloud_provider_enum,
        test_deployment_config,
        test_provider_health,
        test_multi_cloud_orchestrator_initialization,
        test_orchestrator_health_tracking,
        test_orchestrator_operation_logging,
        test_orchestrator_get_operation_log,
        test_global_orchestrator_singleton,
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
