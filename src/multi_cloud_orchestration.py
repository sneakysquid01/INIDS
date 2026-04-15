"""
Multi-Cloud Orchestration (Week 6)

Manages deployments and operations across multiple cloud providers:
- Unified configuration management
- Automatic provider selection
- Failover between providers
- Cost optimization and load balancing
"""

import logging
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


@dataclass
class CloudDeploymentConfig:
    """Configuration for multi-cloud deployment."""
    primary_provider: CloudProvider
    fallback_providers: List[CloudProvider] = field(default_factory=list)
    enable_multi_region: bool = False
    cost_optimization: bool = True
    auto_failover: bool = True
    health_check_interval_sec: int = 60
    
    # Provider-specific configs
    aws_config: Dict[str, Any] = field(default_factory=dict)
    azure_config: Dict[str, Any] = field(default_factory=dict)
    gcp_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderHealth:
    """Health status of a cloud provider."""
    provider: CloudProvider
    is_healthy: bool
    last_check: datetime
    error_count: int = 0
    response_time_ms: float = 0.0
    available_capacity_percent: float = 100.0


class MultiCloudOrchestrator:
    """Orchestrates operations across multiple cloud providers."""
    
    def __init__(self, config: CloudDeploymentConfig):
        """Initialize multi-cloud orchestrator.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.providers: Dict[CloudProvider, Any] = {}
        self.health_status: Dict[CloudProvider, ProviderHealth] = {}
        self.current_provider = config.primary_provider
        self.operation_log: List[Dict] = []
    
    async def initialize(self) -> bool:
        """Initialize all configured providers.
        
        Returns:
            True if at least primary provider initialized successfully
        """
        try:
            from src.cloud_adapters import get_cloud_adapter
            
            # Initialize primary provider
            primary_config = self._get_provider_config(self.config.primary_provider)
            primary_adapter = get_cloud_adapter(
                self.config.primary_provider.value,
                **primary_config
            )
            
            if await primary_adapter.connect():
                self.providers[self.config.primary_provider] = primary_adapter
                self.health_status[self.config.primary_provider] = ProviderHealth(
                    provider=self.config.primary_provider,
                    is_healthy=True,
                    last_check=datetime.now(timezone.utc)
                )
                logger.info(f"Initialized primary provider: {self.config.primary_provider.value}")
            else:
                logger.warning(f"Failed to initialize primary provider")
            
            # Initialize fallback providers
            for provider in self.config.fallback_providers:
                try:
                    fallback_config = self._get_provider_config(provider)
                    fallback_adapter = get_cloud_adapter(
                        provider.value,
                        **fallback_config
                    )
                    
                    if await fallback_adapter.connect():
                        self.providers[provider] = fallback_adapter
                        self.health_status[provider] = ProviderHealth(
                            provider=provider,
                            is_healthy=True,
                            last_check=datetime.now(timezone.utc)
                        )
                        logger.info(f"Initialized fallback provider: {provider.value}")
                
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider.value}: {e}")
            
            return len(self.providers) > 0
        
        except Exception as e:
            logger.error(f"Error initializing multi-cloud orchestrator: {e}")
            return False
    
    async def upload_model(
        self,
        model_name: str,
        local_path: str,
        providers: List[CloudProvider] = None
    ) -> bool:
        """Upload model to cloud provider(s).
        
        Args:
            model_name: Name of model
            local_path: Local file path
            providers: List of providers to upload to (defaults to all)
            
        Returns:
            True if upload successful to at least one provider
        """
        try:
            target_providers = providers or list(self.providers.keys())
            remote_path = f"models/{model_name}"
            successful = False
            
            for provider in target_providers:
                adapter = self.providers.get(provider)
                if not adapter:
                    logger.warning(f"Provider {provider.value} not available")
                    continue
                
                try:
                    if await adapter.upload_file(
                        local_path,
                        remote_path,
                        metadata={
                            "model_name": model_name,
                            "uploaded_at": datetime.now(timezone.utc).isoformat()
                        }
                    ):
                        successful = True
                        self._log_operation(
                            operation="upload_model",
                            provider=provider.value,
                            status="success",
                            details={"model_name": model_name}
                        )
                
                except Exception as e:
                    logger.error(f"Error uploading to {provider.value}: {e}")
                    self._log_operation(
                        operation="upload_model",
                        provider=provider.value,
                        status="failed",
                        details={"error": str(e)}
                    )
            
            return successful
        
        except Exception as e:
            logger.error(f"Error in upload_model: {e}")
            return False
    
    async def download_model(
        self,
        model_name: str,
        local_path: str,
        provider: CloudProvider = None
    ) -> bool:
        """Download model from cloud provider.
        
        Args:
            model_name: Name of model
            local_path: Local destination path
            provider: Specific provider (defaults to primary)
            
        Returns:
            True if download successful
        """
        try:
            target_provider = provider or self.current_provider
            remote_path = f"models/{model_name}"
            
            adapter = self.providers.get(target_provider)
            if not adapter:
                logger.error(f"Provider {target_provider.value} not available")
                return False
            
            if await adapter.download_file(remote_path, local_path):
                self._log_operation(
                    operation="download_model",
                    provider=target_provider.value,
                    status="success",
                    details={"model_name": model_name}
                )
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error in download_model: {e}")
            return False
    
    async def send_metric(
        self,
        metric,
        providers: List[CloudProvider] = None
    ) -> bool:
        """Send metric to cloud provider(s).
        
        Args:
            metric: CloudMetric object
            providers: List of providers (defaults to all)
            
        Returns:
            True if sent to at least one provider
        """
        try:
            target_providers = providers or list(self.providers.keys())
            successful = False
            
            for provider in target_providers:
                adapter = self.providers.get(provider)
                if not adapter:
                    continue
                
                try:
                    if await adapter.send_metric(metric):
                        successful = True
                
                except Exception as e:
                    logger.error(f"Error sending metric to {provider.value}: {e}")
            
            return successful
        
        except Exception as e:
            logger.error(f"Error in send_metric: {e}")
            return False
    
    async def publish_alert(
        self,
        alert,
        providers: List[CloudProvider] = None
    ) -> bool:
        """Publish alert to cloud provider(s).
        
        Args:
            alert: CloudAlert object
            providers: List of providers (defaults to all)
            
        Returns:
            True if published to at least one provider
        """
        try:
            target_providers = providers or list(self.providers.keys())
            successful = False
            
            for provider in target_providers:
                adapter = self.providers.get(provider)
                if not adapter:
                    continue
                
                try:
                    if await adapter.publish_alert(alert):
                        successful = True
                        self._log_operation(
                            operation="publish_alert",
                            provider=provider.value,
                            status="success",
                            details={"alert_id": alert.alert_id}
                        )
                
                except Exception as e:
                    logger.error(f"Error publishing alert to {provider.value}: {e}")
                    self._log_operation(
                        operation="publish_alert",
                        provider=provider.value,
                        status="failed",
                        details={"error": str(e)}
                    )
            
            return successful
        
        except Exception as e:
            logger.error(f"Error in publish_alert: {e}")
            return False
    
    async def health_check(self) -> Dict[CloudProvider, bool]:
        """Check health of all providers.
        
        Returns:
            Dict mapping provider to health status
        """
        try:
            health_results = {}
            
            for provider, adapter in self.providers.items():
                try:
                    # Simple health check: try to list files
                    files = await adapter.list_files("")
                    is_healthy = True
                    
                except Exception as e:
                    logger.warning(f"Health check failed for {provider.value}: {e}")
                    is_healthy = False
                
                health_results[provider] = is_healthy
                
                # Update health status
                if provider in self.health_status:
                    self.health_status[provider].is_healthy = is_healthy
                    self.health_status[provider].last_check = datetime.now(timezone.utc)
            
            return health_results
        
        except Exception as e:
            logger.error(f"Error in health_check: {e}")
            return {p: False for p in self.providers.keys()}
    
    async def failover(self) -> bool:
        """Failover to next available provider.
        
        Returns:
            True if failover successful
        """
        try:
            current_healthy = self.health_status.get(
                self.current_provider, ProviderHealth(
                    provider=self.current_provider,
                    is_healthy=False,
                    last_check=datetime.now(timezone.utc)
                )
            ).is_healthy
            
            if current_healthy:
                logger.info(f"Current provider {self.current_provider.value} is healthy")
                return True
            
            # Find next healthy provider
            for provider in [self.config.primary_provider] + self.config.fallback_providers:
                if provider in self.providers:
                    health = self.health_status.get(provider)
                    if health and health.is_healthy:
                        self.current_provider = provider
                        logger.info(f"Failover to {provider.value}")
                        self._log_operation(
                            operation="failover",
                            provider=provider.value,
                            status="success",
                            details={}
                        )
                        return True
            
            logger.error("No healthy providers available for failover")
            return False
        
        except Exception as e:
            logger.error(f"Error in failover: {e}")
            return False
    
    def _get_provider_config(self, provider: CloudProvider) -> Dict:
        """Get configuration for specific provider."""
        if provider == CloudProvider.AWS:
            return self.config.aws_config
        elif provider == CloudProvider.AZURE:
            return self.config.azure_config
        elif provider == CloudProvider.GCP:
            return self.config.gcp_config
        return {}
    
    def _log_operation(
        self,
        operation: str,
        provider: str,
        status: str,
        details: Dict
    ):
        """Log multi-cloud operation."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "provider": provider,
            "status": status,
            "details": details
        }
        self.operation_log.append(log_entry)
        
        logger.info(f"[{operation}] {provider}: {status}")
    
    def get_operation_log(self, limit: int = 100) -> List[Dict]:
        """Get recent operation logs.
        
        Args:
            limit: Maximum number of entries
            
        Returns:
            List of log entries
        """
        return self.operation_log[-limit:]


# Global orchestrator instance
_orchestrator = None


def get_multi_cloud_orchestrator(config: CloudDeploymentConfig = None):
    """Get or create multi-cloud orchestrator instance.
    
    Args:
        config: Deployment configuration (used on first call)
        
    Returns:
        MultiCloudOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None and config:
        _orchestrator = MultiCloudOrchestrator(config)
    return _orchestrator
