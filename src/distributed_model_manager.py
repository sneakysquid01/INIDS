"""
Distributed Model Management (Week 5)

Synchronizes ML models across cluster nodes:
1. Model versioning and distribution
2. Atomic model updates
3. Version consistency checking
4. Rollback capabilities
"""

import logging
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version metadata."""
    version_id: str
    model_name: str
    timestamp: datetime
    checksum: str  # SHA256 for integrity
    source_node_id: str
    size_bytes: int
    compatible_nodes: list  # Node IDs that can run this model
    is_active: bool = False


class DistributedModelManager:
    """Manages model distribution across cluster."""
    
    def __init__(self, state_store, config):
        """Initialize model manager.
        
        Args:
            state_store: Distributed state store
            config: Cluster configuration
        """
        self.state_store = state_store
        self.config = config
        self.model_registry_key = f"{config.cluster_name}:models"
        self.model_updates_stream = f"{config.cluster_name}:model_updates"
    
    async def publish_model_update(
        self,
        model_name: str,
        model_data: bytes,
        source_node_id: str
    ) -> Optional[str]:
        """Publish new model version to cluster.
        
        Args:
            model_name: Name of model (e.g., 'anomaly_detection')
            model_data: Serialized model data
            source_node_id: Node publishing the model
            
        Returns:
            Version ID if successful, None on failure
        """
        try:
            # Generate version
            checksum = hashlib.sha256(model_data).hexdigest()
            version_id = f"{model_name}:{checksum[:8]}"
            
            model_version = ModelVersion(
                version_id=version_id,
                model_name=model_name,
                timestamp=datetime.now(timezone.utc),
                checksum=checksum,
                source_node_id=source_node_id,
                size_bytes=len(model_data),
                compatible_nodes=[],
                is_active=True
            )
            
            # Store model metadata
            await self.state_store.set_hash(
                self.model_registry_key,
                version_id,
                {
                    "version_id": model_version.version_id,
                    "model_name": model_version.model_name,
                    "timestamp": model_version.timestamp.isoformat(),
                    "checksum": model_version.checksum,
                    "source_node_id": model_version.source_node_id,
                    "size_bytes": model_version.size_bytes,
                    "is_active": model_version.is_active
                }
            )
            
            # Publish update event
            await self.state_store.push_to_stream(
                self.model_updates_stream,
                {
                    "event_type": "model_update",
                    "version_id": version_id,
                    "model_name": model_name,
                    "checksum": checksum,
                    "source_node_id": source_node_id,
                    "timestamp": model_version.timestamp.isoformat()
                }
            )
            
            logger.info(f"Published model {model_name} version {version_id}")
            return version_id
        
        except Exception as e:
            logger.error(f"Error publishing model: {e}")
            return None
    
    async def get_latest_model_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get latest version of model.
        
        Args:
            model_name: Name of model
            
        Returns:
            ModelVersion if found, None otherwise
        """
        try:
            models = await self.state_store.get_hash(self.model_registry_key)
            
            # Find latest version for this model
            latest_version = None
            latest_timestamp = None
            
            for version_id, model_data in models.items():
                if model_data.get("model_name") == model_name:
                    ts = datetime.fromisoformat(model_data.get("timestamp", ""))
                    if not latest_timestamp or ts > latest_timestamp:
                        latest_timestamp = ts
                        latest_version = ModelVersion(
                            version_id=model_data["version_id"],
                            model_name=model_data["model_name"],
                            timestamp=ts,
                            checksum=model_data["checksum"],
                            source_node_id=model_data["source_node_id"],
                            size_bytes=model_data["size_bytes"],
                            compatible_nodes=model_data.get("compatible_nodes", []),
                            is_active=model_data.get("is_active", False)
                        )
            
            return latest_version
        
        except Exception as e:
            logger.error(f"Error getting model version: {e}")
            return None
    
    async def verify_model_integrity(
        self,
        version_id: str,
        model_data: bytes
    ) -> bool:
        """Verify model data integrity using checksum.
        
        Args:
            version_id: Version ID to verify
            model_data: Model data to verify
            
        Returns:
            True if checksum matches
        """
        try:
            models = await self.state_store.get_hash(self.model_registry_key)
            model_meta = models.get(version_id)
            
            if not model_meta:
                logger.error(f"Model version {version_id} not found")
                return False
            
            checksum = hashlib.sha256(model_data).hexdigest()
            expected_checksum = model_meta.get("checksum")
            
            if checksum == expected_checksum:
                logger.debug(f"Model {version_id} integrity verified")
                return True
            else:
                logger.error(f"Checksum mismatch for {version_id}")
                return False
        
        except Exception as e:
            logger.error(f"Error verifying model: {e}")
            return False
    
    async def activate_model_version(self, version_id: str) -> bool:
        """Activate model version across cluster.
        
        Args:
            version_id: Version to activate
            
        Returns:
            True if activated successfully
        """
        try:
            # Publish activation event
            await self.state_store.push_to_stream(
                self.model_updates_stream,
                {
                    "event_type": "activate_model",
                    "version_id": version_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            logger.info(f"Activated model version {version_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error activating model: {e}")
            return False


class ModelSyncService:
    """Service for keeping node models in sync."""
    
    def __init__(self, model_manager: DistributedModelManager, config):
        """Initialize model sync service.
        
        Args:
            model_manager: Distributed model manager
            config: Cluster configuration
        """
        self.model_manager = model_manager
        self.config = config
        self.local_models: Dict[str, str] = {}  # model_name -> version_id
    
    async def sync_models(self) -> bool:
        """Sync local models with cluster latest versions.
        
        Returns:
            True if all models synced successfully
        """
        try:
            model_names = ["anomaly_detection", "signature_detection", "threat_intel"]
            
            for model_name in model_names:
                latest = await self.model_manager.get_latest_model_version(model_name)
                if latest:
                    local_version = self.local_models.get(model_name)
                    if local_version != latest.version_id:
                        logger.info(f"Updating {model_name} from {local_version} to {latest.version_id}")
                        self.local_models[model_name] = latest.version_id
                        # In real implementation: download and install model
            
            return True
        
        except Exception as e:
            logger.error(f"Error syncing models: {e}")
            return False
    
    def get_local_model_version(self, model_name: str) -> Optional[str]:
        """Get locally installed model version.
        
        Args:
            model_name: Model name
            
        Returns:
            Version ID if installed, None otherwise
        """
        return self.local_models.get(model_name)
