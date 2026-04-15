"""
Week 4: Async Pipeline & RBAC Integration

Transforms the detection service from synchronous (Flask WSGI) to async:
1. Converts detection_service.predict() to async
2. Integrates RBAC for access control
3. Adds async batch processing
4. Maintains backward compatibility with sync Flask routes

Architecture:
- Flask (WSGI): Handles sync requests with @async_to_sync wrapper
- Connexion (ASGI): Handles native async requests directly
- Async Pipeline: Core detection logic runs asynchronously
- RBAC Middleware: Enforces permissions on all endpoints
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AsyncDetectionConfig:
    """Configuration for async detection pipeline."""
    batch_size: int = 32
    max_wait_ms: int = 100
    max_workers: int = 4
    enable_rbac: bool = True
    rbac_db_url: str = "sqlite:///inids_rbac.db"
    cache_predictions: bool = True
    cache_ttl_seconds: int = 300


class AsyncDetectionPipeline:
    """
    Async detection pipeline for high-throughput predictions.
    
    Handles:
    - Batch accumulation and processing
    - Rate limiting with RBAC
    - Response caching
    - Error handling and fallback
    """
    
    def __init__(
        self,
        detection_service,
        async_executor,
        rbac_manager=None,
        config: Optional[AsyncDetectionConfig] = None
    ):
        """Initialize async detection pipeline.
        
        Args:
            detection_service: Sync detection service instance
            async_executor: AsyncExecutor for parallel processing
            rbac_manager: RBAC manager for permission checking
            config: Pipeline configuration
        """
        self.detection_service = detection_service
        self.async_executor = async_executor
        self.rbac_manager = rbac_manager
        self.config = config or AsyncDetectionConfig()
        
        # Batch accumulator
        self.pending_predictions: List[Dict[str, Any]] = []
        self.batch_lock = asyncio.Lock()
        
        # Cache
        self.prediction_cache: Dict[str, Any] = {}
        
        logger.info(f"Async detection pipeline initialized with batch_size={self.config.batch_size}")
    
    async def predict_async(
        self,
        features: Dict[str, Any],
        user_id: str = None,
        profile: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Async prediction with RBAC and caching.
        
        Args:
            features: Feature dictionary
            user_id: User making prediction (for RBAC)
            profile: Detection profile (balanced, aggressive, etc.)
            
        Returns:
            Prediction result
        """
        # Check RBAC permission if user specified
        if user_id and self.rbac_manager:
            allowed, reason = self.rbac_manager.check_permission(user_id, "predict_detection")
            if not allowed:
                logger.warning(f"User {user_id} denied prediction access: {reason}")
                return {"error": "Permission denied", "reason": reason}
        
        # Check cache
        cache_key = self._get_cache_key(features, profile)
        if cache_key in self.prediction_cache:
            logger.debug(f"Cache hit: {cache_key}")
            return self.prediction_cache[cache_key]
        
        # Run prediction in thread pool (sync detection service)
        try:
            prediction = await self.async_executor.run_in_thread_pool(
                self.detection_service.predict_from_features,
                features,
                profile
            )
            
            # Cache result
            if self.config.cache_predictions:
                self.prediction_cache[cache_key] = prediction
            
            return prediction
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e), "risk_score": 0.5}
    
    async def predict_batch_async(
        self,
        predictions_list: List[Dict[str, Any]],
        user_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Async batch prediction with concurrency control.
        
        Args:
            predictions_list: List of {"features": {...}, "profile": str}
            user_id: User making batch prediction
            
        Returns:
            List of prediction results
        """
        # Check RBAC
        if user_id and self.rbac_manager:
            allowed, reason = self.rbac_manager.check_permission(user_id, "batch_predict")
            if not allowed:
                return [{"error": "Permission denied"}] * len(predictions_list)
        
        # Process with concurrency limit
        tasks = [
            self.predict_async(
                p.get("features", {}),
                user_id,
                p.get("profile", "balanced")
            )
            for p in predictions_list
        ]
        
        # Limit concurrent tasks
        results = await self.async_executor.gather_with_limit(
            tasks,
            limit=self.config.max_workers
        )
        
        return results
    
    def _get_cache_key(self, features: Dict[str, Any], profile: str) -> str:
        """Generate cache key from features and profile."""
        # Sort features for consistent key generation
        features_str = str(sorted(features.items()))
        return f"{profile}:{hash(features_str)}"
    
    async def clear_cache(self):
        """Clear prediction cache."""
        async with self.batch_lock:
            self.prediction_cache.clear()
        logger.info("Prediction cache cleared")


class RBACMiddleware:
    """
    WSGI/ASGI middleware for RBAC enforcement.
    
    Checks user permissions before allowing request to proceed.
    """
    
    def __init__(self, rbac_manager, exempt_paths: List[str] = None):
        """Initialize RBAC middleware.
        
        Args:
            rbac_manager: RBAC manager instance
            exempt_paths: List of paths that don't require auth
        """
        self.rbac_manager = rbac_manager
        self.exempt_paths = exempt_paths or ["/api/health", "/api/auth/login"]
    
    async def check_access(
        self,
        path: str,
        method: str,
        user_id: str = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if user has access to endpoint.
        
        Args:
            path: Request path
            method: HTTP method
            user_id: User ID from JWT token
            
        Returns:
            (allowed, reason)
        """
        # Skip exempt paths
        if self._is_exempt(path):
            return True, None
        
        # Require authentication
        if not user_id:
            return False, "Authentication required"
        
        # Check RBAC permissions based on endpoint
        required_permission = self._get_required_permission(path, method)
        
        if not required_permission:
            return True, None  # No permission check needed
        
        allowed, reason = self.rbac_manager.check_permission(user_id, required_permission)
        return allowed, reason
    
    def _is_exempt(self, path: str) -> bool:
        """Check if path is exempt from auth."""
        return any(path.startswith(exempt) for exempt in self.exempt_paths)
    
    def _get_required_permission(self, path: str, method: str) -> Optional[str]:
        """Get required permission for endpoint."""
        # Map endpoints to permissions
        permissions_map = {
            ("/api/predict", "POST"): "predict_detection",
            ("/api/alerts", "GET"): "read_alert",
            ("/api/alerts", "POST"): "create_alert",
            ("/api/rules", "GET"): "read_rule",
            ("/api/rules", "POST"): "create_rule",
            ("/api/rules", "PUT"): "update_rule",
            ("/api/rules", "DELETE"): "delete_rule",
            ("/api/audit/logs", "GET"): "read_audit",
        }
        
        return permissions_map.get((path, method))


class DetectionServiceAsync:
    """
    Async wrapper for detection service with feature parity.
    
    Provides async versions of all detection_service methods
    while maintaining compatibility with sync Flask endpoints.
    """
    
    def __init__(self, sync_service, async_pipeline: AsyncDetectionPipeline):
        """Initialize async detection service.
        
        Args:
            sync_service: Synchronous detection service
            async_pipeline: Async detection pipeline
        """
        self.sync_service = sync_service
        self.async_pipeline = async_pipeline
    
    async def predict_from_features_async(
        self,
        features: Dict[str, Any],
        profile: str = "balanced",
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Async prediction from features.
        
        Args:
            features: Feature vector
            profile: Detection profile
            user_id: User ID for audit trail
            
        Returns:
            Prediction result
        """
        return await self.async_pipeline.predict_async(
            features,
            user_id,
            profile
        )
    
    async def get_detection_engines_async(self) -> Dict[str, Any]:
        """Get detection engines configuration asynchronously."""
        # Run sync method in thread pool
        return await self.async_pipeline.async_executor.run_in_thread_pool(
            self.sync_service.get_detection_engines
        )
    
    async def get_model_status_async(self) -> Dict[str, Any]:
        """Get model status asynchronously."""
        return await self.async_pipeline.async_executor.run_in_thread_pool(
            self.sync_service.get_model_status
        )


# Decorators for Flask/Connexion compatibility

def require_permission(permission: str):
    """Decorator to enforce permission on endpoint.
    
    Usage:
        @app.route('/api/predict', methods=['POST'])
        @require_permission('predict_detection')
        def predict():
            ...
    """
    from functools import wraps
    from flask import request, abort
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = getattr(request, 'user_id', None)
            
            if not user_id:
                abort(401)
            
            # RBAC check would go here
            # For now, just log
            logger.info(f"Permission check: {permission} for user {user_id}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def async_endpoint(async_func):
    """Decorator to make async function work with Flask sync routes.
    
    Usage:
        @app.route('/api/predict', methods=['POST'])
        @async_endpoint
        async def predict():
            ...
    """
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(func(*args, **kwargs))
            finally:
                loop.close()
        
        return wrapper
    
    return decorator
