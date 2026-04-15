"""
Elasticsearch Audit Store Integration

Extends existing ops_store to persist audit logs to Elasticsearch
while maintaining backward compatibility with SQLite fallback.

Provides:
- Async audit log storage
- Historical queries
- Analytics
- Automatic cleanup/retention
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from src.elasticsearch_client import ElasticsearchStore, ElasticsearchConfig

logger = logging.getLogger(__name__)


class ElasticsearchAuditBridge:
    """
    Bridge between existing ops_store (SQLite) and Elasticsearch.
    
    Maintains dual-write pattern:
    - Writes audit logs to SQLite (for compatibility)
    - Async writes to Elasticsearch (for analytics)
    
    Queries can be routed to either store.
    """
    
    def __init__(
        self,
        es_store: ElasticsearchStore,
        ops_store_ref: Any = None,
        async_mode: bool = True
    ):
        """Initialize audit bridge.
        
        Args:
            es_store: ElasticsearchStore instance
            ops_store_ref: Reference to existing ops_store for SQLite fallback
            async_mode: Enable async writes to ES
        """
        self.es_store = es_store
        self.ops_store = ops_store_ref
        self.async_mode = async_mode
        self.executor = ThreadPoolExecutor(max_workers=2) if async_mode else None
    
    def add_audit_log(
        self,
        user: str,
        method: str,
        path: str,
        status: int,
        response_time_ms: float,
        source_ip: str,
        error: str = None,
        request_body_size: int = 0,
        response_body_size: int = 0,
        user_agent: str = ""
    ) -> bool:
        """Add audit log entry (synchronous).
        
        Writes to SQLite immediately, queues Elasticsearch write asynchronously.
        
        Returns:
            True if SQLite write succeeded
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user,
            "method": method,
            "path": path,
            "status": status,
            "response_time_ms": response_time_ms,
            "source_ip": source_ip,
            "user_agent": user_agent,
            "request_size": request_body_size,
            "response_size": response_body_size,
            "error": error
        }
        
        # Write to SQLite (primary store)
        sqlite_success = True
        if self.ops_store:
            try:
                self.ops_store.add_audit(
                    user=user,
                    action=f"{method} {path}",
                    details={
                        "status": status,
                        "response_time_ms": response_time_ms,
                        "source_ip": source_ip,
                        "error": error
                    }
                )
            except Exception as e:
                logger.error(f"Failed to write audit log to SQLite: {e}")
                sqlite_success = False
        
        # Queue async write to Elasticsearch
        if self.async_mode and self.es_store and self.es_store.is_available:
            self.executor.submit(
                asyncio.run,
                self.es_store.store_audit_log(log_entry)
            )
        
        return sqlite_success
    
    async def search_audit_logs_async(
        self,
        query: str = None,
        filters: Dict[str, Any] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search audit logs (async, uses Elasticsearch).
        
        Args:
            query: Free-text search query
            filters: Keyword filters
            start_time: Start of time range
            end_time: End of time range
            limit: Max results
            offset: Result offset
            
        Returns:
            List of matching audit logs
        """
        if not self.es_store or not self.es_store.is_available:
            logger.warning("Elasticsearch unavailable for audit log search")
            return []
        
        return await self.es_store.search_audit_logs(
            query=query,
            filters=filters,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset
        )
    
    def search_audit_logs_sync(
        self,
        query: str = None,
        filters: Dict[str, Any] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search audit logs (sync wrapper).
        
        Runs async search in executor to avoid blocking.
        """
        if not self.async_mode or not self.es_store:
            # Fallback to SQLite if Elasticsearch unavailable
            if self.ops_store:
                try:
                    return self.ops_store.list_audit(limit=limit)
                except:
                    return []
            return []
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create new loop
                future = asyncio.ensure_future(
                    self.es_store.search_audit_logs(
                        query=query,
                        filters=filters,
                        start_time=start_time,
                        end_time=end_time,
                        limit=limit,
                        offset=offset
                    )
                )
                return asyncio.get_event_loop().run_until_complete(future)
            else:
                return asyncio.run(
                    self.es_store.search_audit_logs(
                        query=query,
                        filters=filters,
                        start_time=start_time,
                        end_time=end_time,
                        limit=limit,
                        offset=offset
                    )
                )
        except Exception as e:
            logger.error(f"Failed to search audit logs: {e}")
            return []
    
    async def get_audit_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit statistics for time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with statistics
        """
        if not self.es_store or not self.es_store.is_available:
            return {}
        
        try:
            logs = await self.es_store.search_audit_logs(limit=10000)
            
            if not logs:
                return {"message": "No audit logs found"}
            
            # Calculate stats
            by_user = {}
            by_method = {}
            by_status = {}
            
            for log in logs:
                user = log.get('user', 'unknown')
                method = log.get('method', 'unknown')
                status = str(log.get('status', 'unknown'))
                
                by_user[user] = by_user.get(user, 0) + 1
                by_method[method] = by_method.get(method, 0) + 1
                by_status[status] = by_status.get(status, 0) + 1
            
            return {
                "total_logs": len(logs),
                "by_user": by_user,
                "by_method": by_method,
                "by_status": by_status,
                "hours": hours
            }
        
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown audit bridge."""
        if self.executor:
            self.executor.shutdown(wait=True)


# Global bridge instance
_audit_bridge = None


def init_elasticsearch_audit_bridge(
    es_config: ElasticsearchConfig = None,
    ops_store_ref: Any = None
) -> Optional[ElasticsearchAuditBridge]:
    """Initialize Elasticsearch audit bridge.
    
    Args:
        es_config: Elasticsearch configuration
        ops_store_ref: Reference to existing ops_store
        
    Returns:
        ElasticsearchAuditBridge instance or None
    """
    global _audit_bridge
    
    try:
        config = es_config or ElasticsearchConfig(
            hosts=["localhost"],
            port=9200,
            use_ssl=False
        )
        
        es_store = ElasticsearchStore(config)
        
        # Try to connect (non-blocking)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create async task but don't wait
                asyncio.ensure_future(es_store.connect())
            else:
                asyncio.run(es_store.connect())
        except Exception as e:
            logger.warning(f"Could not connect to Elasticsearch: {e}")
            # Continue anyway with degraded functionality
        
        _audit_bridge = ElasticsearchAuditBridge(
            es_store=es_store,
            ops_store_ref=ops_store_ref,
            async_mode=True
        )
        
        return _audit_bridge
    
    except Exception as e:
        logger.error(f"Failed to initialize Elasticsearch audit bridge: {e}")
        return None


def get_elasticsearch_audit_bridge() -> Optional[ElasticsearchAuditBridge]:
    """Get global Elasticsearch audit bridge instance."""
    return _audit_bridge


def shutdown_elasticsearch_audit_bridge():
    """Shutdown Elasticsearch audit bridge."""
    global _audit_bridge
    
    if _audit_bridge:
        _audit_bridge.shutdown()
        _audit_bridge = None
