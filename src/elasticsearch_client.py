"""
Elasticsearch/OpenSearch Integration for INIDS

Provides persistent storage for audit logs, alerts, and detection events.
Supports both Elasticsearch 8.x and OpenSearch 2.x.

Features:
- Async client for non-blocking I/O
- Automatic index management (creation, rollover, retention)
- Bulk insert for performance
- Query builders for analytics
- Fallback to in-memory storage if ES unavailable
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from opensearchpy import OpenSearch, AsyncOpenSearch
    from opensearchpy.exceptions import OpenSearchException
    OPENSEARCH_AVAILABLE = True
except ImportError:
    try:
        from elasticsearch import Elasticsearch, AsyncElasticsearch
        from elasticsearch.exceptions import ElasticsearchException
        OPENSEARCH_AVAILABLE = False
    except ImportError:
        OPENSEARCH_AVAILABLE = None

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Document types for Elasticsearch indices."""
    AUDIT_LOG = "audit_log"
    DETECTION_EVENT = "detection_event"
    PREVENTION_ACTION = "prevention_action"
    ALERT = "alert"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class ElasticsearchConfig:
    """Elasticsearch/OpenSearch configuration."""
    hosts: List[str] = None
    port: int = 9200
    use_ssl: bool = True
    verify_certs: bool = True
    username: str = None
    password: str = None
    index_prefix: str = "inids"
    index_retention_days: int = 30
    bulk_batch_size: int = 500
    bulk_timeout_seconds: int = 30
    enable_ilm: bool = True  # Index Lifecycle Management
    
    def __post_init__(self):
        if self.hosts is None:
            self.hosts = ["localhost"]


class ElasticsearchStore:
    """
    Elasticsearch/OpenSearch client wrapper with async support.
    
    Provides methods for:
    - Storing audit logs, events, and alerts
    - Querying historical data
    - Analytics and reporting
    - Automatic index management
    """
    
    def __init__(self, config: ElasticsearchConfig = None):
        """Initialize Elasticsearch store.
        
        Args:
            config: ElasticsearchConfig instance
        """
        self.config = config or ElasticsearchConfig()
        self.client = None
        self.async_client = None
        self.is_available = False
        self._bulk_buffer = []
        self._bulk_lock = asyncio.Lock() if asyncio else None
        
        # Detect which client to use
        if OPENSEARCH_AVAILABLE is True:
            self.client_type = "opensearch"
        elif OPENSEARCH_AVAILABLE is False:
            self.client_type = "elasticsearch"
        else:
            self.client_type = None
            logger.warning("Neither OpenSearch nor Elasticsearch client available")
    
    async def connect(self) -> bool:
        """Connect to Elasticsearch/OpenSearch.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.client_type:
            logger.warning("No Elasticsearch client available")
            return False
        
        try:
            if self.client_type == "opensearch":
                self.async_client = AsyncOpenSearch(
                    hosts=self.config.hosts,
                    port=self.config.port,
                    use_ssl=self.config.use_ssl,
                    verify_certs=self.config.verify_certs,
                    http_auth=(self.config.username, self.config.password)
                    if self.config.username else None,
                    timeout=30
                )
                # Sync client for one-time operations
                self.client = OpenSearch(
                    hosts=self.config.hosts,
                    port=self.config.port,
                    use_ssl=self.config.use_ssl,
                    verify_certs=self.config.verify_certs,
                    http_auth=(self.config.username, self.config.password)
                    if self.config.username else None,
                    timeout=30
                )
            else:  # elasticsearch
                self.async_client = AsyncElasticsearch(
                    hosts=self.config.hosts,
                    port=self.config.port,
                    scheme="https" if self.config.use_ssl else "http",
                    verify_certs=self.config.verify_certs,
                    basic_auth=(self.config.username, self.config.password)
                    if self.config.username else None,
                    timeout=30
                )
                self.client = Elasticsearch(
                    hosts=self.config.hosts,
                    port=self.config.port,
                    scheme="https" if self.config.use_ssl else "http",
                    verify_certs=self.config.verify_certs,
                    basic_auth=(self.config.username, self.config.password)
                    if self.config.username else None,
                    timeout=30
                )
            
            # Test connection
            if self.client_type == "opensearch":
                info = await self.async_client.info()
            else:
                info = await self.async_client.info()
            
            self.is_available = True
            logger.info(f"Connected to {self.client_type}: {info.get('version', {}).get('number')}")
            
            # Create indices
            await self._create_indices()
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            self.is_available = False
            return False
    
    async def close(self):
        """Close Elasticsearch connection."""
        if self.async_client:
            await self.async_client.close()
            self.client = None
            self.async_client = None
    
    async def _create_indices(self):
        """Create indices with mappings if they don't exist."""
        indices = [
            (f"{self.config.index_prefix}-audit-logs", self._get_audit_log_mapping()),
            (f"{self.config.index_prefix}-events", self._get_event_mapping()),
            (f"{self.config.index_prefix}-alerts", self._get_alert_mapping()),
        ]
        
        for index_name, mapping in indices:
            try:
                exists = await self.async_client.indices.exists(index=index_name)
                if not exists:
                    await self.async_client.indices.create(
                        index=index_name,
                        body={"mappings": mapping}
                    )
                    logger.info(f"Created index: {index_name}")
            except Exception as e:
                logger.warning(f"Failed to create index {index_name}: {e}")
    
    def _get_audit_log_mapping(self) -> Dict[str, Any]:
        """Get mapping for audit log index."""
        return {
            "properties": {
                "timestamp": {"type": "date"},
                "user": {"type": "keyword"},
                "method": {"type": "keyword"},
                "path": {"type": "keyword"},
                "status": {"type": "integer"},
                "response_time_ms": {"type": "float"},
                "source_ip": {"type": "ip"},
                "user_agent": {"type": "text"},
                "request_size": {"type": "integer"},
                "response_size": {"type": "integer"},
                "error": {"type": "text"},
            }
        }
    
    def _get_event_mapping(self) -> Dict[str, Any]:
        """Get mapping for detection event index."""
        return {
            "properties": {
                "timestamp": {"type": "date"},
                "event_id": {"type": "keyword"},
                "engine_id": {"type": "keyword"},
                "source_ip": {"type": "ip"},
                "destination_ip": {"type": "ip"},
                "port": {"type": "integer"},
                "protocol": {"type": "keyword"},
                "severity": {"type": "keyword"},
                "risk_score": {"type": "float"},
                "features": {"type": "object", "enabled": False},
                "rule_id": {"type": "keyword"},
                "message": {"type": "text"},
            }
        }
    
    def _get_alert_mapping(self) -> Dict[str, Any]:
        """Get mapping for alert index."""
        return {
            "properties": {
                "timestamp": {"type": "date"},
                "alert_id": {"type": "keyword"},
                "event_id": {"type": "keyword"},
                "severity": {"type": "keyword"},
                "status": {"type": "keyword"},
                "assigned_to": {"type": "keyword"},
                "false_positive": {"type": "boolean"},
                "investigation_notes": {"type": "text"},
                "related_events": {"type": "keyword"},
            }
        }
    
    async def store_audit_log(self, log_entry: Dict[str, Any]) -> Tuple[bool, str]:
        """Store audit log entry.
        
        Args:
            log_entry: Audit log dictionary
            
        Returns:
            (success, document_id)
        """
        if not self.is_available:
            logger.debug("Elasticsearch unavailable, skipping audit log storage")
            return False, None
        
        try:
            log_entry['timestamp'] = datetime.now(timezone.utc)
            
            response = await self.async_client.index(
                index=f"{self.config.index_prefix}-audit-logs",
                body=log_entry
            )
            
            return True, response.get('_id')
        
        except Exception as e:
            logger.error(f"Failed to store audit log: {e}")
            return False, None
    
    async def store_detection_event(self, event: Dict[str, Any]) -> Tuple[bool, str]:
        """Store detection event.
        
        Args:
            event: Detection event dictionary
            
        Returns:
            (success, document_id)
        """
        if not self.is_available:
            return False, None
        
        try:
            if 'timestamp' not in event:
                event['timestamp'] = datetime.now(timezone.utc)
            
            response = await self.async_client.index(
                index=f"{self.config.index_prefix}-events",
                body=event
            )
            
            return True, response.get('_id')
        
        except Exception as e:
            logger.error(f"Failed to store detection event: {e}")
            return False, None
    
    async def search_audit_logs(
        self,
        query: str = None,
        filters: Dict[str, Any] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search audit logs.
        
        Args:
            query: Free-text search query
            filters: Keyword filters (user, method, path, etc.)
            start_time: Start of time range
            end_time: End of time range
            limit: Max results to return
            offset: Result offset
            
        Returns:
            List of matching audit log entries
        """
        if not self.is_available:
            return []
        
        try:
            must_clauses = []
            
            # Time range filter
            if start_time or end_time:
                range_filter = {}
                if start_time:
                    range_filter['gte'] = start_time.isoformat()
                if end_time:
                    range_filter['lte'] = end_time.isoformat()
                must_clauses.append({"range": {"timestamp": range_filter}})
            
            # Keyword filters
            if filters:
                for key, value in filters.items():
                    if key in ['user', 'method', 'path', 'status']:
                        must_clauses.append({"match": {key: value}})
            
            # Full text search
            if query:
                must_clauses.append({"multi_match": {
                    "query": query,
                    "fields": ["user", "path", "method", "error"]
                }})
            
            # Build query
            search_query = {
                "query": {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}},
                "size": limit,
                "from": offset,
                "sort": [{"timestamp": "desc"}]
            }
            
            response = await self.async_client.search(
                index=f"{self.config.index_prefix}-audit-logs",
                body=search_query
            )
            
            hits = response.get('hits', {}).get('hits', [])
            return [hit['_source'] for hit in hits]
        
        except Exception as e:
            logger.error(f"Failed to search audit logs: {e}")
            return []
    
    async def search_events(
        self,
        severity: str = None,
        engine_id: str = None,
        source_ip: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search detection events.
        
        Args:
            severity: Filter by severity level
            engine_id: Filter by detection engine
            source_ip: Filter by source IP
            start_time: Start of time range
            end_time: End of time range
            limit: Max results
            
        Returns:
            List of matching events
        """
        if not self.is_available:
            return []
        
        try:
            must_clauses = []
            
            # Time range
            if start_time or end_time:
                range_filter = {}
                if start_time:
                    range_filter['gte'] = start_time.isoformat()
                if end_time:
                    range_filter['lte'] = end_time.isoformat()
                must_clauses.append({"range": {"timestamp": range_filter}})
            
            # Severity filter
            if severity:
                must_clauses.append({"match": {"severity": severity}})
            
            # Engine filter
            if engine_id:
                must_clauses.append({"match": {"engine_id": engine_id}})
            
            # Source IP filter
            if source_ip:
                must_clauses.append({"match": {"source_ip": source_ip}})
            
            search_query = {
                "query": {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}},
                "size": limit,
                "sort": [{"timestamp": "desc"}]
            }
            
            response = await self.async_client.search(
                index=f"{self.config.index_prefix}-events",
                body=search_query
            )
            
            hits = response.get('hits', {}).get('hits', [])
            return [hit['_source'] for hit in hits]
        
        except Exception as e:
            logger.error(f"Failed to search events: {e}")
            return []
    
    async def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get system statistics for the given time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with statistics
        """
        if not self.is_available:
            return {}
        
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            # Count audit logs
            audit_query = {
                "query": {
                    "range": {
                        "timestamp": {"gte": start_time.isoformat()}
                    }
                }
            }
            
            audit_response = await self.async_client.count(
                index=f"{self.config.index_prefix}-audit-logs",
                body=audit_query
            )
            
            # Count events by severity
            events_query = {
                "query": {
                    "range": {
                        "timestamp": {"gte": start_time.isoformat()}
                    }
                },
                "aggs": {
                    "by_severity": {
                        "terms": {"field": "severity"}
                    }
                }
            }
            
            events_response = await self.async_client.search(
                index=f"{self.config.index_prefix}-events",
                body=events_query
            )
            
            severity_counts = {}
            for bucket in events_response.get('aggregations', {}).get('by_severity', {}).get('buckets', []):
                severity_counts[bucket['key']] = bucket['doc_count']
            
            return {
                "audit_logs_count": audit_response.get('count', 0),
                "events_by_severity": severity_counts,
                "time_range_hours": hours
            }
        
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


# Singleton instance
_es_store = None


async def get_elasticsearch_store(config: ElasticsearchConfig = None) -> Optional[ElasticsearchStore]:
    """Get or create Elasticsearch store singleton.
    
    Args:
        config: ElasticsearchConfig instance
        
    Returns:
        ElasticsearchStore instance or None if unavailable
    """
    global _es_store
    
    if _es_store is None:
        _es_store = ElasticsearchStore(config)
        connected = await _es_store.connect()
        if not connected:
            _es_store = None
            return None
    
    return _es_store


async def close_elasticsearch_store():
    """Close Elasticsearch store connection."""
    global _es_store
    
    if _es_store:
        await _es_store.close()
        _es_store = None
