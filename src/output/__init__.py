"""
INIDS Output Module

Comprehensive output formatting and delivery system.
Supports multiple backends and aggregation strategies.
"""

from .eve_json import (
    EVEEvent,
    EVEEventBuilder,
    EventType,
    AlertSeverity,
    AlertPayload,
    HTTPPayload,
    DNSPayload,
    TLSPayload,
)

from .output_backends import (
    OutputBackend,
    FileBackend,
    SyslogBackend,
    RedisBackend,
    WebhookBackend,
    OutputAggregator,
    BackendStats,
)

from .flow_aggregator import (
    FlowAggregator,
    AggregationMode,
    OutputPipeline,
    AlertThrottler,
)

__all__ = [
    # EVE JSON
    "EVEEvent",
    "EVEEventBuilder",
    "EventType",
    "AlertSeverity",
    "AlertPayload",
    "HTTPPayload",
    "DNSPayload",
    "TLSPayload",
    # Backends
    "OutputBackend",
    "FileBackend",
    "SyslogBackend",
    "RedisBackend",
    "WebhookBackend",
    "OutputAggregator",
    "BackendStats",
    # Aggregation
    "FlowAggregator",
    "AggregationMode",
    "OutputPipeline",
    "AlertThrottler",
]
