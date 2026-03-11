"""Structured JSON logging for SIEM integration and production observability.

Provides a ``JSONFormatter`` that outputs one JSON object per log line,
compatible with Elasticsearch, Splunk, and Graylog ingest pipelines.
"""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any

from flask import has_request_context, request


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def __init__(self, *, service_name: str = "inids", extra_fields: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.service_name = service_name
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "@timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }

        # Request context (Flask)
        if has_request_context():
            payload["request_id"] = request.headers.get("X-Request-ID", "-")
            payload["endpoint"] = request.path
            payload["method"] = request.method
            payload["source_ip"] = request.headers.get("X-Forwarded-For", request.remote_addr or "-")

        # Extra structured fields attached by callers
        for key in ("risk_score", "action", "engine_id", "verdict", "severity", "trace_id"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val

        # Merge static extra fields
        payload.update(self.extra_fields)

        # Exception info
        if record.exc_info and record.exc_info[1]:
            payload["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_json_logging(level: int = logging.INFO, service_name: str = "inids") -> None:
    """Replace root handlers with a single JSON-formatted handler."""
    root = logging.getLogger()
    if getattr(root, "_inids_json_configured", False):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter(service_name=service_name))

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root._inids_json_configured = True
