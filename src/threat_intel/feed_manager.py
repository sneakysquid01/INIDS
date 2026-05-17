"""Threat Intelligence feed manager.

Manages multiple TI feed sources with in-memory + optional persistent caching.
Feeds can be loaded from local files (CSV/JSON), STIX/TAXII, or REST APIs.
"""
from __future__ import annotations

import csv
import io
import ipaddress
import json
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# D-04: RFC-1918 private ranges that must never appear as TI indicators.
# Blocking internal addresses based on TI feeds would cause network outages.
_RFC1918_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),    # loopback
    ipaddress.ip_network("169.254.0.0/16"), # link-local
    ipaddress.ip_network("::1/128"),         # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),        # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),       # IPv6 link-local
)


def _is_rfc1918(address: str) -> bool:
    """Return True if address falls in any RFC-1918 / internal range."""
    try:
        ip = ipaddress.ip_address(address.strip())
        return any(ip in net for net in _RFC1918_NETWORKS)
    except ValueError:
        return False


def _reject_rfc1918(value: str, source: str) -> bool:
    """Log and return True if value should be rejected (it is an internal IP)."""
    if _is_rfc1918(value):
        logger.error(
            "TI feed '%s': rejected RFC-1918/internal indicator '%s'. "
            "Internal IPs must not appear in TI feeds — they would cause "
            "internal network disruptions if used for blocking.",
            source, value,
        )
        return True
    return False


@dataclass
class TIIndicator:
    """A single Threat Intelligence indicator (IoC)."""
    indicator_type: str  # "ip", "domain", "hash", "url"
    value: str
    source: str = "unknown"
    severity: str = "medium"
    tags: list[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    ttl_seconds: float = 86400.0  # 24 hours default

    def is_expired(self, now: float | None = None) -> bool:
        now = now or time.time()
        if self.last_seen == 0:
            return False
        return (now - self.last_seen) > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "indicator_type": self.indicator_type,
            "value": self.value,
            "source": self.source,
            "severity": self.severity,
            "tags": self.tags,
        }


class ThreatIntelCache:
    """In-memory cache of TI indicators, keyed by (type, value)."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], TIIndicator] = {}
        self._lock = Lock()

    def upsert(self, indicator: TIIndicator) -> None:
        key = (indicator.indicator_type, indicator.value.strip().lower())
        with self._lock:
            self._store[key] = indicator

    def lookup(self, indicator_type: str, value: str) -> TIIndicator | None:
        key = (indicator_type, value.strip().lower())
        with self._lock:
            return self._store.get(key)

    def lookup_ip(self, ip: str) -> TIIndicator | None:
        return self.lookup("ip", ip)

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def purge_expired(self) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            keys_to_remove = [k for k, v in self._store.items() if v.is_expired(now)]
            for k in keys_to_remove:
                del self._store[k]
                removed += 1
        return removed

    def all_indicators(self, indicator_type: str | None = None) -> list[TIIndicator]:
        with self._lock:
            items = list(self._store.values())
        if indicator_type:
            items = [i for i in items if i.indicator_type == indicator_type]
        return items


class ThreatIntelManager:
    """Manages multiple TI feed sources and a unified cache.

    Usage::

        ti = ThreatIntelManager()
        ti.load_csv_feed(path_or_string, source="abuse-ch")
        ti.load_json_feed(path_or_string, source="alienvault")
        indicator = ti.lookup_ip("1.2.3.4")
    """

    def __init__(self, cache: ThreatIntelCache | None = None) -> None:
        self.cache = cache or ThreatIntelCache()
        self._feed_metadata: list[dict[str, Any]] = []
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Feed loading
    # ------------------------------------------------------------------

    def load_csv_feed(
        self,
        data: str,
        *,
        source: str = "csv",
        indicator_type: str = "ip",
        value_column: str = "indicator",
        severity: str = "medium",
        tags: list[str] | None = None,
    ) -> int:
        """Load indicators from CSV text. Returns count of indicators loaded."""
        count = 0
        now = time.time()
        reader = csv.DictReader(io.StringIO(data))
        for row in reader:
            value = row.get(value_column, "").strip()
            if not value:
                continue
            # D-04: reject RFC-1918/internal IP indicators
            if indicator_type == "ip" and _reject_rfc1918(value, source):
                continue
            ind = TIIndicator(
                indicator_type=indicator_type,
                value=value,
                source=source,
                severity=row.get("severity", severity),
                tags=tags or [],
                first_seen=now,
                last_seen=now,
            )
            self.cache.upsert(ind)
            count += 1

        self._record_feed(source, count)
        logger.info("Loaded %d indicators from CSV feed '%s'", count, source)
        return count

    def load_json_feed(
        self,
        data: str,
        *,
        source: str = "json",
        indicator_type: str = "ip",
    ) -> int:
        """Load indicators from a JSON array of objects."""
        count = 0
        now = time.time()
        items = json.loads(data)
        if not isinstance(items, list):
            items = [items]
        for item in items:
            if not isinstance(item, dict):
                continue
            value = item.get("value", item.get("indicator", item.get("ip", ""))).strip()
            if not value:
                continue
            item_type = item.get("type", indicator_type)
            # D-04: reject RFC-1918/internal IP indicators
            if item_type == "ip" and _reject_rfc1918(value, source):
                continue
            ind = TIIndicator(
                indicator_type=item_type,
                value=value,
                source=source,
                severity=item.get("severity", "medium"),
                tags=item.get("tags", []),
                first_seen=now,
                last_seen=now,
            )
            self.cache.upsert(ind)
            count += 1

        self._record_feed(source, count)
        logger.info("Loaded %d indicators from JSON feed '%s'", count, source)
        return count

    def add_indicator(self, indicator: TIIndicator) -> None:
        self.cache.upsert(indicator)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def lookup_ip(self, ip: str) -> TIIndicator | None:
        return self.cache.lookup_ip(ip)

    def lookup(self, indicator_type: str, value: str) -> TIIndicator | None:
        return self.cache.lookup(indicator_type, value)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def feed_summary(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._feed_metadata)

    def stats(self) -> dict[str, Any]:
        return {
            "total_indicators": self.cache.size(),
            "feeds_loaded": len(self._feed_metadata),
        }

    def _record_feed(self, source: str, count: int) -> None:
        with self._lock:
            self._feed_metadata.append({
                "source": source,
                "indicators_loaded": count,
                "loaded_at": time.time(),
            })
            if len(self._feed_metadata) > 1000:
                self._feed_metadata = self._feed_metadata[-500:]
