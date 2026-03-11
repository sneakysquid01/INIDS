"""Persistent IP / CIDR allowlist.

IPs or networks on the allowlist are exempt from all prevention actions.
The allowlist is stored in the OpsStore so it survives process restarts.
"""
from __future__ import annotations

import ipaddress
import logging
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class Allowlist:
    """Thread-safe allowlist backed by in-memory cache + OpsStore persistence."""

    def __init__(self, ops_store: Any | None = None) -> None:
        self._entries: set[str] = set()  # normalized IP strings or CIDR notations
        self._networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        self._lock = Lock()
        self._ops_store = ops_store
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, entry: str, *, reason: str = "") -> bool:
        """Add an IP or CIDR to the allowlist. Returns True if newly added."""
        normalized = self._normalize(entry)
        if normalized is None:
            return False
        with self._lock:
            if normalized in self._entries:
                return False
            self._entries.add(normalized)
            self._rebuild_networks()
        self._persist_add(normalized, reason)
        logger.info("Allowlist: added %s (reason=%s)", normalized, reason)
        return True

    def remove(self, entry: str) -> bool:
        """Remove an entry from the allowlist."""
        normalized = self._normalize(entry)
        if normalized is None:
            return False
        with self._lock:
            if normalized not in self._entries:
                return False
            self._entries.discard(normalized)
            self._rebuild_networks()
        self._persist_remove(normalized)
        logger.info("Allowlist: removed %s", normalized)
        return True

    def contains(self, ip: str) -> bool:
        """Check if an IP is covered by the allowlist (exact match or CIDR)."""
        try:
            addr = ipaddress.ip_address(ip.strip())
        except ValueError:
            return False
        with self._lock:
            if str(addr) in self._entries:
                return True
            for net in self._networks:
                if addr in net:
                    return True
        return False

    def list_entries(self) -> list[str]:
        with self._lock:
            return sorted(self._entries)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(entry: str) -> str | None:
        entry = entry.strip()
        if "/" in entry:
            try:
                net = ipaddress.ip_network(entry, strict=False)
                return str(net)
            except ValueError:
                return None
        try:
            return str(ipaddress.ip_address(entry))
        except ValueError:
            return None

    def _rebuild_networks(self) -> None:
        """Rebuild list of parsed network objects from entries (call under lock)."""
        nets = []
        for e in self._entries:
            if "/" in e:
                try:
                    nets.append(ipaddress.ip_network(e, strict=False))
                except ValueError:
                    pass
        self._networks = nets

    def _load(self) -> None:
        if self._ops_store is None:
            return
        try:
            rows = self._ops_store.list_allowlist()
            with self._lock:
                for row in rows:
                    entry = row if isinstance(row, str) else row.get("entry", "")
                    n = self._normalize(entry)
                    if n:
                        self._entries.add(n)
                self._rebuild_networks()
            logger.info("Allowlist: loaded %d entries from store", len(self._entries))
        except Exception:
            logger.debug("Allowlist: no persistent entries loaded (store may not support allowlist yet)")

    def _persist_add(self, entry: str, reason: str) -> None:
        if self._ops_store is None:
            return
        try:
            self._ops_store.add_allowlist_entry(entry, reason=reason)
        except Exception:
            logger.debug("Allowlist persistence not available for add")

    def _persist_remove(self, entry: str) -> None:
        if self._ops_store is None:
            return
        try:
            self._ops_store.remove_allowlist_entry(entry)
        except Exception:
            logger.debug("Allowlist persistence not available for remove")
