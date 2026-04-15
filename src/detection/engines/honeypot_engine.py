"""Honeypot / Canary Detection Engine — detects access to decoy resources.

Inspiration from WatchAD's honeypot account detection pattern.
Any flow to/from a honeypot IP or port is immediately flagged as CRITICAL
with 100% confidence — honeypots should have zero legitimate traffic.
"""
from __future__ import annotations

import logging
from typing import Any

from src.detection.engine_base import DetectionEngine, EngineResult

logger = logging.getLogger(__name__)


class HoneypotDetectionEngine(DetectionEngine):
    """Detects access to honeypot/canary IPs and ports.

    Configuration:
    - HONEYPOT_IPS: comma-separated list of canary IPs (e.g., "10.1.1.254,10.1.1.253")
    - HONEYPOT_PORTS: comma-separated list of canary ports (e.g., "22222,3389")
    - HONEYPOT_ENABLED: set to "1" or "true" to enable

    Any flow involving a honeypot IP or port generates a CRITICAL alert
    with 100% confidence, as honeypots should never have legitimate traffic.
    """

    def __init__(
        self,
        *,
        engine_id: str = "honeypot",
        honeypot_ips: list[str] | None = None,
        honeypot_ports: list[int] | None = None,
    ) -> None:
        self._engine_id = engine_id
        self._honeypot_ips = set(honeypot_ips or [])
        self._honeypot_ports = set(honeypot_ports or [])
        self._enabled = bool(self._honeypot_ips or self._honeypot_ports)
        logger.info(
            "HoneypotDetectionEngine initialized: "
            "ips=%s, ports=%s, enabled=%s",
            self._honeypot_ips,
            self._honeypot_ports,
            self._enabled,
        )

    # ------------------------------------------------------------------
    # DetectionEngine interface
    # ------------------------------------------------------------------

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def engine_type(self) -> str:
        return "honeypot"

    def is_ready(self) -> bool:
        return self._enabled

    def evaluate(self, features: dict[str, Any]) -> EngineResult:
        """Detect if flow involves honeypot IP or port."""
        if not self._enabled:
            return EngineResult(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                verdict="unknown",
                confidence=0.0,
                severity="low",
                metadata={"reason": "honeypot_engine_disabled"},
            )

        src_ip = features.get("source_ip") or features.get("src_ip") or ""
        dst_ip = features.get("dest_ip") or features.get("dst_ip") or ""
        src_port = features.get("source_port") or features.get("src_port")
        dst_port = features.get("dest_port") or features.get("dst_port")

        # Check if DIP is honeypot
        if dst_ip in self._honeypot_ips:
            return EngineResult(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                verdict="attack",
                confidence=100.0,
                severity="critical",
                attack_type="honeypot_access",
                metadata={
                    "reason": "honeypot_ip_accessed",
                    "honeypot_ip": dst_ip,
                    "source_ip": src_ip,
                },
            )

        # Check if SIP is honeypot (reverse connection from compromised honeypot)
        if src_ip in self._honeypot_ips:
            return EngineResult(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                verdict="attack",
                confidence=100.0,
                severity="critical",
                attack_type="honeypot_outbound",
                metadata={
                    "reason": "honeypot_outbound_connection",
                    "honeypot_ip": src_ip,
                    "destination_ip": dst_ip,
                },
            )

        # Check if destination port is honeypot
        if dst_port and dst_port in self._honeypot_ports:
            return EngineResult(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                verdict="attack",
                confidence=100.0,
                severity="critical",
                attack_type="honeypot_port_accessed",
                metadata={
                    "reason": "honeypot_port_accessed",
                    "honeypot_port": dst_port,
                    "source_ip": src_ip,
                    "destination_ip": dst_ip,
                },
            )

        # Check if source port is honeypot (unusual but possible)
        if src_port and src_port in self._honeypot_ports:
            return EngineResult(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                verdict="attack",
                confidence=100.0,
                severity="critical",
                attack_type="honeypot_port_source",
                metadata={
                    "reason": "honeypot_port_source",
                    "honeypot_port": src_port,
                    "source_ip": src_ip,
                    "destination_ip": dst_ip,
                },
            )

        # No honeypot involvement
        return EngineResult(
            engine_id=self.engine_id,
            engine_type=self.engine_type,
            verdict="normal",
            confidence=0.0,
            severity="low",
            metadata={"reason": "no_honeypot_involvement"},
        )

    # ------------------------------------------------------------------
    # Configuration management
    # ------------------------------------------------------------------

    def update_honeypot_ips(self, ips: list[str]) -> None:
        """Update honeypot IPs at runtime."""
        self._honeypot_ips = set(ips)
        self._enabled = bool(self._honeypot_ips or self._honeypot_ports)
        logger.info("Updated honeypot IPs: %s", self._honeypot_ips)

    def update_honeypot_ports(self, ports: list[int]) -> None:
        """Update honeypot ports at runtime."""
        self._honeypot_ports = set(ports)
        self._enabled = bool(self._honeypot_ips or self._honeypot_ports)
        logger.info("Updated honeypot ports: %s", self._honeypot_ports)

    def get_config(self) -> dict[str, Any]:
        """Get current honeypot configuration."""
        return {
            "honeypot_ips": list(self._honeypot_ips),
            "honeypot_ports": list(self._honeypot_ports),
            "enabled": self._enabled,
        }
