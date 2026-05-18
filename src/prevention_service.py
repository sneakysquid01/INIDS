from __future__ import annotations

import dataclasses
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from src.firewall_adapters import FirewallAdapter, MockFirewallAdapter


def _read_dry_run_from_env() -> bool:
    """Read dry_run from INIDS_DRY_RUN env var.

    D-04: dry_run must require explicit INIDS_DRY_RUN configuration.
    Fail-safe default: if INIDS_DRY_RUN is not set or is unrecognised,
    stay in dry-run mode (True). Only "false", "0", "no", "off" disable it.
    """
    raw = os.environ.get("INIDS_DRY_RUN", "").strip().lower()
    if raw in {"false", "0", "no", "off"}:
        return False
    return True


@dataclass(frozen=True)
class PolicyConfig:
    mode: str = "monitor"  # monitor | auto_block
    block_ttl_seconds: int = 300
    confidence_block_threshold: float = 85.0
    risk_alert_threshold: float = 0.4
    risk_rate_limit_threshold: float = 0.6
    risk_temp_block_threshold: float = 0.75
    risk_block_threshold: float = 0.85
    dry_run: bool = field(default_factory=_read_dry_run_from_env)
    block_requires_approval: bool = False
    risk_weight_confidence: float = 0.5
    risk_weight_severity: float = 0.3
    risk_weight_frequency: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicyConfigManager:
    """Thread-safe manager for a frozen PolicyConfig snapshot.

    B-03: readers always get a consistent frozen config; writers atomically
    replace the whole snapshot via dataclasses.replace() under a lock.
    """

    def __init__(self, initial: PolicyConfig | None = None) -> None:
        self._config: PolicyConfig = initial or PolicyConfig()
        self._lock = threading.RLock()

    def get(self) -> PolicyConfig:
        """Return the current config snapshot (lock-free read)."""
        return self._config

    def update(self, **kwargs: Any) -> PolicyConfig:
        """Replace the config with a new frozen snapshot containing updated fields."""
        with self._lock:
            self._config = dataclasses.replace(self._config, **kwargs)
            return self._config


@dataclass
class PreventionAction:
    action: str
    target: str
    reason: str
    expires_at: str | None
    created_at: str
    dry_run: bool
    executed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PreventionService:
    def __init__(
        self,
        policy: PolicyConfig | None = None,
        adapter: FirewallAdapter | None = None,
    ):
        self.config_manager = PolicyConfigManager(policy)
        self.adapter = adapter or MockFirewallAdapter()

    @property
    def policy(self) -> PolicyConfig:
        """Read-only snapshot of the current policy (B-03 compat shim)."""
        return self.config_manager.get()

    def set_policy(
        self,
        mode: str | None = None,
        block_ttl_seconds: int | None = None,
        confidence_block_threshold: float | None = None,
        risk_alert_threshold: float | None = None,
        risk_rate_limit_threshold: float | None = None,
        risk_temp_block_threshold: float | None = None,
        risk_block_threshold: float | None = None,
        dry_run: bool | None = None,
        block_requires_approval: bool | None = None,
        risk_weight_confidence: float | None = None,
        risk_weight_severity: float | None = None,
        risk_weight_frequency: float | None = None,
    ) -> PolicyConfig:
        updates: dict[str, Any] = {}
        if mode is not None:
            normalized_mode = mode.strip().lower()
            if normalized_mode not in {"monitor", "auto_block"}:
                raise ValueError("mode must be either 'monitor' or 'auto_block'")
            updates["mode"] = normalized_mode
        if block_ttl_seconds is not None:
            if block_ttl_seconds <= 0:
                raise ValueError("block_ttl_seconds must be > 0")
            updates["block_ttl_seconds"] = int(block_ttl_seconds)
        if confidence_block_threshold is not None:
            if confidence_block_threshold < 0 or confidence_block_threshold > 100:
                raise ValueError("confidence_block_threshold must be between 0 and 100")
            updates["confidence_block_threshold"] = float(confidence_block_threshold)
        for attr, val in (
            ("risk_alert_threshold", risk_alert_threshold),
            ("risk_rate_limit_threshold", risk_rate_limit_threshold),
            ("risk_temp_block_threshold", risk_temp_block_threshold),
            ("risk_block_threshold", risk_block_threshold),
        ):
            if val is not None:
                fval = float(val)
                if fval < 0 or fval > 1:
                    raise ValueError(f"{attr} must be between 0 and 1")
                updates[attr] = fval
        if dry_run is not None:
            updates["dry_run"] = bool(dry_run)
        if block_requires_approval is not None:
            updates["block_requires_approval"] = bool(block_requires_approval)
        for attr, val in (
            ("risk_weight_confidence", risk_weight_confidence),
            ("risk_weight_severity", risk_weight_severity),
            ("risk_weight_frequency", risk_weight_frequency),
        ):
            if val is not None:
                fval = float(val)
                if fval < 0 or fval > 1:
                    raise ValueError(f"{attr} must be between 0 and 1")
                updates[attr] = fval
        return self.config_manager.update(**updates) if updates else self.config_manager.get()

    def evaluate(self, prediction: str, confidence: float, source: str = "unknown") -> PreventionAction | None:
        cfg = self.config_manager.get()
        if cfg.mode != "auto_block":
            return None
        if prediction != "Attack":
            return None
        if confidence < cfg.confidence_block_threshold:
            return None

        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=cfg.block_ttl_seconds)

        executed = False
        if not cfg.dry_run:
            executed = self.adapter.block(source, cfg.block_ttl_seconds)

        action = PreventionAction(
            action="block",
            target=source,
            reason=f"attack_confidence_{confidence}",
            expires_at=expires.isoformat(),
            created_at=now.isoformat(),
            dry_run=cfg.dry_run,
            executed=executed,
        )
        return action
