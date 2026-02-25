from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


@dataclass
class PolicyConfig:
    mode: str = "monitor"  # monitor | auto_block
    block_ttl_seconds: int = 300
    confidence_block_threshold: float = 85.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreventionAction:
    action: str
    target: str
    reason: str
    expires_at: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InMemoryPreventionStore:
    def __init__(self):
        self.actions: list[PreventionAction] = []

    def add_action(self, action: PreventionAction) -> None:
        self.actions.insert(0, action)

    def list_actions(self, limit: int = 50) -> list[PreventionAction]:
        return self.actions[:limit]


class PreventionService:
    def __init__(self, policy: PolicyConfig | None = None, store: InMemoryPreventionStore | None = None):
        self.policy = policy or PolicyConfig()
        self.store = store or InMemoryPreventionStore()

    def set_policy(
        self,
        mode: str | None = None,
        block_ttl_seconds: int | None = None,
        confidence_block_threshold: float | None = None,
    ) -> PolicyConfig:
        if mode is not None:
            normalized_mode = mode.strip().lower()
            if normalized_mode not in {"monitor", "auto_block"}:
                raise ValueError("mode must be either 'monitor' or 'auto_block'")
            self.policy.mode = normalized_mode
        if block_ttl_seconds is not None:
            if block_ttl_seconds <= 0:
                raise ValueError("block_ttl_seconds must be > 0")
            self.policy.block_ttl_seconds = int(block_ttl_seconds)
        if confidence_block_threshold is not None:
            if confidence_block_threshold < 0 or confidence_block_threshold > 100:
                raise ValueError("confidence_block_threshold must be between 0 and 100")
            self.policy.confidence_block_threshold = float(confidence_block_threshold)
        return self.policy

    def evaluate(self, prediction: str, confidence: float, source: str = "unknown") -> PreventionAction | None:
        if self.policy.mode != "auto_block":
            return None
        if prediction != "Attack":
            return None
        if confidence < self.policy.confidence_block_threshold:
            return None

        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=self.policy.block_ttl_seconds)
        action = PreventionAction(
            action="block",
            target=source,
            reason=f"attack_confidence_{confidence}",
            expires_at=expires.isoformat(),
            created_at=now.isoformat(),
        )
        self.store.add_action(action)
        return action
