from __future__ import annotations

from dataclasses import asdict, dataclass
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
from threading import Lock
from typing import Any

from src.core.event_bus import DetectionEvent, EventBus
from src.firewall_adapters import FirewallAdapter, MockFirewallAdapter
from src.ips.action_executor import ActionExecutor
from src.ips.policy_engine import PolicyEngine
from src.ips.risk_engine import RiskEngine
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
from datetime import datetime, timedelta, timezone
from typing import Any

from src.firewall_adapters import FirewallAdapter, MockFirewallAdapter
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs


@dataclass
class PolicyConfig:
    mode: str = "monitor"  # monitor | auto_block
    block_ttl_seconds: int = 300
    confidence_block_threshold: float = 85.0
    dry_run: bool = True
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    risk_alert_threshold: float = 0.4
    risk_rate_limit_threshold: float = 0.95
    risk_block_threshold: float = 0.7
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreventionAction:
    action: str
    target: str
    reason: str
    expires_at: str | None
    created_at: str
    dry_run: bool
    executed: bool
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    status: str
    adapter: str
    risk_score: float | None = None
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InMemoryPreventionStore:
    def __init__(self):
        self.actions: list[PreventionAction] = []
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        self._lock = Lock()

    def add_action(self, action: PreventionAction) -> None:
        with self._lock:
            self.actions.insert(0, action)

    def list_actions(self, limit: int = 50) -> list[PreventionAction]:
        with self._lock:
            return list(self.actions[:limit])
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs

    def add_action(self, action: PreventionAction) -> None:
        self.actions.insert(0, action)

    def list_actions(self, limit: int = 50) -> list[PreventionAction]:
        return self.actions[:limit]
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs


class PreventionService:
    def __init__(
        self,
        policy: PolicyConfig | None = None,
        store: InMemoryPreventionStore | None = None,
        adapter: FirewallAdapter | None = None,
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        ops_store=None,
        event_bus: EventBus | None = None,
        risk_engine: RiskEngine | None = None,
        policy_engine: PolicyEngine | None = None,
        action_executor: ActionExecutor | None = None,
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    ):
        self.policy = policy or PolicyConfig()
        self.store = store or InMemoryPreventionStore()
        self.adapter = adapter or MockFirewallAdapter()
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        self.event_bus = event_bus or EventBus()
        self.risk_engine = risk_engine or RiskEngine()
        self.policy_engine = policy_engine or PolicyEngine()
        self.action_executor = action_executor or ActionExecutor(
            adapter=self.adapter,
            adapter_name=self.adapter.__class__.__name__.replace("FirewallAdapter", "").lower() or "mock",
            ops_store=ops_store,
            event_bus=self.event_bus,
        )
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs

    def set_policy(
        self,
        mode: str | None = None,
        block_ttl_seconds: int | None = None,
        confidence_block_threshold: float | None = None,
        dry_run: bool | None = None,
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        risk_alert_threshold: float | None = None,
        risk_rate_limit_threshold: float | None = None,
        risk_block_threshold: float | None = None,
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
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
        if dry_run is not None:
            self.policy.dry_run = bool(dry_run)
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        if risk_alert_threshold is not None:
            val = float(risk_alert_threshold)
            if val < 0 or val > 1:
                raise ValueError("risk_alert_threshold must be between 0 and 1")
            self.policy.risk_alert_threshold = val
        if risk_rate_limit_threshold is not None:
            val = float(risk_rate_limit_threshold)
            if val < 0 or val > 1:
                raise ValueError("risk_rate_limit_threshold must be between 0 and 1")
            self.policy.risk_rate_limit_threshold = val
        if risk_block_threshold is not None:
            val = float(risk_block_threshold)
            if val < 0 or val > 1:
                raise ValueError("risk_block_threshold must be between 0 and 1")
            self.policy.risk_block_threshold = val
        return self.policy

    def evaluate(self, prediction: str, confidence: float, source: str = "unknown") -> PreventionAction | None:
        detection_event = DetectionEvent(
            source=source,
            prediction=prediction,
            confidence=float(confidence),
            severity="high" if str(prediction).lower() == "attack" else "low",
            suspicious=float(confidence) < self.policy.confidence_block_threshold,
            reason="model_prediction",
        )
        self.event_bus.publish(detection_event)

        risk_event = self.risk_engine.calculate(detection_event)
        self.event_bus.publish(risk_event)

        decision_event = self.policy_engine.decide(risk_event, self.policy)
        self.event_bus.publish(decision_event)

        action_event = self.action_executor.execute(decision_event, self.policy)
        if action_event is None:
            return None

        action = PreventionAction(
            action=action_event.action,
            target=action_event.target,
            reason=action_event.reason,
            expires_at=action_event.expires_at,
            created_at=action_event.created_at,
            dry_run=action_event.dry_run,
            executed=action_event.executed,
            status=action_event.status,
            adapter=action_event.adapter,
            risk_score=decision_event.risk.risk_score,
        )
        self.store.add_action(action)
        return action

    def cleanup_expired_actions(self, now_iso: str | None = None) -> int:
        return self.action_executor.cleanup_expired_actions(now_iso=now_iso)

    def reconcile(self) -> dict[str, Any]:
        return self.action_executor.reconcile()
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
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

        executed = False
        if not self.policy.dry_run:
            executed = self.adapter.block(source, self.policy.block_ttl_seconds)

        action = PreventionAction(
            action="block",
            target=source,
            reason=f"attack_confidence_{confidence}",
            expires_at=expires.isoformat(),
            created_at=now.isoformat(),
            dry_run=self.policy.dry_run,
            executed=executed,
        )
        self.store.add_action(action)
        return action
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
