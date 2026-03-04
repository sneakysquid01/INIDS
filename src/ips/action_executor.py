from __future__ import annotations

import ipaddress
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.core.event_bus import ActionEvent, AuditEvent, EventBus, PolicyDecisionEvent
from src.firewall_adapters import FirewallAdapter


class ActionExecutor:
    """Executes prevention actions and keeps ops/audit state synchronized."""

    def __init__(
        self,
        *,
        adapter: FirewallAdapter,
        adapter_name: str = "mock",
        ops_store=None,
        event_bus: EventBus | None = None,
    ):
        self.adapter = adapter
        self.adapter_name = adapter_name
        self.ops_store = ops_store
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _normalize_ip(target: str) -> str | None:
        try:
            return str(ipaddress.ip_address(target))
        except ValueError:
            return None

    def execute(self, decision_event: PolicyDecisionEvent, policy) -> ActionEvent | None:
        decision = decision_event.decision
        if decision not in {"block", "rate_limit"}:
            return None

        target = self._normalize_ip(decision_event.risk.detection.source or "")
        if target is None:
            self._emit_audit("action_skipped", f"invalid_target source={decision_event.risk.detection.source}")
            return None

        ttl_seconds = int(decision_event.ttl_seconds or getattr(policy, "block_ttl_seconds", 300))
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat() if ttl_seconds > 0 else None

        dry_run = bool(getattr(policy, "dry_run", True))
        executed = False
        status = "dry_run"
        action_name = "block" if decision == "block" else "rate_limit"

        if not dry_run:
            executed = self.adapter.block(target, ttl_seconds)
            status = "active" if executed else "execution_failed"

        action = ActionEvent(
            decision=decision_event,
            action=action_name,
            target=target,
            reason=decision_event.reason,
            dry_run=dry_run,
            executed=executed,
            status=status,
            adapter=self.adapter_name,
            expires_at=expires_at,
            created_at=now.isoformat(),
        )
        self._persist_action(action)
        self._emit_audit(
            "prevention_action",
            (
                f"decision={decision_event.decision} action={action.action} target={action.target} "
                f"status={action.status} dry_run={action.dry_run} executed={action.executed}"
            ),
        )
        return action

    def cleanup_expired_actions(self, now_iso: str | None = None) -> int:
        if self.ops_store is None:
            return 0
        expired = self.ops_store.list_expired_actions(now_iso=now_iso)
        removed_ids: list[int] = []
        for row in expired:
            action_id = int(row["id"])
            target = str(row["target"])
            should_unblock = bool(row.get("executed")) and not bool(row.get("dry_run"))
            if should_unblock:
                ok = self.adapter.unblock(target)
                if not ok:
                    self.ops_store.update_action_status(action_id, "unblock_failed")
                    self._emit_audit("cleanup_failed", f"action_id={action_id} target={target}")
                    continue
                self._emit_audit("ip_unblock", f"target={target} action_id={action_id}")
            removed_ids.append(action_id)
        if removed_ids:
            self.ops_store.delete_actions(removed_ids)
        return len(removed_ids)

    def reconcile(self, limit: int = 5000) -> dict[str, Any]:
        if self.ops_store is None:
            return {"db_active": 0, "firewall_rules": 0, "missing_in_firewall": 0, "orphan_firewall_rules": 0}
        active = self.ops_store.list_active_blocks(limit=limit)
        db_targets = {
            row["target"]
            for row in active
            if bool(row.get("executed")) and not bool(row.get("dry_run"))
        }
        fw_targets = set(self.adapter.list_rules())
        missing = sorted(db_targets - fw_targets)
        orphan = sorted(fw_targets - db_targets)
        for row in active:
            if row["target"] in missing:
                self.ops_store.update_action_status(int(row["id"]), "desynced")
        if missing:
            self._emit_audit("reconcile_missing_rules", f"missing={','.join(missing)}")
        if orphan:
            self._emit_audit("reconcile_orphan_rules", f"orphan={','.join(orphan)}")
        return {
            "db_active": len(db_targets),
            "firewall_rules": len(fw_targets),
            "missing_in_firewall": len(missing),
            "orphan_firewall_rules": len(orphan),
        }

    def _persist_action(self, action: ActionEvent) -> None:
        if self.ops_store is None:
            return
        self.ops_store.save_action(
            {
                "action": action.action,
                "target": action.target,
                "reason": action.reason,
                "expires_at": action.expires_at,
                "created_at": action.created_at,
                "executed": action.executed,
                "dry_run": action.dry_run,
                "status": action.status,
                "adapter": action.adapter,
            }
        )

    def _emit_audit(self, event_type: str, message: str) -> None:
        event = AuditEvent(event_type=event_type, message=message)
        if self.ops_store is not None:
            self.ops_store.add_audit(
                event_type=event.event_type,
                message=event.message,
                created_at=event.created_at,
            )
        if self.event_bus is not None:
            self.event_bus.publish(event)
        self.logger.info("audit_event type=%s message=%s", event_type, message)

