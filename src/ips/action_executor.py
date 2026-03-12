from __future__ import annotations

import ipaddress
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
import uuid

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

    def block_ip(self, ip: str, ttl: int) -> tuple[bool, str]:
        target = self._normalize_ip(ip)
        if target is None:
            return False, "invalid_ip"
        try:
            ok = self.adapter.block(target, ttl)
            return bool(ok), "blocked" if ok else "block_failed"
        except Exception:
            self.logger.exception("block_ip failed target=%s", target)
            return False, "block_exception"

    def unblock_ip(self, ip: str) -> tuple[bool, str]:
        target = self._normalize_ip(ip)
        if target is None:
            return False, "invalid_ip"
        try:
            ok = self.adapter.unblock(target)
            return bool(ok), "unblocked" if ok else "unblock_failed"
        except Exception:
            self.logger.exception("unblock_ip failed target=%s", target)
            return False, "unblock_exception"

    def rate_limit(self, ip: str, ttl: int = 60) -> tuple[bool, str]:
        # In firewall adapters without native shaping, we enforce short-lived block windows.
        ok, status = self.block_ip(ip, max(30, int(ttl)))
        return ok, "rate_limited" if ok else status

    def execute(self, decision_event: PolicyDecisionEvent, policy) -> ActionEvent | None:
        decision = str(decision_event.decision).strip().upper()
        if decision not in {"BLOCK", "TEMP_BLOCK", "RATE_LIMIT"}:
            return None

        target = self._normalize_ip(decision_event.risk.detection.source or "")
        if target is None:
            self._emit_audit("action_skipped", f"invalid_target source={decision_event.risk.detection.source}")
            return None

        ttl_seconds = int(decision_event.ttl_seconds or getattr(policy, "block_ttl_seconds", 300))
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat() if ttl_seconds > 0 else None

        # Idempotency: skip if this target already has an active enforcement record.
        if self.ops_store is not None and decision in {"BLOCK", "TEMP_BLOCK", "RATE_LIMIT"}:
            if self.ops_store.has_active_block(target):
                self.logger.debug("Idempotency: %s already has active block, skipping duplicate enforcement", target)
                return None

        dry_run = bool(getattr(policy, "dry_run", True))
        executed = False
        status = "DRY_RUN"
        action_name = "block" if decision in {"BLOCK", "TEMP_BLOCK"} else "rate_limit"
        action_id = f"act_{uuid.uuid4().hex[:16]}"
        executed_at = None

        if not dry_run:
            if decision in {"BLOCK", "TEMP_BLOCK"}:
                executed, status = self.block_ip(target, ttl_seconds)
            else:
                executed, status = self.rate_limit(target, ttl_seconds)
            status = "ACTIVE" if executed else status.upper()
            if executed:
                executed_at = now.isoformat()

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
        self._persist_action(action, action_id=action_id, executed_at=executed_at)
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
            self.ops_store.update_action_status(action_id, "EXPIRED")
            should_unblock = bool(row.get("executed")) and not bool(row.get("dry_run"))
            if should_unblock:
                ok, status = self.unblock_ip(target)
                if not ok:
                    self.ops_store.update_action_status(action_id, status.upper())
                    self._emit_audit("cleanup_failed", f"action_id={action_id} target={target}")
                    continue
                self.ops_store.update_action_status(action_id, "UNBLOCKED", executed_at=datetime.now(timezone.utc).isoformat())
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
                self.ops_store.update_action_status(int(row["id"]), "DESYNCED")
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

    def _persist_action(self, action: ActionEvent, *, action_id: str, executed_at: str | None) -> None:
        if self.ops_store is None:
            return
        self.ops_store.save_action(
            {
                "action_id": action_id,
                "action": action.action,
                "action_type": action.action,
                "target": action.target,
                "ip": action.target,
                "reason": action.reason,
                "expires_at": action.expires_at,
                "created_at": action.created_at,
                "executed_at": executed_at,
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
