from __future__ import annotations

import concurrent.futures as cf
import ipaddress
import logging
import time
from datetime import datetime, timedelta, timezone
from threading import Lock
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
        adapter_timeout_s: float = 3.0,
        cb_failure_threshold: int = 3,
        cb_open_duration_s: float = 60.0,
    ):
        self.adapter = adapter
        self.adapter_name = adapter_name
        self.ops_store = ops_store
        self.event_bus = event_bus
        self.adapter_timeout_s = adapter_timeout_s
        self.logger = logging.getLogger(__name__)
        # Circuit breaker state
        self._cb_failure_count: int = 0
        self._cb_open_until: float = 0.0
        self._cb_failure_threshold: int = cb_failure_threshold
        self._cb_open_duration_s: float = cb_open_duration_s
        self._cb_lock = Lock()

    def _call_adapter_with_timeout(self, fn, *args) -> tuple[bool, str]:
        """
        Execute adapter call in isolated thread with hard timeout.
        Returns (success, status_string).
        Never raises — failure is returned as status.
        """
        with cf.ThreadPoolExecutor(max_workers=1) as _exec:
            future = _exec.submit(fn, *args)
            try:
                result = future.result(timeout=self.adapter_timeout_s)
                # Handle the result tuple returned by adapter methods
                if isinstance(result, tuple):
                    return result[0], str(result[1])
                return bool(result), "success"
            except cf.TimeoutError:
                self.logger.error(
                    "adapter_timeout target=%s timeout_s=%s",
                    args[0] if args else "unknown",
                    self.adapter_timeout_s
                )
                return False, "adapter_timeout"
            except Exception as exc:
                self.logger.exception("adapter_call_failed target=%s", args[0] if args else "unknown")
                return False, f"adapter_exception:{type(exc).__name__}"

    def _circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        with self._cb_lock:
            if self._cb_open_until == 0.0:
                return False
            if time.time() > self._cb_open_until:
                self._cb_open_until = 0.0
                self._cb_failure_count = 0
                self.logger.info("circuit_breaker_closed adapter=%s", self.adapter_name)
                return False
            return True

    def _record_adapter_result(self, success: bool) -> None:
        """Record adapter result and update circuit breaker state."""
        with self._cb_lock:
            if success:
                self._cb_failure_count = 0
            else:
                self._cb_failure_count += 1
                if self._cb_failure_count >= self._cb_failure_threshold:
                    self._cb_open_until = time.time() + self._cb_open_duration_s
                    self.logger.error(
                        "circuit_breaker_open adapter=%s for %ss",
                        self.adapter_name, self._cb_open_duration_s
                    )

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
            ok, status = self._call_adapter_with_timeout(self.adapter.block, target, ttl)
            self._record_adapter_result(ok)
            return ok, "blocked" if ok else status
        except Exception:
            self.logger.exception("block_ip failed target=%s", target)
            self._record_adapter_result(False)
            return False, "block_exception"

    def unblock_ip(self, ip: str) -> tuple[bool, str]:
        target = self._normalize_ip(ip)
        if target is None:
            return False, "invalid_ip"
        try:
            ok, status = self._call_adapter_with_timeout(self.adapter.unblock, target)
            self._record_adapter_result(ok)
            return ok, "unblocked" if ok else status
        except Exception:
            self.logger.exception("unblock_ip failed target=%s", target)
            self._record_adapter_result(False)
            return False, "unblock_exception"

    def rate_limit(self, ip: str, ttl: int = 60) -> tuple[bool, str]:
        # In firewall adapters without native shaping, we enforce short-lived block windows.
        ok, status = self.block_ip(ip, max(30, int(ttl)))
        return ok, "rate_limited" if ok else status

    def execute(self, decision_event: PolicyDecisionEvent, policy) -> ActionEvent | None:
        decision = str(decision_event.decision).strip().upper()
        if decision not in {"BLOCK", "TEMP_BLOCK", "RATE_LIMIT", "PENDING_BLOCK"}:
            return None

        target = self._normalize_ip(decision_event.risk.detection.source or "")
        if target is None:
            self._emit_audit("action_skipped", f"invalid_target source={decision_event.risk.detection.source}")
            return None

        # Check circuit breaker before processing action
        if self._circuit_open():
            now = datetime.now(timezone.utc)
            ttl_seconds = int(decision_event.ttl_seconds or getattr(policy, "block_ttl_seconds", 300))
            expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat() if ttl_seconds > 0 else None
            action_id = f"act_{uuid.uuid4().hex[:16]}"
            action = ActionEvent(
                decision=decision_event,
                action="block",
                target=target,
                reason=f"{decision_event.reason} (circuit_open)",
                dry_run=False,
                executed=False,
                status="CIRCUIT_OPEN",
                adapter=self.adapter_name,
                expires_at=expires_at,
                created_at=now.isoformat(),
            )
            self._persist_action(action, action_id=action_id, executed_at=None)
            self._emit_audit("circuit_breaker_fast_fail", f"target={target} decision={decision}")
            return action

        ttl_seconds = int(decision_event.ttl_seconds or getattr(policy, "block_ttl_seconds", 300))
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat() if ttl_seconds > 0 else None

        # PENDING_BLOCK: save for operator approval without executing.
        if decision == "PENDING_BLOCK":
            action_id = f"act_{uuid.uuid4().hex[:16]}"
            action = ActionEvent(
                decision=decision_event,
                action="block",
                target=target,
                reason=decision_event.reason,
                dry_run=False,
                executed=False,
                status="PENDING_APPROVAL",
                adapter=self.adapter_name,
                expires_at=expires_at,
                created_at=now.isoformat(),
            )
            self._persist_action(action, action_id=action_id, executed_at=None)
            self._emit_audit(
                "pending_block",
                f"target={target} reason={decision_event.reason} action_id={action_id}",
            )
            return action

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

    def approve_pending_block(self, action_id: str, policy, *, ttl_seconds: int | None = None) -> dict[str, Any]:
        """Approve a PENDING_BLOCK action and execute it.

        Returns a dict with ``ok``, ``action_id``, ``target``, and ``status``.
        """
        if self.ops_store is None:
            return {"ok": False, "error": "no_ops_store"}

        rows = self._fetchall_pending(action_id)
        if not rows:
            return {"ok": False, "error": "not_found"}

        row = rows[0]
        target = str(row.get("target") or row.get("ip") or "")
        ip = self._normalize_ip(target)
        if ip is None:
            return {"ok": False, "error": "invalid_ip"}

        ttl = int(ttl_seconds or row.get("ttl_seconds") or getattr(policy, "block_ttl_seconds", 300))
        dry_run = bool(getattr(policy, "dry_run", False))
        now = datetime.now(timezone.utc)

        if dry_run:
            self.ops_store.update_action_status(str(row["action_id"]), "DRY_RUN", executed_at=now.isoformat())
            self._emit_audit("block_approved_dry_run", f"action_id={action_id} target={ip}")
            return {"ok": True, "action_id": action_id, "target": ip, "status": "DRY_RUN"}

        ok, status = self.block_ip(ip, ttl)
        new_status = "ACTIVE" if ok else status.upper()
        self.ops_store.update_action_status(str(row["action_id"]), new_status, executed_at=now.isoformat())
        self._emit_audit("block_approved", f"action_id={action_id} target={ip} status={new_status} ok={ok}")
        return {"ok": ok, "action_id": action_id, "target": ip, "status": new_status}

    def _fetchall_pending(self, action_id: str) -> list[dict[str, Any]]:
        """Fetch pending approval actions by action_id string."""
        if self.ops_store is None:
            return []
        return self.ops_store._fetchall(
            "SELECT * FROM actions WHERE action_id = :action_id AND lower(COALESCE(status, '')) = 'pending_approval'",
            {"action_id": str(action_id)},
        )

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
