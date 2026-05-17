from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable
import ipaddress
import json
import logging
import subprocess
import urllib.request

logger = logging.getLogger(__name__)


class FirewallAdapter(ABC):
    # Subclasses that can enumerate currently-enforced rules set this to True.
    # Stateless adapters (e.g. webhook) that have no way to query remote state
    # set it to False so reconcile() knows to skip the comparison.
    supports_rule_query: bool = True

    @abstractmethod
    def block(self, ip: str, ttl_seconds: int | None = None) -> bool:
        raise NotImplementedError

    @abstractmethod
    def unblock(self, ip: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_rules(self) -> list[str]:
        raise NotImplementedError


def _validate_target_ip(target: str) -> str:
    ip = ipaddress.ip_address(target)
    return str(ip)


@dataclass
class MockFirewallAdapter(FirewallAdapter):
    """In-memory firewall adapter for local demos/tests."""

    supports_rule_query: bool = True
    blocked_targets: dict[str, int] | None = None

    def __post_init__(self):
        if self.blocked_targets is None:
            self.blocked_targets = {}

    def block(self, ip: str, ttl_seconds: int | None = None) -> bool:
        try:
            target = _validate_target_ip(ip)
            self.blocked_targets[target] = int(ttl_seconds or 0)
            return True
        except Exception:
            return False

    def unblock(self, ip: str) -> bool:
        try:
            target = _validate_target_ip(ip)
            return self.blocked_targets.pop(target, None) is not None
        except Exception:
            return False

    def list_rules(self) -> list[str]:
        return sorted(self.blocked_targets.keys())


@dataclass
class UfwFirewallAdapter(FirewallAdapter):
    """UFW-backed firewall adapter. Requires ufw and appropriate permissions."""

    supports_rule_query: bool = True
    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run

    def _run(self, args: list[str]) -> tuple[bool, str]:
        try:
            result = self.run_cmd(args, capture_output=True, text=True, timeout=5, check=False)
            out = (str(getattr(result, "stdout", "") or "") + str(getattr(result, "stderr", "") or ""))
            return result.returncode == 0, out
        except subprocess.TimeoutExpired:
            logger.error("ufw_subprocess_timeout args=%s", args)
            return False, "timeout"
        except Exception as exc:
            logger.exception("ufw_subprocess_failed args=%s", args)
            return False, f"exception:{type(exc).__name__}"

    def block(self, ip: str, ttl_seconds: int | None = None) -> bool:
        try:
            target = _validate_target_ip(ip)
        except Exception:
            return False
        ok, out = self._run(["ufw", "deny", "from", target])
        # UFW exits non-zero when the rule already exists but the block is in effect
        if not ok:
            lower = out.lower()
            if "skipping" in lower or "already exists" in lower:
                return True
        return ok

    def unblock(self, ip: str) -> bool:
        try:
            target = _validate_target_ip(ip)
        except Exception:
            return False
        ok, _ = self._run(["ufw", "delete", "deny", "from", target])
        return ok

    def list_rules(self) -> list[str]:
        ok, out = self._run(["ufw", "status"])
        if not ok:
            return []
        rules: list[str] = []
        for line in out.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].count(".") == 3:
                try:
                    rules.append(_validate_target_ip(parts[0]))
                except Exception:
                    continue
        return sorted(set(rules))


@dataclass
class NftablesFirewallAdapter(FirewallAdapter):
    """nftables-backed adapter that inserts/deletes source-IP drop rules in inet filter input."""

    supports_rule_query: bool = True
    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run

    def _run(self, args: list[str]) -> tuple[bool, str]:
        try:
            result = self.run_cmd(args, capture_output=True, text=True, timeout=5, check=False)
            return result.returncode == 0, str(getattr(result, "stdout", "") or "")
        except subprocess.TimeoutExpired:
            logger.error("nft_subprocess_timeout args=%s", args)
            return False, "timeout"
        except Exception as exc:
            logger.exception("nft_subprocess_failed args=%s", args)
            return False, f"exception:{type(exc).__name__}"

    def block(self, ip: str, ttl_seconds: int | None = None) -> bool:
        try:
            target = _validate_target_ip(ip)
        except Exception:
            return False
        ok, _ = self._run(["nft", "add", "rule", "inet", "filter", "input", "ip", "saddr", target, "drop"])
        return ok

    def _parse_handles_json(self, target: str, json_out: str) -> list[int]:
        """Parse `nft -j list chain` JSON output and return handles for target drop rules.

        D-05: JSON mode replaces fragile text parsing to reliably extract handles.
        The nft JSON schema wraps everything in {"nftables": [...]}.  Each rule
        element has the shape {"rule": {"handle": <int>, "expr": [...]}}.
        We look for rules that contain a match on ip saddr == target with a drop
        terminal expression.
        """
        handles: list[int] = []
        try:
            data = json.loads(json_out)
        except (json.JSONDecodeError, ValueError):
            logger.warning("nftables: JSON parse failed; output was: %r", json_out[:200])
            return handles

        items = data.get("nftables", []) if isinstance(data, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            rule = item.get("rule")
            if not isinstance(rule, dict):
                continue
            handle = rule.get("handle")
            if handle is None:
                continue
            exprs = rule.get("expr", [])
            if not isinstance(exprs, list):
                continue
            has_target_match = False
            has_drop = False
            for expr in exprs:
                if not isinstance(expr, dict):
                    continue
                # Drop terminal
                if "drop" in expr:
                    has_drop = True
                # Match on ip saddr
                match = expr.get("match", {})
                if not isinstance(match, dict):
                    continue
                left = match.get("left", {})
                right = match.get("right", "")
                if not isinstance(left, dict):
                    continue
                payload = left.get("payload", {})
                if (
                    isinstance(payload, dict)
                    and payload.get("protocol") in ("ip", "ip6")
                    and payload.get("field") == "saddr"
                    and str(right).strip() == target
                ):
                    has_target_match = True
            if has_target_match and has_drop:
                handles.append(int(handle))
        return handles

    def unblock(self, ip: str) -> bool:
        try:
            target = _validate_target_ip(ip)
        except Exception:
            return False
        # D-05: use JSON output mode for reliable handle parsing
        ok, out = self._run(["nft", "-j", "list", "chain", "inet", "filter", "input"])
        if not ok:
            return False

        handles = self._parse_handles_json(target, out)
        if not handles:
            return True

        for handle in handles:
            deleted, _ = self._run(
                ["nft", "delete", "rule", "inet", "filter", "input", "handle", str(handle)]
            )
            if not deleted:
                return False
        return True

    def list_rules(self) -> list[str]:
        # D-05: use JSON output mode for reliable IP extraction
        ok, out = self._run(["nft", "-j", "list", "chain", "inet", "filter", "input"])
        if not ok:
            return []
        rules: list[str] = []
        try:
            data = json.loads(out)
        except (json.JSONDecodeError, ValueError):
            return []
        items = data.get("nftables", []) if isinstance(data, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            rule = item.get("rule")
            if not isinstance(rule, dict):
                continue
            exprs = rule.get("expr", [])
            if not isinstance(exprs, list):
                continue
            has_drop = any("drop" in e for e in exprs if isinstance(e, dict))
            if not has_drop:
                continue
            for expr in exprs:
                if not isinstance(expr, dict):
                    continue
                match = expr.get("match", {})
                if not isinstance(match, dict):
                    continue
                left = match.get("left", {})
                right = match.get("right", "")
                if not isinstance(left, dict):
                    continue
                payload = left.get("payload", {})
                if (
                    isinstance(payload, dict)
                    and payload.get("protocol") in ("ip", "ip6")
                    and payload.get("field") == "saddr"
                ):
                    try:
                        rules.append(_validate_target_ip(str(right).strip()))
                    except Exception:
                        continue
        return sorted(set(rules))


@dataclass
class WebhookFirewallAdapter(FirewallAdapter):
    """Sends block/unblock commands via HTTP POST to an external webhook.

    Designed for integration with third-party firewalls, SOAR platforms, or
    cloud security groups that expose a REST API.

    IMPORTANT: This adapter is best-effort. Blocks enforced via webhook are NOT
    guaranteed to persist across restarts. If persistence is required, use UFW
    or nftables adapters instead. reconcile() is skipped for this adapter.
    """

    # Webhook adapters cannot query remote firewall state — stateless by design.
    supports_rule_query: bool = False

    webhook_url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 10

    def __post_init__(self) -> None:
        if self.webhook_url and not self.webhook_url.startswith("https://"):
            raise ValueError(
                f"WebhookFirewallAdapter: webhook_url must use HTTPS. "
                f"Got: {self.webhook_url!r}"
            )
        logger.warning(
            "WebhookFirewallAdapter: blocks are best-effort and NOT guaranteed to "
            "persist across restarts. Use UFW or nftables if persistence is required."
        )

    def block(self, ip: str, ttl_seconds: int | None = None) -> bool:
        try:
            target = _validate_target_ip(ip)
        except Exception:
            return False
        payload = {"action": "block", "target": target, "ttl_seconds": ttl_seconds}
        return self._post(payload)

    def unblock(self, ip: str) -> bool:
        try:
            target = _validate_target_ip(ip)
        except Exception:
            return False
        payload = {"action": "unblock", "target": target}
        return self._post(payload)

    def list_rules(self) -> list[str]:
        raise NotImplementedError(
            "WebhookFirewallAdapter cannot query remote firewall state. "
            "Check supports_rule_query before calling list_rules()."
        )

    def _post(self, payload: dict) -> bool:
        if not self.webhook_url:
            logger.warning("WebhookFirewallAdapter: no webhook_url configured")
            return False
        try:
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            headers.update(self.headers)
            req = urllib.request.Request(self.webhook_url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return resp.status < 400
        except Exception:
            logger.exception("Webhook POST failed to %s", self.webhook_url)
            return False
