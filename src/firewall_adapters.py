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

    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run

    def _run(self, args: list[str]) -> tuple[bool, str]:
        try:
            result = self.run_cmd(args, capture_output=True, text=True, timeout=5, check=False)
            return result.returncode == 0, str(getattr(result, "stdout", "") or "")
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
        ok, _ = self._run(["ufw", "deny", "from", target])
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

    def unblock(self, ip: str) -> bool:
        try:
            target = _validate_target_ip(ip)
        except Exception:
            return False
        ok, out = self._run(["nft", "-a", "list", "chain", "inet", "filter", "input"])
        if not ok:
            return False

        handles: list[str] = []
        for line in out.splitlines():
            if f"ip saddr {target} drop" in line and "# handle" in line:
                handles.append(line.split("# handle")[-1].strip())

        if not handles:
            return True

        for handle in handles:
            deleted, _ = self._run(["nft", "delete", "rule", "inet", "filter", "input", "handle", handle])
            if not deleted:
                return False
        return True

    def list_rules(self) -> list[str]:
        ok, out = self._run(["nft", "-a", "list", "chain", "inet", "filter", "input"])
        if not ok:
            return []
        rules: list[str] = []
        for line in out.splitlines():
            marker = "ip saddr "
            if marker not in line or " drop" not in line:
                continue
            fragment = line.split(marker, 1)[1]
            ip_part = fragment.split(" ", 1)[0].strip()
            try:
                rules.append(_validate_target_ip(ip_part))
            except Exception:
                continue
        return sorted(set(rules))


@dataclass
class WebhookFirewallAdapter(FirewallAdapter):
    """Sends block/unblock commands via HTTP POST to an external webhook.

    Designed for integration with third-party firewalls, SOAR platforms, or
    cloud security groups that expose a REST API.
    """

    webhook_url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 10
    blocked_targets: dict[str, int] = field(default_factory=dict)

    def block(self, ip: str, ttl_seconds: int | None = None) -> bool:
        try:
            target = _validate_target_ip(ip)
        except Exception:
            return False
        payload = {"action": "block", "target": target, "ttl_seconds": ttl_seconds}
        ok = self._post(payload)
        if ok:
            self.blocked_targets[target] = int(ttl_seconds or 0)
        return ok

    def unblock(self, ip: str) -> bool:
        try:
            target = _validate_target_ip(ip)
        except Exception:
            return False
        payload = {"action": "unblock", "target": target}
        ok = self._post(payload)
        if ok:
            self.blocked_targets.pop(target, None)
        return ok

    def list_rules(self) -> list[str]:
        return sorted(self.blocked_targets.keys())

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
