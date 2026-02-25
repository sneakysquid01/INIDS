from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable
import ipaddress
import subprocess


class FirewallAdapter(Protocol):
    def block(self, target: str, ttl_seconds: int) -> bool:
        ...

    def unblock(self, target: str) -> bool:
        ...


def _validate_target_ip(target: str) -> str:
    ip = ipaddress.ip_address(target)
    return str(ip)


@dataclass
class MockFirewallAdapter:
    """In-memory firewall adapter for local demos/tests."""

    blocked_targets: dict[str, int] | None = None

    def __post_init__(self):
        if self.blocked_targets is None:
            self.blocked_targets = {}

    def block(self, target: str, ttl_seconds: int) -> bool:
        target = _validate_target_ip(target)
        self.blocked_targets[target] = ttl_seconds
        return True

    def unblock(self, target: str) -> bool:
        target = _validate_target_ip(target)
        return self.blocked_targets.pop(target, None) is not None


@dataclass
class UfwFirewallAdapter:
    """UFW-backed firewall adapter. Requires ufw and appropriate permissions."""

    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run

    def block(self, target: str, ttl_seconds: int) -> bool:
        # ttl_seconds is tracked by scheduler/cleanup; ufw itself doesn't do TTL.
        target = _validate_target_ip(target)
        result = self.run_cmd(["ufw", "deny", "from", target], capture_output=True, text=True)
        return result.returncode == 0

    def unblock(self, target: str) -> bool:
        target = _validate_target_ip(target)
        result = self.run_cmd(["ufw", "delete", "deny", "from", target], capture_output=True, text=True)
        return result.returncode == 0


@dataclass
class NftablesFirewallAdapter:
    """nftables-backed adapter that inserts/deletes source-IP drop rules in inet filter input."""

    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run

    def block(self, target: str, ttl_seconds: int) -> bool:
        target = _validate_target_ip(target)
        result = self.run_cmd(
            ["nft", "add", "rule", "inet", "filter", "input", "ip", "saddr", target, "drop"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def unblock(self, target: str) -> bool:
        target = _validate_target_ip(target)
        # Conservative approach: list ruleset and delete matching handles.
        list_result = self.run_cmd(["nft", "-a", "list", "chain", "inet", "filter", "input"], capture_output=True, text=True)
        if list_result.returncode != 0:
            return False

        handles: list[str] = []
        for line in list_result.stdout.splitlines():
            if f"ip saddr {target} drop" in line and "# handle" in line:
                handle = line.split("# handle")[-1].strip()
                handles.append(handle)

        ok = True
        for handle in handles:
            del_result = self.run_cmd(
                ["nft", "delete", "rule", "inet", "filter", "input", "handle", handle],
                capture_output=True,
                text=True,
            )
            ok = ok and del_result.returncode == 0
        return ok
