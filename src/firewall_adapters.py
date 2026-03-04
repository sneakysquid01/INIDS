from __future__ import annotations

from dataclasses import dataclass
<<<<<<< ours
<<<<<<< ours
from threading import Lock
from typing import Callable, Protocol
=======
from typing import Protocol, Callable
>>>>>>> theirs
=======
from typing import Protocol, Callable
>>>>>>> theirs
import ipaddress
import subprocess


class FirewallAdapter(Protocol):
    def block(self, target: str, ttl_seconds: int) -> bool:
        ...

    def unblock(self, target: str) -> bool:
        ...

<<<<<<< ours
<<<<<<< ours
    def list_rules(self) -> list[str]:
        ...

=======
>>>>>>> theirs
=======
>>>>>>> theirs

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
<<<<<<< ours
<<<<<<< ours
        self._lock = Lock()

    def block(self, target: str, ttl_seconds: int) -> bool:
        target = _validate_target_ip(target)
        with self._lock:
            self.blocked_targets[target] = ttl_seconds
=======
=======
>>>>>>> theirs

    def block(self, target: str, ttl_seconds: int) -> bool:
        target = _validate_target_ip(target)
        self.blocked_targets[target] = ttl_seconds
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
        return True

    def unblock(self, target: str) -> bool:
        target = _validate_target_ip(target)
<<<<<<< ours
<<<<<<< ours
        with self._lock:
            return self.blocked_targets.pop(target, None) is not None

    def list_rules(self) -> list[str]:
        with self._lock:
            return sorted(self.blocked_targets.keys())


@dataclass
class IptablesFirewallAdapter:
    """iptables-backed adapter for Linux environments."""

    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run

    def block(self, target: str, ttl_seconds: int) -> bool:
        target = _validate_target_ip(target)
        result = self.run_cmd(
            ["iptables", "-I", "INPUT", "-s", target, "-j", "DROP"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def unblock(self, target: str) -> bool:
        target = _validate_target_ip(target)
        result = self.run_cmd(
            ["iptables", "-D", "INPUT", "-s", target, "-j", "DROP"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def list_rules(self) -> list[str]:
        result = self.run_cmd(["iptables", "-S", "INPUT"], capture_output=True, text=True)
        if result.returncode != 0:
            return []
        rules: list[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("-A INPUT") and "-j DROP" in line and "-s " in line:
                src = line.split("-s ", 1)[1].split()[0]
                try:
                    rules.append(_validate_target_ip(src))
                except ValueError:
                    continue
        return sorted(set(rules))


@dataclass
class PfFirewallAdapter:
    """pfctl-backed adapter for BSD/macOS style packet filters."""

    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run
    table_name: str = "inids_block"

    def block(self, target: str, ttl_seconds: int) -> bool:
        target = _validate_target_ip(target)
        result = self.run_cmd(
            ["pfctl", "-t", self.table_name, "-T", "add", target],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def unblock(self, target: str) -> bool:
        target = _validate_target_ip(target)
        result = self.run_cmd(
            ["pfctl", "-t", self.table_name, "-T", "delete", target],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def list_rules(self) -> list[str]:
        result = self.run_cmd(
            ["pfctl", "-t", self.table_name, "-T", "show"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return []
        ips: list[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ips.append(_validate_target_ip(line))
            except ValueError:
                continue
        return sorted(set(ips))
=======
        return self.blocked_targets.pop(target, None) is not None
>>>>>>> theirs
=======
        return self.blocked_targets.pop(target, None) is not None
>>>>>>> theirs


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

<<<<<<< ours
<<<<<<< ours
    def list_rules(self) -> list[str]:
        result = self.run_cmd(["ufw", "status", "numbered"], capture_output=True, text=True)
        if result.returncode != 0:
            return []
        rules: list[str] = []
        for line in result.stdout.splitlines():
            if "DENY IN" not in line:
                continue
            for token in line.split():
                try:
                    rules.append(_validate_target_ip(token))
                    break
                except ValueError:
                    continue
        return sorted(set(rules))

=======
>>>>>>> theirs
=======
>>>>>>> theirs

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
<<<<<<< ours
<<<<<<< ours

    def list_rules(self) -> list[str]:
        list_result = self.run_cmd(["nft", "-a", "list", "chain", "inet", "filter", "input"], capture_output=True, text=True)
        if list_result.returncode != 0:
            return []
        rules: list[str] = []
        for line in list_result.stdout.splitlines():
            line = line.strip()
            if "ip saddr " not in line or " drop" not in line:
                continue
            try:
                candidate = line.split("ip saddr ", 1)[1].split()[0]
                rules.append(_validate_target_ip(candidate))
            except (IndexError, ValueError):
                continue
        return sorted(set(rules))
=======
>>>>>>> theirs
=======
>>>>>>> theirs
