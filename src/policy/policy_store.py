"""Policy versioning and audit trail.

Extends the base PolicyConfig concept with versioned snapshots so that every
policy change is recorded with who/when/what for compliance and forensics.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PolicyVersion:
    """An immutable snapshot of a policy configuration at a point in time."""
    version: int
    config: dict[str, Any]
    changed_by: str = "system"
    reason: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicyStore:
    """Append-only policy version store with rollback support.

    Every call to ``update()`` creates a new version.  ``rollback()``
    reverts to a previous version (also creating a new version entry for
    the audit trail).
    """

    def __init__(self, initial_config: dict[str, Any] | None = None) -> None:
        self._versions: list[PolicyVersion] = []
        self._lock = Lock()
        if initial_config:
            self._versions.append(PolicyVersion(
                version=1,
                config=copy.deepcopy(initial_config),
                changed_by="system",
                reason="initial",
                timestamp=time.time(),
            ))

    @property
    def current(self) -> PolicyVersion | None:
        with self._lock:
            return self._versions[-1] if self._versions else None

    @property
    def current_config(self) -> dict[str, Any]:
        v = self.current
        return copy.deepcopy(v.config) if v else {}

    def update(self, config: dict[str, Any], *, changed_by: str = "system", reason: str = "") -> PolicyVersion:
        with self._lock:
            ver = len(self._versions) + 1
            pv = PolicyVersion(
                version=ver,
                config=copy.deepcopy(config),
                changed_by=changed_by,
                reason=reason,
                timestamp=time.time(),
            )
            self._versions.append(pv)
        logger.info("Policy updated to v%d by %s: %s", ver, changed_by, reason)
        return pv

    def rollback(self, to_version: int, *, changed_by: str = "system") -> PolicyVersion | None:
        with self._lock:
            target = None
            for v in self._versions:
                if v.version == to_version:
                    target = v
                    break
            if target is None:
                return None

            ver = len(self._versions) + 1
            pv = PolicyVersion(
                version=ver,
                config=copy.deepcopy(target.config),
                changed_by=changed_by,
                reason=f"rollback_to_v{to_version}",
                timestamp=time.time(),
            )
            self._versions.append(pv)
        logger.info("Policy rolled back to v%d (now v%d) by %s", to_version, ver, changed_by)
        return pv

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return [v.to_dict() for v in reversed(self._versions[:limit])]

    def get_version(self, version: int) -> PolicyVersion | None:
        with self._lock:
            for v in self._versions:
                if v.version == version:
                    return v
        return None
