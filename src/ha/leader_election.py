"""Leader election for multi-instance deployments.

Uses Redis SETNX-based leader election with TTL renewal so that only one
INIDS instance runs singleton tasks (e.g., TI feed refresh, expired-action
cleanup, model rotation).

B-05: Fail-closed on Redis failure.  Single-instance mode controlled by
INIDS_REDIS_REQUIRED=false (permits always-leader without Redis).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# B-05: observable gauge; 1 = elected leader, 0 = not leader / Redis down.
_leader_election_state: int = 0


def get_leader_election_state() -> int:
    """Return the last emitted leader_election_state gauge value (0 or 1)."""
    return _leader_election_state


class LeaderElection:
    """Simple Redis-based leader election with heartbeat renewal.

    Parameters
    ----------
    redis_client:
        A Redis client instance (or None triggers single-instance mode when
        INIDS_REDIS_REQUIRED=false; raises if INIDS_REDIS_REQUIRED is unset/true).
    key:
        Redis key used for the lock.
    ttl_seconds:
        Lock TTL — must be renewed before expiry.
    instance_id:
        Unique identifier for this instance.
    """

    def __init__(
        self,
        redis_client: Any = None,
        *,
        key: str = "inids:leader",
        ttl_seconds: int = 30,
        instance_id: str = "instance-1",
    ) -> None:
        self._redis = redis_client
        self._key = key
        self._ttl = ttl_seconds
        self._instance_id = instance_id
        # B-05: INIDS_REDIS_REQUIRED=false → single-instance always-leader mode.
        redis_required_env = os.environ.get("INIDS_REDIS_REQUIRED", "true").strip().lower()
        self._redis_required: bool = redis_required_env not in {"false", "0", "no"}
        # Fail-closed default: not leader until election confirms it.
        # Exception: single-instance mode (no Redis, not required).
        self._is_leader: bool = (redis_client is None and not self._redis_required)
        self._running = False
        self._thread: threading.Thread | None = None

    def _emit_metric(self, name: str, value: int) -> None:
        global _leader_election_state
        if name == "leader_election_state":
            _leader_election_state = value
        logger.debug("metric %s=%d instance=%s", name, value, self._instance_id)

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._redis is None:
            if not self._redis_required:
                self._is_leader = True
                self._emit_metric("leader_election_state", 1)
                logger.info("LeaderElection: no Redis — running as standalone leader")
            else:
                # Redis required but unavailable — fail-closed (not leader).
                self._is_leader = False
                self._emit_metric("leader_election_state", 0)
                logger.error(
                    "leader_election_redis_failure: assuming NOT leader — "
                    "prevention scheduler paused"
                )
            return
        self._running = True
        self._thread = threading.Thread(target=self._election_loop, daemon=True, name="leader-election")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._is_leader and self._redis is not None:
            try:
                self._redis.delete(self._key)
            except Exception:
                pass
        self._is_leader = False
        self._emit_metric("leader_election_state", 0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _election_loop(self) -> None:
        while self._running:
            try:
                acquired = self._redis.set(self._key, self._instance_id, nx=True, ex=self._ttl)
                if acquired:
                    self._is_leader = True
                    self._emit_metric("leader_election_state", 1)
                    logger.info("Became leader: %s", self._instance_id)
                else:
                    current = self._redis.get(self._key)
                    if isinstance(current, bytes):
                        current = current.decode()
                    if current == self._instance_id:
                        # Atomically renew only if we still own the key.
                        renewed = self._redis.set(
                            self._key, self._instance_id, xx=True, ex=self._ttl
                        )
                        self._is_leader = bool(renewed)
                    else:
                        self._is_leader = False
                    self._emit_metric("leader_election_state", 1 if self._is_leader else 0)
            except Exception:
                # B-05: fail-closed — Redis failure means we are NOT the leader.
                logger.error(
                    "leader_election_redis_failure: assuming NOT leader — "
                    "prevention scheduler paused"
                )
                self._is_leader = False
                self._emit_metric("leader_election_state", 0)

            time.sleep(max(1, int(self._ttl / 3.0)))

    def status(self) -> dict[str, Any]:
        return {
            "instance_id": self._instance_id,
            "is_leader": self._is_leader,
            "redis_connected": self._redis is not None,
            "redis_required": self._redis_required,
            "leader_election_state": _leader_election_state,
        }
