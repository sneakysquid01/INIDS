"""Leader election for multi-instance deployments.

Uses Redis SETNX-based leader election with TTL renewal so that only one
INIDS instance runs singleton tasks (e.g., TI feed refresh, expired-action
cleanup, model rotation).

Falls back to "always leader" when Redis is unavailable — safe for single-
instance deployments.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class LeaderElection:
    """Simple Redis-based leader election with heartbeat renewal.

    Parameters
    ----------
    redis_client:
        A Redis client instance (or None for local-only mode).
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
        self._is_leader = redis_client is None  # default leader when no Redis
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._redis is None:
            self._is_leader = True
            logger.info("LeaderElection: no Redis — running as standalone leader")
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _election_loop(self) -> None:
        while self._running:
            try:
                acquired = self._redis.set(self._key, self._instance_id, nx=True, ex=self._ttl)
                if acquired:
                    self._is_leader = True
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
            except Exception:
                logger.exception("Leader election error")
                self._is_leader = False

            time.sleep(self._ttl // 3)

    def status(self) -> dict[str, Any]:
        return {
            "instance_id": self._instance_id,
            "is_leader": self._is_leader,
            "redis_connected": self._redis is not None,
        }
