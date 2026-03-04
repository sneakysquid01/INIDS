from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from time import time


@dataclass(frozen=True)
class RateLimitConfig:
    requests: int = 120
    window_seconds: int = 60


class InMemoryRateLimiter:
    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> tuple[bool, int]:
        now = time()
        window_start = now - self.config.window_seconds
        q = self._events[key]
        while q and q[0] < window_start:
            q.popleft()
        if len(q) >= self.config.requests:
            retry_after = int(max(1, self.config.window_seconds - (now - q[0])))
            return False, retry_after
        q.append(now)
        return True, 0
