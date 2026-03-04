from __future__ import annotations

<<<<<<< ours
<<<<<<< ours
from collections import OrderedDict, deque
from dataclasses import dataclass
from threading import Lock
=======
from collections import deque, defaultdict
from dataclasses import dataclass
>>>>>>> theirs
=======
from collections import deque, defaultdict
from dataclasses import dataclass
>>>>>>> theirs
from time import time


@dataclass(frozen=True)
class RateLimitConfig:
    requests: int = 120
    window_seconds: int = 60
<<<<<<< ours
<<<<<<< ours
    max_keys: int = 10000
=======
>>>>>>> theirs
=======
>>>>>>> theirs


class InMemoryRateLimiter:
    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
<<<<<<< ours
<<<<<<< ours
        self._events: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = Lock()
=======
        self._events: dict[str, deque[float]] = defaultdict(deque)
>>>>>>> theirs
=======
        self._events: dict[str, deque[float]] = defaultdict(deque)
>>>>>>> theirs

    def allow(self, key: str) -> tuple[bool, int]:
        now = time()
        window_start = now - self.config.window_seconds
<<<<<<< ours
<<<<<<< ours
        with self._lock:
            q = self._events.get(key)
            if q is None:
                q = deque()
                self._events[key] = q
            else:
                self._events.move_to_end(key)

            while q and q[0] < window_start:
                q.popleft()

            if len(q) >= self.config.requests:
                retry_after = int(max(1, self.config.window_seconds - (now - q[0])))
                return False, retry_after

            q.append(now)

            # Bound memory growth by evicting least-recently-used keys.
            while len(self._events) > self.config.max_keys:
                self._events.popitem(last=False)

=======
=======
>>>>>>> theirs
        q = self._events[key]
        while q and q[0] < window_start:
            q.popleft()
        if len(q) >= self.config.requests:
            retry_after = int(max(1, self.config.window_seconds - (now - q[0])))
            return False, retry_after
        q.append(now)
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
        return True, 0
