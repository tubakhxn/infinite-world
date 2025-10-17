from __future__ import annotations

import time


class HoldTimer:
    def __init__(self, hold_seconds: float = 1.0):
        self.hold_seconds = hold_seconds
        self._start: float | None = None
        self._progress: float = 0.0

    def update(self, active: bool) -> bool:
        now = time.time()
        if active:
            if self._start is None:
                self._start = now
            self._progress = min(1.0, (now - self._start) / self.hold_seconds)
            return self._progress >= 1.0
        else:
            self._start = None
            self._progress = 0.0
            return False

    @property
    def progress(self) -> float:
        return self._progress


class RateLimiter:
    """Simple wall-clock rate limiter to avoid updating too frequently."""
    def __init__(self, fps: float = 6.0):
        import time
        self.min_interval = 1.0 / max(0.1, fps)
        self._last = 0.0

    def ready(self) -> bool:
        import time
        now = time.time()
        if now - self._last >= self.min_interval:
            self._last = now
            return True
        return False
