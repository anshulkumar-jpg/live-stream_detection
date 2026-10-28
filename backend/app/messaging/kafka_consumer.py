"""Consumer-side helpers: idempotency tracking for at-least-once delivery."""
from __future__ import annotations

from collections import OrderedDict


class SeenCache:
    """Bounded recently-seen set for in-process de-duplication (DB unique constraint is the durable guard)."""

    def __init__(self, size: int = 50000):
        self.size, self._d = size, OrderedDict()

    def seen(self, key: str) -> bool:
        if key in self._d:
            self._d.move_to_end(key)
            return True
        self._d[key] = None
        if len(self._d) > self.size:
            self._d.popitem(last=False)
        return False
