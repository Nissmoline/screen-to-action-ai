"""Action rate limiting for live control."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic, sleep


@dataclass
class RateLimiter:
    """Permit one action at most every ``action_delay`` seconds."""

    action_delay: float
    _last_action_at: float | None = None

    @classmethod
    def from_action_delay(cls, action_delay: float) -> "RateLimiter":
        if action_delay < 0:
            raise ValueError("action_delay must be 0 or greater")
        return cls(action_delay=action_delay)

    def allow(self) -> bool:
        """Return True if an action can execute now and update internal state."""
        now = monotonic()
        if self._last_action_at is None:
            self._last_action_at = now
            return True
        if now - self._last_action_at >= self.action_delay:
            self._last_action_at = now
            return True
        return False

    def seconds_until_allowed(self) -> float:
        """Return seconds remaining before the next action is allowed."""
        if self._last_action_at is None:
            return 0.0
        elapsed = monotonic() - self._last_action_at
        return max(0.0, self.action_delay - elapsed)

    def wait_until_allowed(self) -> None:
        """Block until an action is allowed, then reserve the slot."""
        remaining = self.seconds_until_allowed()
        if remaining > 0:
            sleep(remaining)
        self._last_action_at = monotonic()
