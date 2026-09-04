"""Bearer token generation/validation + sliding-window rate limiting.

Two pieces:

1. ``generate_token()`` produces a URL-safe random token at verifier startup.
   The token is written to ``verifier.url`` (mode 0o600 on POSIX) so that
   only callers who can read the file know the token. The agent reads it
   and includes it in the ``Authorization: Bearer <token>`` header.

2. ``RateLimiter`` is a sliding-window counter, keyed by token. Used by
   the FastAPI middleware to defend against the leaderboard-probing attack
   (repeated crafted submissions to reverse-engineer ground truth).
"""

from __future__ import annotations

import secrets
import time
from collections import deque
from threading import Lock
from typing import Deque, Dict


def generate_token(*, n_bytes: int = 32) -> str:
    """Generate a URL-safe random token for the bearer-auth header.

    Re-rolls until the token does not start with ``-`` or ``+`` so it is
    safe to pass as a positional value on any CLI: argparse treats a
    leading ``-`` as a flag prefix and errors with ``"expected one
    argument"`` (verifier spawn was failing ~1.5% of the time before
    this guard). URL-safe base64 alphabet is ``[A-Za-z0-9_-]``, so the
    expected reroll rate is ~1/64 — negligible.
    """
    while True:
        t = secrets.token_urlsafe(n_bytes)
        if not t.startswith(("-", "+")):
            return t


def constant_time_token_eq(a: str, b: str) -> bool:
    """Constant-time string comparison for bearer-token validation.

    Defends against timing side-channels even though our threat model
    is "single trust domain on a host" — cheap to do right.
    """
    return secrets.compare_digest(a, b)


# Sliding-window rate limiter


class RateLimiter:
    """Per-token sliding-window submission rate limiter.

    A token may make up to ``max_requests`` requests within any rolling
    ``window_sec``. The 61st request in a 60-second window returns
    429 with a ``Retry-After`` header.

    Implementation: a deque of timestamps per token. On each check we
    evict timestamps older than ``window_sec`` from the left, then test
    ``len(deque) < max_requests``. O(1) amortized per check.
    """

    def __init__(self, *, max_requests: int = 60, window_sec: float = 60.0) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if window_sec <= 0:
            raise ValueError("window_sec must be positive")
        self._max = max_requests
        self._window = window_sec
        self._deques: Dict[str, Deque[float]] = {}
        self._lock = Lock()

    def check_and_record(self, key: str) -> tuple[bool, float]:
        """Record a request and return ``(allowed, retry_after_sec)``.

        ``retry_after_sec`` is 0.0 when the request is allowed; otherwise
        the time until the oldest in-window request falls out of the
        window (the earliest moment a new request would be admitted).
        """
        now = time.time()
        with self._lock:
            dq = self._deques.get(key)
            if dq is None:
                dq = deque()
                self._deques[key] = dq

            # Evict timestamps older than window_sec.
            cutoff = now - self._window
            while dq and dq[0] < cutoff:
                dq.popleft()

            if len(dq) < self._max:
                dq.append(now)
                return True, 0.0

            # Over budget. Compute retry-after from the oldest in-window stamp.
            retry_after = max(0.0, (dq[0] + self._window) - now)
            return False, retry_after

    def reset(self, key: str | None = None) -> None:
        """Clear state for ``key`` (or all keys if ``None``). Used in tests."""
        with self._lock:
            if key is None:
                self._deques.clear()
            else:
                self._deques.pop(key, None)
