"""Per-client rate limiting for a free, unauthenticated public endpoint.

Closes D5 item 3.

Why this is a security control and not a nicety: `/analyze/*` accepts a 25MB
upload and runs model inference on it. Without a limit, one client can occupy the
inference path indefinitely at essentially no cost to itself, which is both a
denial-of-service against everyone else and — because the endpoint returns a
score — a free evaluation harness for tuning a fake until it passes. D3 keeps the
numbers hidden; this keeps the number of attempts finite.

**Implemented as middleware, not a per-route dependency, on purpose.** A
dependency has to be remembered on every new route. Middleware cannot be
forgotten, and a route added in step 4 or 7 is covered the moment it exists.

Two windows per client, because one is always wrong:

* a **burst** window stops a script hammering the endpoint,
* a **sustained** window stops a slow drip that a burst limit alone would miss.

Reads (polling for a result) get their own, much larger allowance: the frontend
polls once a second by design (D1), so a limit tuned for uploads would break
normal use.

Known limits of this implementation, both matching the in-process job store
(`storage.py`) and both needing to change together before horizontal scaling:

* **State is per process.** Two workers mean two independent allowances. Redis
  is the fix; the interface here is small enough to swap.
* **Restarts clear it.** Acceptable for a limit measured in minutes and hours.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import settings

logger = logging.getLogger(__name__)

#: Stop the client dict growing without bound under a spoofed-IP flood. At this
#: size the limiter itself becomes a memory target, so old entries are dropped.
MAX_TRACKED_CLIENTS = 20_000


@dataclass(frozen=True)
class Rule:
    limit: int
    window_seconds: int
    name: str


class SlidingWindowLimiter:
    """Sliding-window counter, keyed by client.

    A sliding window rather than a fixed one: a fixed window lets a client send
    its full allowance in the last second of one window and again in the first
    second of the next, which is twice the intended rate exactly when it hurts.
    """

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str, rule: Rule, now: float | None = None) -> float | None:
        """Record a hit. Returns None if allowed, else seconds until retry."""
        now = time.monotonic() if now is None else now
        cutoff = now - rule.window_seconds
        key = (client, rule.name)

        with self._lock:
            if len(self._hits) > MAX_TRACKED_CLIENTS:
                self._evict(now)

            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= rule.limit:
                # Not counted: a rejected request must not extend the window, or
                # a client that keeps retrying locks itself out indefinitely.
                return max(hits[0] + rule.window_seconds - now, 0.0)

            hits.append(now)
            return None

    def _evict(self, now: float) -> None:
        """Drop entries with no live hits. Caller holds the lock."""
        stale = [key for key, hits in self._hits.items() if not hits or hits[-1] <= now - 3600]
        for key in stale:
            del self._hits[key]
        if not stale:
            # Everything is live: a real flood from many distinct IPs. Dropping
            # the oldest is wrong (it frees exactly the attacker's entries) but
            # unbounded growth is worse. Log it — this is worth alerting on.
            logger.warning(
                "Rate limiter tracking %d live clients; possible distributed flood",
                len(self._hits),
            )
            self._hits.clear()

    def reset(self) -> None:
        """Test hook."""
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()


def client_key(request: Request) -> str:
    """Identify the caller.

    ``X-Forwarded-For`` is only consulted when ``trust_forwarded_for`` is on,
    because any client can send that header. Trusting it by default would mean
    the limiter could be bypassed with one line of curl. Enable it only behind a
    proxy that overwrites the header — and note that the *leftmost* entry is
    client-supplied, so the rightmost is taken instead.
    """
    if settings.trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def rules_for(method: str, path: str) -> list[Rule]:
    """Which limits apply. Expensive writes are capped far harder than reads."""
    if method == "POST" and (path.startswith("/analyze") or path.startswith("/report")):
        return [
            Rule(settings.analyze_burst_limit, settings.analyze_burst_window, "analyze_burst"),
            Rule(
                settings.analyze_sustained_limit,
                settings.analyze_sustained_window,
                "analyze_sustained",
            ),
        ]
    # Polling and metadata. Generous: the client polls ~1/s while a job runs.
    return [Rule(settings.read_limit, settings.read_window, "read")]


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.rate_limit_enabled:
            return await call_next(request)

        client = client_key(request)
        for rule in rules_for(request.method, request.url.path):
            retry_after = limiter.check(client, rule)
            if retry_after is not None:
                logger.info("Rate limited %s on %s (%s)", client, request.url.path, rule.name)
                seconds = max(int(retry_after) + 1, 1)
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            "Too many requests. This is a free service with "
                            f"limited capacity — please wait {seconds} seconds "
                            "and try again."
                        )
                    },
                    # Standard, and lets a well-behaved client back off properly
                    # instead of retrying into the same wall.
                    headers={"Retry-After": str(seconds)},
                )

        return await call_next(request)
