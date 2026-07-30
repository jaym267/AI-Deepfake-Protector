"""Job store for analysis records.

In-memory for now. The interface is deliberately narrow (``create`` / ``get`` /
``update``) so it can be swapped for Redis or Postgres without touching the
routers.

NOTE: an in-process dict only works with a single worker. Before deploying with
``--workers > 1`` (or more than one container), this must move to a shared store,
and background execution must move to a real queue — see services/pipeline.py.

Retention is bounded two ways, because the TTL alone was not doing the job.
``_is_expired`` was only ever consulted when something looked a job up, and the
normal client polls once, reads its result and never returns — so nothing ever
triggered the check and every analysis leaked a record for the lifetime of the
process. ``purge_expired`` now runs on a timer from the app lifespan, and
``create`` enforces a hard ceiling for bursts that outrun the timer.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone

from .config import settings
from .schemas import AnalysisJob, AnalysisStatus, MediaKind

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, AnalysisJob] = {}
        self._lock = threading.Lock()

    def create(self, media_kind: MediaKind, filename: str | None) -> AnalysisJob:
        job = AnalysisJob(
            analysis_id=uuid.uuid4().hex,
            media_kind=media_kind,
            status=AnalysisStatus.PENDING,
            created_at=_now(),
            filename=filename,
        )
        with self._lock:
            # Bounded on write as well as swept on a timer. The sweep handles the
            # steady state; this handles a burst that arrives faster than the
            # sweep interval, so the store cannot grow without limit in between.
            # Oldest-first is right here (unlike in the rate limiter, where the
            # oldest entries are the ones worth keeping): the oldest analysis is
            # the one closest to expiring anyway.
            if len(self._jobs) >= settings.max_tracked_jobs:
                self._evict_locked()
            self._jobs[job.analysis_id] = job
        return job

    def _evict_locked(self) -> None:
        """Drop expired records, then oldest-first until under the ceiling.

        Caller holds the lock. Dropping a live record loses a result someone may
        still be polling for, so expired ones go first and the rest only if that
        was not enough.
        """
        for key in [k for k, v in self._jobs.items() if self._is_expired(v)]:
            self._jobs.pop(key, None)

        overflow = len(self._jobs) - settings.max_tracked_jobs + 1
        if overflow <= 0:
            return

        by_age = sorted(self._jobs.items(), key=lambda kv: kv[1].created_at)
        for key, _ in by_age[:overflow]:
            self._jobs.pop(key, None)
        logger.warning(
            "Job store at capacity (%d); evicted %d live record(s) early",
            settings.max_tracked_jobs,
            overflow,
        )

    def get(self, analysis_id: str) -> AnalysisJob | None:
        with self._lock:
            job = self._jobs.get(analysis_id)
        if job is None:
            return None
        if self._is_expired(job):
            self.delete(analysis_id)
            return None
        return job

    def update(self, job: AnalysisJob) -> None:
        with self._lock:
            self._jobs[job.analysis_id] = job

    def delete(self, analysis_id: str) -> None:
        with self._lock:
            self._jobs.pop(analysis_id, None)

    def purge_expired(self) -> int:
        """Drop results past their TTL. The media itself is already long gone."""
        with self._lock:
            stale = [k for k, v in self._jobs.items() if self._is_expired(v)]
            for key in stale:
                self._jobs.pop(key, None)
        return len(stale)

    @staticmethod
    def _is_expired(job: AnalysisJob) -> bool:
        cutoff = timedelta(seconds=settings.result_ttl_seconds)
        return _now() - job.created_at > cutoff


store = JobStore()
