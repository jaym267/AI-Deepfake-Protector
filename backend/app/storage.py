"""Job store for analysis records.

In-memory for now. The interface is deliberately narrow (``create`` / ``get`` /
``update``) so it can be swapped for Redis or Postgres without touching the
routers.

NOTE: an in-process dict only works with a single worker. Before deploying with
``--workers > 1`` (or more than one container), this must move to a shared store,
and background execution must move to a real queue — see services/pipeline.py.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone

from .config import settings
from .schemas import AnalysisJob, AnalysisStatus, MediaKind


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
            self._jobs[job.analysis_id] = job
        return job

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
