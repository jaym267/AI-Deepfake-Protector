"""Retention of analysis records.

The privacy commitment (D2) is about the uploaded media, which is deleted the
moment analysis finishes. This is about the thing that *is* kept — the derived
result — and the fact that it has to actually go away when its TTL is up.
"""

from __future__ import annotations

from datetime import timedelta

from app.config import settings
from app.schemas import MediaKind
from app.storage import JobStore


def _aged(store: JobStore, job, seconds: float):
    """Backdate a record so it reads as expired without sleeping for a day."""
    job.created_at = job.created_at - timedelta(seconds=seconds)
    store.update(job)
    return job


def test_purge_expired_drops_records_nothing_ever_looks_up():
    """The bug this closes: expiry was only evaluated inside `get`, and the normal
    client polls until the result is ready, reads it, and never returns. Nothing
    triggered the check, so every analysis retained its record — including its
    InternalScores — for the lifetime of the process.
    """
    store = JobStore()
    fresh = store.create(MediaKind.IMAGE, "fresh.png")
    stale = store.create(MediaKind.IMAGE, "stale.png")
    _aged(store, stale, settings.result_ttl_seconds + 60)

    assert store.purge_expired() == 1
    assert store.get(stale.analysis_id) is None
    assert store.get(fresh.analysis_id) is not None


def test_purge_is_idempotent():
    store = JobStore()
    _aged(store, store.create(MediaKind.AUDIO, "a.wav"), settings.result_ttl_seconds + 1)
    assert store.purge_expired() == 1
    assert store.purge_expired() == 0


def test_store_is_bounded_even_without_a_sweep(monkeypatch):
    """The TTL bounds how long a record lives; the ceiling bounds how many can
    exist at once, so a burst arriving faster than the sweep interval cannot grow
    the store without limit in between."""
    monkeypatch.setattr(settings, "max_tracked_jobs", 50)
    store = JobStore()
    for index in range(200):
        store.create(MediaKind.IMAGE, f"{index}.png")
    assert len(store._jobs) <= settings.max_tracked_jobs


def test_eviction_prefers_expired_records_over_live_ones(monkeypatch):
    """Dropping a live record loses a result someone may still be polling for, so
    expired ones go first and live ones only if that was not enough."""
    monkeypatch.setattr(settings, "max_tracked_jobs", 10)
    store = JobStore()

    expired = [store.create(MediaKind.IMAGE, f"old-{i}.png") for i in range(9)]
    for job in expired:
        _aged(store, job, settings.result_ttl_seconds + 60)
    live = store.create(MediaKind.IMAGE, "live.png")

    # Tips the store over its ceiling; the nine expired records are reclaimable.
    store.create(MediaKind.IMAGE, "trigger.png")

    assert store.get(live.analysis_id) is not None
    for job in expired:
        assert store.get(job.analysis_id) is None
